#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# MIT License
"""Fused All-to-All + GEMM in a single Triton persistent kernel.

    C[M, N] = A_full[M, K] @ B[K, N]

where A_full = [A_shard_0 | A_shard_1 | ... | A_shard_{W-1}] along K.
Each GPU holds one K-shard of A and the full B.

    torchrun --nnodes=1 --nproc_per_node=<gpus> a2a_gemm_fused.py
"""

import argparse
import gc
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

import mori.allocator  # noqa: F401
from mori.cco import Communicator
from mori.ir.triton import cco

PER_RANK_VMM = 1 << 30
COMM_WGS = 4  # number of persistent programs dedicated to communication


# ---------------------------------------------------------------------------
# Fused A2A + GEMM kernel
# ---------------------------------------------------------------------------

@triton.jit(do_not_specialize=[
    "num_k_shards", "my_rank", "generation",
    "ready_byte_off",
])
def a2a_gemm_kernel(
    # communication
    win,              # ccoWindow for recv_A buffer (+ ready flags at tail)
    local_a_ptr,      # [M, K_shard] bf16 — this rank's A shard
    a_shard_bytes,    # M * K_shard * 2 bytes per shard
    num_k_shards,     # = world_size
    my_rank,
    generation,
    ready_byte_off,   # byte offset of ready flags region in the window
    # GEMM
    recv_a_ptr,       # local data_ptr of recv_A buffer (for compute WGs)
    b_ptr,            # [K, N] bf16
    c_ptr,            # [M, N] bf16 output
    M, N, K_SHARD,    # constexpr dimensions
    stride_a,         # K (full) for recv_a layout, or K_shard for local
    stride_b,         # N
    stride_c,         # N
    # constexpr
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    COMM_WGS: tl.constexpr,
):
    pid = tl.program_id(0)
    is_comm = pid < COMM_WGS

    # ═══ Communication programs: push local_a to all peers ═══
    if is_comm:
        comm_slot = pid
        src = local_a_ptr.to(tl.pointer_type(tl.bfloat16), bitcast=True)
        shard_elems = tl.cast(a_shard_bytes, tl.int64) // 2  # bf16 count
        stride = tl.cast(COMM_WGS, tl.int64) * 256  # elems per stride

        for step in range(1, WORLD_SIZE):
            dst_rank = (tl.cast(my_rank, tl.int32) - step + WORLD_SIZE) % WORLD_SIZE

            # Destination: peer's recv buffer at [my_rank * shard_bytes]
            dst_off = tl.cast(my_rank, tl.int64) * tl.cast(a_shard_bytes, tl.int64)
            dst_base = cco.Window.lsa_ptr(win, dst_rank, dst_off)
            dst = dst_base.to(tl.pointer_type(tl.bfloat16), bitcast=True)

            # Strided copy
            base_idx = tl.cast(comm_slot, tl.int64) * 256 + tl.arange(0, 256).to(tl.int64)
            for start in range(0, tl.cast(shard_elems, tl.int32), tl.cast(stride, tl.int32)):
                idx = tl.cast(start, tl.int64) + base_idx
                mask = idx < shard_elems
                tl.store(dst + idx, tl.load(src + idx, mask=mask), mask=mask)

            # Signal ready: atomic_add peer's flag for my_rank
            rbo = tl.cast(ready_byte_off, tl.int64)
            flag_addr = cco.Window.lsa_ptr(win, dst_rank,
                                           rbo + tl.cast(my_rank, tl.int64) * 4)
            flag_p = flag_addr.to(tl.pointer_type(tl.int32), bitcast=True)
            if comm_slot == 0:
                tl.atomic_add(flag_p, 1)
        return

    # ═══ Compute programs: tiled GEMM with per-shard barriers ═══
    compute_pid = pid - COMM_WGS
    num_compute = NUM_PROGRAMS - COMM_WGS

    # Tile mapping: 2D grid over M and N
    num_m_tiles = (M + BLOCK_M - 1) // BLOCK_M
    num_n_tiles = (N + BLOCK_N - 1) // BLOCK_N
    total_tiles = num_m_tiles * num_n_tiles

    for tile_id in range(compute_pid, total_tiles, num_compute):
        m_tile = tile_id // num_n_tiles
        n_tile = tile_id % num_n_tiles

        row = m_tile * BLOCK_M
        col = n_tile * BLOCK_N

        # Accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        recv_a = recv_a_ptr.to(tl.pointer_type(tl.bfloat16), bitcast=True)
        b = b_ptr.to(tl.pointer_type(tl.bfloat16), bitcast=True)

        # Iterate over K shards (one per rank)
        for shard in range(WORLD_SIZE):
            # Wait for this shard (skip local — already in recv buffer)
            if shard != tl.cast(my_rank, tl.int32):
                # Poll the ready flag via local VMM VA (not LSA).
                # recv_a_ptr is uint64 addr of recv_buf. Flag region starts
                # at ready_byte_off bytes from recv_buf start.
                flag_byte = recv_a_ptr + tl.cast(ready_byte_off, tl.int64) + shard.to(tl.int64) * 4
                flag_p = flag_byte.to(tl.pointer_type(tl.int32), bitcast=True)
                while tl.load(flag_p) < tl.cast(generation, tl.int32):
                    pass

            # K offset in the full K dimension
            k_base = shard * K_SHARD

            # GEMM tile: A[row:row+BM, shard*K_shard:(shard+1)*K_shard] @ B[k_base:k_base+K_shard, col:col+BN]
            for k_off in range(0, K_SHARD, BLOCK_K):
                k = k_base + k_off

                # Load A tile: recv_a layout is [WORLD_SIZE, M, K_SHARD]
                # A[shard, row:row+BM, k_off:k_off+BK]
                a_offs_m = row + tl.arange(0, BLOCK_M)
                a_offs_k = k_off + tl.arange(0, BLOCK_K)
                a_idx = shard.to(tl.int64) * tl.cast(M * K_SHARD, tl.int64) + \
                        a_offs_m[:, None].to(tl.int64) * tl.cast(K_SHARD, tl.int64) + \
                        a_offs_k[None, :].to(tl.int64)
                a_mask = (a_offs_m[:, None] < M) & (a_offs_k[None, :] < K_SHARD)
                a_tile = tl.load(recv_a + a_idx, mask=a_mask, other=0.0).to(tl.bfloat16)

                # Load B tile: B[k, col:col+BN]
                b_offs_k = k + tl.arange(0, BLOCK_K)
                b_offs_n = col + tl.arange(0, BLOCK_N)
                b_idx = b_offs_k[:, None].to(tl.int64) * tl.cast(stride_b, tl.int64) + \
                        b_offs_n[None, :].to(tl.int64)
                b_mask = (b_offs_k[:, None] < tl.cast(num_k_shards, tl.int32) * K_SHARD) & \
                         (b_offs_n[None, :] < N)
                b_tile = tl.load(b + b_idx, mask=b_mask, other=0.0).to(tl.bfloat16)

                acc += tl.dot(a_tile, b_tile)

        # Store C tile
        c = c_ptr.to(tl.pointer_type(tl.bfloat16), bitcast=True)
        c_offs_m = row + tl.arange(0, BLOCK_M)
        c_offs_n = col + tl.arange(0, BLOCK_N)
        c_idx = c_offs_m[:, None].to(tl.int64) * tl.cast(stride_c, tl.int64) + \
                c_offs_n[None, :].to(tl.int64)
        c_mask = (c_offs_m[:, None] < M) & (c_offs_n[None, :] < N)
        tl.store(c + c_idx, acc.to(tl.bfloat16), mask=c_mask)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m", type=int, default=512)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--k-shard", type=int, default=128, help="K per rank")
    p.add_argument("--programs", type=int, default=128)
    args = p.parse_args()

    for key, val in (
        ("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0"),
        ("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "29500"),
    ):
        os.environ.setdefault(key, val)

    rank = int(os.environ["RANK"])
    ws = int(os.environ["WORLD_SIZE"])
    lr = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("gloo")
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)

    M, N, K_SHARD = args.m, args.n, args.k_shard
    K = K_SHARD * ws
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    NUM_PROGRAMS = args.programs

    if rank == 0:
        print(f"Fused A2A GEMM  world={ws}  M={M} N={N} K={K} (K_shard={K_SHARD})  "
              f"programs={NUM_PROGRAMS} (comm={COMM_WGS}, compute={NUM_PROGRAMS - COMM_WGS})")

    # Deterministic data
    torch.manual_seed(42)
    # Each rank has one K-shard of A
    A_full = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    B = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
    A_local = A_full[:, rank * K_SHARD : (rank + 1) * K_SHARD].contiguous()

    # recv_A buffer: [ws, M, K_SHARD] — holds all gathered shards
    # Plus ready flags at the tail: [ws] int32
    recv_elems = ws * M * K_SHARD
    flag_elems_bf16 = ws * 2 + 256  # int32 → bf16 equivalent, with padding
    total_bf16 = recv_elems + flag_elems_bf16
    ready_byte_off = recv_elems * 2  # byte offset of flags
    a_shard_bytes = M * K_SHARD * 2

    symm_mem.set_backend("MORI")
    recv_buf = symm_mem.empty(total_bf16, dtype=torch.bfloat16, device=dev)
    recv_buf.zero_()

    # Pre-fill local shard in recv buffer (no communication needed for self)
    recv_buf[rank * M * K_SHARD : (rank + 1) * M * K_SHARD] = A_local.reshape(-1)
    torch.cuda.synchronize()

    C = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)

    ct = [Communicator.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(ct, src=0)

    with Communicator.init(ws, rank, ct[0], per_rank_vmm=PER_RANK_VMM) as comm:
        win = comm.register_external_window(recv_buf.data_ptr(), recv_buf.nbytes)
        el = cco.get_extern_libs()

        # Serialize JIT compilation
        for cr in range(ws):
            if rank == cr:
                a2a_gemm_kernel.warmup(
                    win.handle, A_local.data_ptr(), a_shard_bytes, ws, rank, 1,
                    ready_byte_off,
                    recv_buf.data_ptr(), B.data_ptr(), C.data_ptr(),
                    M, N, K_SHARD,
                    K_SHARD, N, N,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                    WORLD_SIZE=ws, NUM_PROGRAMS=NUM_PROGRAMS, COMM_WGS=COMM_WGS,
                    grid=(NUM_PROGRAMS,), extern_libs=el,
                )
            dist.barrier()
        comm.barrier()

        # ONE kernel launch: comm + GEMM fused
        a2a_gemm_kernel[(NUM_PROGRAMS,)](
            win.handle, A_local.data_ptr(), a_shard_bytes, ws, rank, 1,
            ready_byte_off,
            recv_buf.data_ptr(), B.data_ptr(), C.data_ptr(),
            M, N, K_SHARD,
            K_SHARD, N, N,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            WORLD_SIZE=ws, NUM_PROGRAMS=NUM_PROGRAMS, COMM_WGS=COMM_WGS,
            extern_libs=el,
        )
        torch.cuda.synchronize()
        comm.barrier()

        # Verify
        ref = (A_full @ B).to(torch.bfloat16)
        diff = (C - ref).abs().max().item()
        rel = diff / (ref.abs().max().item() + 1e-8)
        ok = rel < 0.05

        if rank == 0:
            print(f"max abs diff: {diff:.6f}  max rel diff: {rel:.6f}")
            print(f"correctness: {'OK' if ok else 'FAILED'}")
        comm.barrier()

    torch.cuda.synchronize()
    dist.barrier()
    del recv_buf
    gc.collect()
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

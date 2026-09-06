#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Communication-fused MoE: dispatch + SwiGLU + combine in ONE persistent kernel.

A single kernel launch covers:

  Phase 1 — Dispatch:    push tokens to expert-owning ranks (CCO LSA)
  Barrier — Epoch flags: in-kernel cross-rank sync (no host round-trip)
  Phase 2 — SwiGLU:      GEMV expert compute on received tokens
  Phase 3 — Combine:     push expert outputs back to source ranks
  Barrier — Second epoch-flag sync

Cross-rank sync uses epoch flags embedded at a known offset in the dispatch
window — each rank writes its generation into every peer's flag slot via LSA,
then polls its own window until all peers have published.

    torchrun --nnodes=1 --nproc_per_node=<gpus> moe_ep_fused.py

Requires mori PR #544 (torch SymmetricMemory) and PR #594 (CCO Triton API).
"""

import argparse
import gc
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F
import triton
import triton.language as tl

import mori.allocator  # noqa: F401
from mori.cco import Communicator
from mori.ir.triton import cco

PER_RANK_VMM = 1 << 30
H_CONST = 256
I_CONST = 128
TILE = 32
PROGRAMS = 64


# ---------------------------------------------------------------------------
# Helper: push rows via LSA
# ---------------------------------------------------------------------------

@triton.jit
def _lsa_push_rows(
    window, src_p, ranks_p, slots_p, count, pid,
    H: tl.constexpr, TILE: tl.constexpr, PROGRAMS: tl.constexpr,
):
    for row_idx in range(pid, count, PROGRAMS):
        row = tl.cast(row_idx, tl.int64)
        r = tl.load(ranks_p + row_idx)
        s = tl.load(slots_p + row_idx)
        dst = cco.Window.lsa_ptr(window, r, s.to(tl.int64) * H * 2)
        dst = dst.to(tl.pointer_type(tl.bfloat16), bitcast=True)
        base = row * H
        for off in range(0, H, TILE):
            idx = off + tl.arange(0, TILE)
            mask = idx < H
            tl.store(dst + idx, tl.load(src_p + base + idx, mask=mask), mask=mask)


# ---------------------------------------------------------------------------
# Helper: epoch-flag cross-rank barrier (embedded in dispatch window)
# ---------------------------------------------------------------------------

@triton.jit
def _epoch_barrier(
    dispatch_win,  # reuse dispatch window for flag storage
    flag_byte_base,  # byte offset of the flag region within the window
    arrive_p, phase,  # local atomic counter (int32 ptr) and phase index
    pid, my_rank, gen,
    WORLD_SIZE: tl.constexpr, PROGRAMS: tl.constexpr,
):
    # Intra-rank: all programs arrive.
    slot = phase * 2
    tl.atomic_add(arrive_p + slot, 1)
    while tl.atomic_add(arrive_p + slot, 0) < PROGRAMS:
        pass

    # Cross-rank: program 0 publishes and polls.
    if pid == 0:
        flag_base = tl.cast(flag_byte_base, tl.int64)
        # Write my generation into every peer's window at slot [my_rank].
        for peer in range(WORLD_SIZE):
            addr = cco.Window.lsa_ptr(
                dispatch_win, peer, flag_base + tl.cast(my_rank, tl.int64) * 8
            )
            tl.store(addr.to(tl.pointer_type(tl.int64), bitcast=True), gen)
        # Poll my own window until all peers reached this generation.
        for peer in range(WORLD_SIZE):
            addr = cco.Window.lsa_ptr(
                dispatch_win, tl.constexpr(0), flag_base + peer.to(tl.int64) * 8
            )
            ptr = addr.to(tl.pointer_type(tl.int64), bitcast=True)
            while tl.load(ptr) < gen:
                pass
        tl.atomic_add(arrive_p + slot + 1, 1)

    while tl.atomic_add(arrive_p + slot + 1, 0) < 1:
        pass


# ---------------------------------------------------------------------------
# Fused persistent kernel
# ---------------------------------------------------------------------------

@triton.jit(do_not_specialize=[
    "d_count", "recv_count", "c_count", "my_rank", "generation",
    "flag_byte_base", "combine_byte_base",
])
def fused_moe_ep_kernel(
    # dispatch
    dispatch_win, send_ptr, d_ranks_ptr, d_slots_ptr, d_count,
    # expert
    recv_ptr, expert_out_ptr, eids_ptr, W13_ptr, W2_ptr, recv_count,
    # combine (same window as dispatch, but combine data at combine_byte_base)
    combine_win, c_ranks_ptr, c_slots_ptr, c_count,
    # sync
    arrive_ptr, my_rank, generation, flag_byte_base, combine_byte_base,
    # constexpr
    WORLD_SIZE: tl.constexpr, EPR: tl.constexpr,
    H: tl.constexpr, I: tl.constexpr,
    TILE: tl.constexpr, PROGRAMS: tl.constexpr,
):
    pid = tl.program_id(0)
    send_p = send_ptr.to(tl.pointer_type(tl.bfloat16), bitcast=True)
    d_r = d_ranks_ptr.to(tl.pointer_type(tl.int32), bitcast=True)
    d_s = d_slots_ptr.to(tl.pointer_type(tl.int32), bitcast=True)
    arrive_p = arrive_ptr.to(tl.pointer_type(tl.int32), bitcast=True)
    gen64 = tl.cast(generation, tl.int64)

    # ═══ PHASE 1: Dispatch ═══
    _lsa_push_rows(dispatch_win, send_p, d_r, d_s, d_count, pid,
                   H=H, TILE=TILE, PROGRAMS=PROGRAMS)

    # ═══ BARRIER 1 ═══
    _epoch_barrier(dispatch_win, flag_byte_base, arrive_p, tl.constexpr(0),
                   pid, my_rank, gen64,
                   WORLD_SIZE=WORLD_SIZE, PROGRAMS=PROGRAMS)

    # ═══ PHASE 2: Expert SwiGLU ═══
    recv_p = recv_ptr.to(tl.pointer_type(tl.bfloat16), bitcast=True)
    out_p = expert_out_ptr.to(tl.pointer_type(tl.bfloat16), bitcast=True)
    # Cast to int64 first to ensure the bitcast works even if Triton inferred i32
    # from a small pointer value on some ranks.
    eid_p = tl.cast(eids_ptr, tl.int64).to(tl.pointer_type(tl.int32), bitcast=True)
    w13_p = W13_ptr.to(tl.pointer_type(tl.bfloat16), bitcast=True)
    w2_p = W2_ptr.to(tl.pointer_type(tl.bfloat16), bitcast=True)

    for tok in range(pid, recv_count, PROGRAMS):
        eid = tl.load(eid_p + tok)
        x = tl.load(recv_p + tok.to(tl.int64) * tl.cast(H, tl.int64) + tl.arange(0, H)).to(tl.float32)

        w13_base = eid.to(tl.int64) * tl.cast(2 * I, tl.int64) * tl.cast(H, tl.int64)
        w3_off = tl.cast(I, tl.int64) * tl.cast(H, tl.int64)
        out_acc = tl.zeros((H,), dtype=tl.float32)

        for t in range(0, I, TILE):
            j64 = (t + tl.arange(0, TILE)).to(tl.int64)
            # gate GEMV tile
            w1_idx = w13_base + j64[:, None] * tl.cast(H, tl.int64) + tl.arange(0, H)[None, :].to(tl.int64)
            gate = tl.sum(tl.load(w13_p + w1_idx).to(tl.float32) * x[None, :], axis=1)
            # up GEMV tile
            w3_idx = w13_base + w3_off + j64[:, None] * tl.cast(H, tl.int64) + tl.arange(0, H)[None, :].to(tl.int64)
            up = tl.sum(tl.load(w13_p + w3_idx).to(tl.float32) * x[None, :], axis=1)
            # SwiGLU
            act = (gate * tl.sigmoid(gate)) * up  # [TILE]
            # W2 down-projection contribution
            w2_base = eid.to(tl.int64) * tl.cast(H, tl.int64) * tl.cast(I, tl.int64)
            w2_idx = w2_base + tl.arange(0, H)[:, None].to(tl.int64) * tl.cast(I, tl.int64) + j64[None, :]
            out_acc += tl.sum(tl.load(w2_p + w2_idx).to(tl.float32) * act[None, :], axis=1)

        tl.store(out_p + tok.to(tl.int64) * tl.cast(H, tl.int64) + tl.arange(0, H), out_acc.to(tl.bfloat16))

    # ═══ PHASE 3: Combine ═══
    c_r = tl.cast(c_ranks_ptr, tl.int64).to(tl.pointer_type(tl.int32), bitcast=True)
    c_s = tl.cast(c_slots_ptr, tl.int64).to(tl.pointer_type(tl.int32), bitcast=True)
    for row in range(pid, c_count, PROGRAMS):
        rank_r = tl.load(c_r + row)
        slot_r = tl.load(c_s + row)
        cbb = tl.cast(combine_byte_base, tl.int64)
        dst = cco.Window.lsa_ptr(combine_win, rank_r,
                                cbb + slot_r.to(tl.int64) * tl.cast(H, tl.int64) * 2)
        dst = dst.to(tl.pointer_type(tl.bfloat16), bitcast=True)
        src_row = row % recv_count
        src_off = src_row.to(tl.int64) * tl.cast(H, tl.int64)
        for off in range(0, H, TILE):
            idx = off + tl.arange(0, TILE)
            mask = idx < H
            tl.store(dst + idx, tl.load(out_p + src_off + idx, mask=mask), mask=mask)

    # ═══ BARRIER 2 ═══
    _epoch_barrier(dispatch_win, flag_byte_base, arrive_p, tl.constexpr(1),
                   pid, my_rank, gen64 + 1,
                   WORLD_SIZE=WORLD_SIZE, PROGRAMS=PROGRAMS)


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

def expert_swiglu(x, W1, W3, W2):
    return (F.silu(x @ W1.T) * (x @ W3.T)) @ W2.T


def moe_reference(tokens, router_w, W1, W3, W2, top_k):
    scores = torch.sigmoid(tokens @ router_w.T)
    tw, ti = torch.topk(scores, top_k, dim=-1)
    tw = tw / tw.sum(dim=-1, keepdim=True)
    out = torch.zeros_like(tokens)
    for t in range(tokens.shape[0]):
        for k in range(top_k):
            e = ti[t, k].item()
            out[t] += tw[t, k] * expert_swiglu(tokens[t:t+1], W1[e], W3[e], W2[e]).squeeze(0)
    return out


def build_plan(topk_indices, epr, world_size, top_k, device):
    N = topk_indices.shape[0]
    fe = topk_indices.reshape(-1)
    S = fe.shape[0]
    dr = (fe // epr).to(torch.int32)
    le = (fe % epr).to(torch.int32)
    tid = torch.arange(N, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1).int()
    kid = torch.arange(top_k, device=device).unsqueeze(0).expand(N, -1).reshape(-1).int()
    ds = torch.empty(S, dtype=torch.int32, device=device)
    for r in range(world_size):
        m = dr == r
        ds[m] = torch.arange(m.sum().item(), dtype=torch.int32, device=device)
    cs = (tid * top_k + kid).int()
    return dict(flat_dst_rank=dr, flat_local_eid=le, flat_token_id=tid,
                flat_k_id=kid, dispatch_slot=ds, combine_slot=cs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experts-per-rank", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--num-tokens", type=int, default=16)
    args = p.parse_args()
    H, I = H_CONST, I_CONST

    for k, v in (("RANK","0"),("WORLD_SIZE","1"),("LOCAL_RANK","0"),
                 ("MASTER_ADDR","127.0.0.1"),("MASTER_PORT","29500")):
        os.environ.setdefault(k, v)

    rank = int(os.environ["RANK"])
    ws = int(os.environ["WORLD_SIZE"])
    lr = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("gloo")
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)

    EPR, top_k, N = args.experts_per_rank, args.top_k, args.num_tokens
    TE = EPR * ws
    max_recv = N * top_k
    # Reserve space for sync flags at the tail of dispatch_recv.
    # Round up to a nice alignment for VMM.
    data_elems = max_recv * H
    flag_elems_bf16 = 512  # generous padding for alignment
    dispatch_buf_elems = data_elems + flag_elems_bf16
    flag_byte_base = data_elems * 2  # byte offset of flags in the buffer

    if rank == 0:
        print(f"Fused MoE EP  world={ws}  experts={TE}({EPR}/rank)  "
              f"top_k={top_k}  tokens={N}  H={H}  I={I}  programs={PROGRAMS}")

    torch.manual_seed(42)
    rw = torch.randn(TE, H, device=dev, dtype=torch.bfloat16)
    W1 = torch.randn(TE, I, H, device=dev, dtype=torch.bfloat16) * 0.02
    W3 = torch.randn(TE, I, H, device=dev, dtype=torch.bfloat16) * 0.02
    W2 = torch.randn(TE, H, I, device=dev, dtype=torch.bfloat16) * 0.02
    tokens = torch.randn(N, H, device=dev, dtype=torch.bfloat16)

    W13 = torch.cat([W1, W3], dim=1).contiguous()
    lW13 = W13[rank*EPR:(rank+1)*EPR].contiguous()
    lW2 = W2[rank*EPR:(rank+1)*EPR].contiguous()

    scores = torch.sigmoid(tokens @ rw.T)
    tw, ti = torch.topk(scores, top_k, dim=-1)
    tw = tw / tw.sum(dim=-1, keepdim=True)
    plan = build_plan(ti, EPR, ws, top_k, dev)

    S = plan["flat_dst_rank"].shape[0]
    send_buf = tokens[plan["flat_token_id"].long()].contiguous()
    mm = plan["flat_dst_rank"] == rank
    mrn = mm.sum().item()
    mi = torch.where(mm)[0]
    me = plan["flat_local_eid"][mm].contiguous()
    mcs = plan["combine_slot"][mi].contiguous()

    cr = torch.cat([torch.full((mrn,), r, dtype=torch.int32, device=dev) for r in range(ws)]) if mrn > 0 else torch.zeros(0,dtype=torch.int32,device=dev)
    cs = mcs.repeat(ws) if mrn > 0 else torch.zeros(0,dtype=torch.int32,device=dev)
    eo = torch.zeros(max(mrn,1), H, device=dev, dtype=torch.bfloat16)

    symm_mem.set_backend("MORI")
    # Single symmetric buffer: [dispatch_data | sync_flags | combine_data]
    # Only one register_external_window call needed.
    combine_data_elems = N * top_k * H
    combine_byte_base = dispatch_buf_elems * 2  # byte offset of combine region
    total_elems = dispatch_buf_elems + combine_data_elems
    symm_buf = symm_mem.empty(total_elems, dtype=torch.bfloat16, device=dev)
    symm_buf.zero_()
    disp = symm_buf  # alias for clarity — dispatch data starts at offset 0
    comb = symm_buf  # combine data starts at combine_byte_base (used as ptr offset in kernel)
    la = torch.zeros(4, dtype=torch.int32, device=dev)
    torch.cuda.synchronize()

    ct = [Communicator.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(ct, src=0)

    with Communicator.init(ws, rank, ct[0], per_rank_vmm=PER_RANK_VMM) as comm:
        win = comm.register_external_window(symm_buf.data_ptr(), symm_buf.nbytes)
        el = cco.get_extern_libs()

        # Serialize Triton JIT compilation: each rank compiles in turn.
        # Without this, concurrent JIT with extern_libs (CCO device bitcode)
        # causes "error encountered during parsing" on some ranks due to
        # cache file write races.
        for compiling_rank in range(ws):
            if rank == compiling_rank:
                fused_moe_ep_kernel.warmup(
                    win.handle, send_buf.data_ptr(),
                    plan["flat_dst_rank"].data_ptr(), plan["dispatch_slot"].data_ptr(), S,
                    symm_buf.data_ptr(), eo.data_ptr(), me.data_ptr(),
                    lW13.data_ptr(), lW2.data_ptr(), mrn,
                    win.handle, cr.data_ptr(), cs.data_ptr(), cr.shape[0],
                    la.data_ptr(), rank, 1, flag_byte_base, combine_byte_base,
                    WORLD_SIZE=ws, EPR=EPR, H=H, I=I, TILE=TILE, PROGRAMS=PROGRAMS,
                    grid=(PROGRAMS,), extern_libs=el,
                )
            dist.barrier()

        comm.barrier()

        fused_moe_ep_kernel[(PROGRAMS,)](
            win.handle, send_buf.data_ptr(),
            plan["flat_dst_rank"].data_ptr(), plan["dispatch_slot"].data_ptr(), S,
            symm_buf.data_ptr(), eo.data_ptr(), me.data_ptr(),
            lW13.data_ptr(), lW2.data_ptr(), mrn,
            win.handle, cr.data_ptr(), cs.data_ptr(), cr.shape[0],
            la.data_ptr(), rank, 1, flag_byte_base, combine_byte_base,
            WORLD_SIZE=ws, EPR=EPR, H=H, I=I, TILE=TILE, PROGRAMS=PROGRAMS,
            extern_libs=el,
        )
        torch.cuda.synchronize()
        comm.barrier()

        # combine data starts at combine_byte_base in symm_buf
        comb_start = dispatch_buf_elems
        cb = symm_buf[comb_start:comb_start + N*top_k*H].reshape(N, top_k, H)
        out = (cb * tw.unsqueeze(-1)).sum(dim=1)
        ref = moe_reference(tokens, rw, W1, W3, W2, top_k)
        d = (out - ref).abs().max().item()
        r2 = d / (ref.abs().max().item() + 1e-8)
        ok = r2 < 0.05
        if rank == 0:
            print(f"max abs diff: {d:.6f}  max rel diff: {r2:.6f}")
            print(f"correctness: {'OK' if ok else 'FAILED'}")
        comm.barrier()

    torch.cuda.synchronize()
    dist.barrier()
    del symm_buf
    gc.collect()
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

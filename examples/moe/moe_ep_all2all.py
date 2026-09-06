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
"""MoE Expert-Parallel all-to-all over mori's torch SymmetricMemory + CCO.

A complete Mixture-of-Experts forward pass with expert parallelism:

  1. Router selects top-k experts per token
  2. Dispatch: all-to-all push of tokens to the rank owning each expert
     (Triton kernel via mori CCO LSA window addressing)
  3. Expert compute: SwiGLU FFN on local experts
  4. Combine: all-to-all push of results back to source ranks
  5. Weighted merge of expert outputs

Verified against a single-GPU reference.  Uses mori PR #544 (torch
SymmetricMemory backend) for allocation and PR #594 (CCO Triton device API)
for LSA-addressed cross-GPU writes in the dispatch/combine kernels.

Every rank holds all tokens and computes the same routing table (deterministic
seed).  This sidesteps metadata exchange: each rank knows the full dispatch
plan and can reconstruct which tokens it received and where to send results
back.  In production EP the tokens would be sharded across ranks and the
routing metadata would travel with the token data.

    torchrun --nnodes=1 --nproc_per_node=<gpus> moe_ep_all2all.py
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

import mori.allocator  # noqa: F401  -- importing registers the "MORI" backend
from mori.cco import Communicator
from mori.ir.triton import cco

PER_RANK_VMM = 1 << 30
BLOCK = 256


# ---------------------------------------------------------------------------
# Triton kernel: push rows into a peer's CCO LSA window
# ---------------------------------------------------------------------------
# Used for both dispatch (token → expert rank) and combine (result → source rank).


@triton.jit(do_not_specialize=["num_rows", "row_elems"])
def lsa_push_rows_kernel(
    window,  # ccoWindow_t handle
    src_buf,  # pointer to contiguous [num_rows, row_elems] bf16 source
    dst_ranks,  # [num_rows] int32 — target rank per row
    dst_slots,  # [num_rows] int32 — slot index in target's window
    num_rows,
    row_elems,
    BLOCK: tl.constexpr,
):
    """Push ``src_buf[pid]`` into ``window[dst_rank][dst_slot]``."""
    pid = tl.program_id(0)
    if pid >= num_rows:
        return

    ranks_ptr = dst_ranks.to(tl.pointer_type(tl.int32), bitcast=True)
    slots_ptr = dst_slots.to(tl.pointer_type(tl.int32), bitcast=True)
    rank = tl.load(ranks_ptr + pid)
    slot = tl.load(slots_ptr + pid)

    byte_off = slot.to(tl.int64) * row_elems.to(tl.int64) * 2  # bf16 = 2B
    dst_base = cco.Window.lsa_ptr(window, rank, byte_off)
    dst = dst_base.to(tl.pointer_type(tl.bfloat16), bitcast=True)

    src = src_buf.to(tl.pointer_type(tl.bfloat16), bitcast=True)
    base = pid.to(tl.int64) * row_elems.to(tl.int64)

    for start in range(0, row_elems, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < row_elems
        tl.store(dst + offs, tl.load(src + base + offs, mask=mask), mask=mask)


# ---------------------------------------------------------------------------
# SwiGLU expert
# ---------------------------------------------------------------------------


def expert_swiglu(x, W1, W3, W2):
    """SwiGLU(x) = (SiLU(x W1^T) * x W3^T) W2^T."""
    return (F.silu(x @ W1.T) * (x @ W3.T)) @ W2.T


# ---------------------------------------------------------------------------
# Single-GPU reference (no communication)
# ---------------------------------------------------------------------------


def moe_reference(tokens, router_weight, all_W1, all_W3, all_W2, top_k):
    scores = torch.sigmoid(tokens @ router_weight.T)
    topk_w, topk_idx = torch.topk(scores, top_k, dim=-1)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

    out = torch.zeros_like(tokens)
    for t in range(tokens.shape[0]):
        for k in range(top_k):
            e = topk_idx[t, k].item()
            out[t] += topk_w[t, k] * expert_swiglu(
                tokens[t : t + 1], all_W1[e], all_W3[e], all_W2[e]
            ).squeeze(0)
    return out


# ---------------------------------------------------------------------------
# Build a globally-consistent dispatch / combine plan
# ---------------------------------------------------------------------------


def build_plan(topk_indices, experts_per_rank, world_size, top_k, device):
    """Return tensors describing every (token, k) → expert send.

    Because every rank computes this identically, no metadata needs to cross
    the wire.

    Returns a dict with:
      flat_dst_rank  [S] int32  — which rank owns the expert
      flat_local_eid [S] int32  — expert index within that rank
      flat_token_id  [S] int32  — originating token index
      flat_k_id      [S] int32  — which of the top_k slots this send fills
      dispatch_slot  [S] int32  — slot index inside dst rank's dispatch window
      combine_slot   [S] int32  — slot index inside src rank's combine window
                                  (= token_id * top_k + k_id)
    where S = num_tokens * top_k.
    """
    num_tokens = topk_indices.shape[0]
    flat_eid = topk_indices.reshape(-1)
    S = flat_eid.shape[0]

    dst_rank = (flat_eid // experts_per_rank).to(torch.int32)
    local_eid = (flat_eid % experts_per_rank).to(torch.int32)
    token_id = (
        torch.arange(num_tokens, device=device)
        .unsqueeze(1)
        .expand(-1, top_k)
        .reshape(-1)
        .to(torch.int32)
    )
    k_id = (
        torch.arange(top_k, device=device)
        .unsqueeze(0)
        .expand(num_tokens, -1)
        .reshape(-1)
        .to(torch.int32)
    )

    # Assign consecutive slot indices per destination rank.
    dispatch_slot = torch.empty(S, dtype=torch.int32, device=device)
    for r in range(world_size):
        mask = dst_rank == r
        n = mask.sum().item()
        dispatch_slot[mask] = torch.arange(n, dtype=torch.int32, device=device)

    # Combine slot: the position in the *source* rank's combine window.
    combine_slot = (token_id * top_k + k_id).to(torch.int32)

    return dict(
        flat_dst_rank=dst_rank,
        flat_local_eid=local_eid,
        flat_token_id=token_id,
        flat_k_id=k_id,
        dispatch_slot=dispatch_slot,
        combine_slot=combine_slot,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--intermediate-dim", type=int, default=128)
    p.add_argument("--experts-per-rank", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--num-tokens", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()

    for key, val in (
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("LOCAL_RANK", "0"),
        ("MASTER_ADDR", "127.0.0.1"),
        ("MASTER_PORT", "29500"),
    ):
        os.environ.setdefault(key, val)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group("gloo")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    H = args.hidden_dim
    I = args.intermediate_dim
    EPR = args.experts_per_rank
    top_k = args.top_k
    N = args.num_tokens
    total_experts = EPR * world_size
    # Upper bound on tokens any single rank could receive via dispatch.
    max_dispatch_recv = N * top_k

    if rank == 0:
        print(
            f"MoE EP all-to-all  world={world_size}  experts={total_experts}"
            f"({EPR}/rank)  top_k={top_k}  tokens={N}  "
            f"hidden={H}  intermediate={I}"
        )

    # -- Deterministic weights and tokens (same on every rank) ----------------
    torch.manual_seed(42)
    router_weight = torch.randn(total_experts, H, device=device, dtype=torch.bfloat16)
    all_W1 = torch.randn(total_experts, I, H, device=device, dtype=torch.bfloat16) * 0.02
    all_W3 = torch.randn(total_experts, I, H, device=device, dtype=torch.bfloat16) * 0.02
    all_W2 = torch.randn(total_experts, H, I, device=device, dtype=torch.bfloat16) * 0.02
    tokens = torch.randn(N, H, device=device, dtype=torch.bfloat16)

    local_W1 = all_W1[rank * EPR : (rank + 1) * EPR]
    local_W3 = all_W3[rank * EPR : (rank + 1) * EPR]
    local_W2 = all_W2[rank * EPR : (rank + 1) * EPR]

    # ==== Step 1: Router =====================================================
    scores = torch.sigmoid(tokens @ router_weight.T)
    topk_weights, topk_indices = torch.topk(scores, top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    plan = build_plan(topk_indices, EPR, world_size, top_k, device)

    # Identify *this rank's* sends (tokens whose expert lives on another rank)
    # and local work (tokens whose expert lives here).
    S = plan["flat_dst_rank"].shape[0]
    send_buf = tokens[plan["flat_token_id"].long()]  # [S, H]

    # How many tokens will this rank receive?
    my_recv_n = (plan["flat_dst_rank"] == rank).sum().item()

    # ==== Step 2–5: Dispatch → Experts → Combine ============================
    symm_mem.set_backend("MORI")

    # Symmetric windows: dispatch (token data) and combine (expert outputs).
    # Every rank allocates the same size so register_external_window is collective.
    dispatch_recv_buf = symm_mem.empty(
        max_dispatch_recv * H, dtype=torch.bfloat16, device=device
    )
    dispatch_recv_buf.zero_()
    combine_recv_buf = symm_mem.empty(
        N * top_k * H, dtype=torch.bfloat16, device=device
    )
    combine_recv_buf.zero_()
    torch.cuda.synchronize()

    cco_token = [Communicator.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(cco_token, src=0)

    with Communicator.init(
        world_size, rank, cco_token[0], per_rank_vmm=PER_RANK_VMM
    ) as comm:
        dispatch_win = comm.register_external_window(
            dispatch_recv_buf.data_ptr(), dispatch_recv_buf.nbytes
        )
        combine_win = comm.register_external_window(
            combine_recv_buf.data_ptr(), combine_recv_buf.nbytes
        )
        extern_libs = cco.get_extern_libs()
        comm.barrier()

        # ---- Dispatch: push token data to expert-owning ranks ---------------
        if S > 0:
            lsa_push_rows_kernel[(S,)](
                dispatch_win.handle,
                send_buf.data_ptr(),
                plan["flat_dst_rank"].data_ptr(),
                plan["dispatch_slot"].data_ptr(),
                S,
                H,
                BLOCK=BLOCK,
                extern_libs=extern_libs,
            )
        torch.cuda.synchronize()
        comm.barrier()

        # ---- Expert compute (local, PyTorch) --------------------------------
        recv_tokens = dispatch_recv_buf[: my_recv_n * H].reshape(my_recv_n, H)

        my_mask = plan["flat_dst_rank"] == rank
        my_local_eids = plan["flat_local_eid"][my_mask]
        # dst_slot was assigned consecutively per rank, so recv order matches.

        expert_out = torch.zeros_like(recv_tokens)
        for e in range(EPR):
            emask = my_local_eids == e
            if emask.any():
                expert_out[emask] = expert_swiglu(
                    recv_tokens[emask], local_W1[e], local_W3[e], local_W2[e]
                )

        # ---- Combine: push expert outputs back to every rank ----------------
        # All ranks hold all tokens, so every rank needs all expert outputs.
        # Each rank pushes its expert_out rows back to *every* rank's combine
        # buffer at the right (token_id, k_id) slot.
        #
        # The combine_slot for send i is token_id[i]*top_k + k_id[i], and the
        # data lives in expert_out at the dispatch_slot index for *this rank*.

        # Build per-peer combine sends.
        my_indices = torch.where(my_mask)[0]  # indices into the global plan
        my_dispatch_slots = plan["dispatch_slot"][my_indices]  # order in recv
        my_combine_slots = plan["combine_slot"][my_indices]

        # We need to push each expert_out row to ALL ranks (every rank wants
        # the full set of expert outputs for the merge).  For simplicity, push
        # to all ranks in a loop.
        for target_r in range(world_size):
            if my_recv_n > 0:
                target_ranks = torch.full(
                    (my_recv_n,), target_r, dtype=torch.int32, device=device
                )
                # expert_out is ordered by dispatch_slot (consecutive 0..my_recv_n-1)
                lsa_push_rows_kernel[(my_recv_n,)](
                    combine_win.handle,
                    expert_out.data_ptr(),
                    target_ranks.data_ptr(),
                    my_combine_slots.data_ptr(),
                    my_recv_n,
                    H,
                    BLOCK=BLOCK,
                    extern_libs=extern_libs,
                )
        torch.cuda.synchronize()
        comm.barrier()

        # ---- Weighted merge -------------------------------------------------
        combine_buf = combine_recv_buf[: N * top_k * H].reshape(N, top_k, H)
        moe_output = (combine_buf * topk_weights.unsqueeze(-1)).sum(dim=1)  # [N, H]

        # ==== Step 6: Correctness check ======================================
        ref = moe_reference(tokens, router_weight, all_W1, all_W3, all_W2, top_k)
        diff = (moe_output - ref).abs().max().item()
        rel = diff / (ref.abs().max().item() + 1e-8)
        ok = rel < 0.05

        if rank == 0:
            print(f"max abs diff: {diff:.6f}  max rel diff: {rel:.6f}")
            print(f"correctness: {'OK' if ok else 'FAILED'}")

        comm.barrier()

    # Explicitly delete symmetric tensors after the communicator is destroyed
    # (the `with` block above already called comm.destroy). Deleting while
    # the CCO resources still reference the underlying VMM causes a segfault.
    torch.cuda.synchronize()
    dist.barrier()
    del dispatch_recv_buf, combine_recv_buf
    gc.collect()
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

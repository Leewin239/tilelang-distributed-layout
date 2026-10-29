"""Four-rank decoder block: exact TileScale AR default vs small-tile multicast.

Attention is common to both paths; the MLP down projection is changed.
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from tilelang.distributed.allocator import get_allocator
from tilelang.distributed.host import init_dist

from _paths import add_tilescale_example

add_tilescale_example("gemm_allreduce")
from example_gemm_ar_specialized import (
    gemm_ar_sm_specialized_kernel,
    reset_gemm_ar_specialized_state_kernel,
)

os.environ.setdefault("NCCL_DEBUG", "ERROR")


@tilelang.jit(compile_once=True)
def rank_barrier_kernel(num_ranks):
    @T.prim_func
    def main(barrier: T.Tensor((num_ranks,), T.int32)):
        with T.Kernel(1, threads=32):
            T.barrier_blocks(barrier)
            T.fence_sys()
    return main


def rank_main(local_rank, args):
    rank, ranks, group = init_dist(local_rank, args.num_processes)
    assert rank == local_rank
    M, H, F_local = args.tokens, args.hidden, args.ffn // ranks
    assert F_local == H and H in (1024, 2048)
    tiles = (M // 128) * (H // 256)
    bm, bn = (64, 128) if H == 1024 else (64, 256)
    candidate_tiles = (M // bm) * (H // bn)
    allocator = get_allocator(
        size=2**30, device=f"cuda:{local_rank}", is_distributed=True,
        local_rank=local_rank, num_local_ranks=ranks, group=group,
        mcast_size=M * H * 2 + (tiles + candidate_tiles) * 4 + 16384, use_vmm=True,
    )
    original = gemm_ar_sm_specialized_kernel(M, H, F_local, ranks, 128, 256, 64, 256, 12, 4, 4096)
    reset = reset_gemm_ar_specialized_state_kernel(tiles, 256)
    candidate = gemm_ar_sm_specialized_kernel(
        M, H, F_local, ranks, bm, bn, 64, 256, 12, 4, 4096)
    reset_candidate = reset_gemm_ar_specialized_state_kernel(candidate_tiles, 256)
    barrier = rank_barrier_kernel(ranks)
    for kernel in (original, reset, candidate, reset_candidate, barrier):
        kernel.initialize(allocator=allocator)

    # Tensor-parallel input is replicated; weight shards differ by rank.
    torch.manual_seed(42)
    dtype = torch.bfloat16
    X = tilelang.tensor((M, H), dtype, allocator=allocator).normal_() / H**0.25
    torch.manual_seed(100 + local_rank)
    W_up = tilelang.tensor((H, F_local), dtype, allocator=allocator).normal_() / H**0.25
    W_down = tilelang.tensor((F_local, H), dtype, allocator=allocator).normal_() / F_local**0.25
    head_dim = 64
    heads_local = H // (ranks * head_dim)
    W_qkv = torch.randn((H, 3 * H // ranks), dtype=dtype, device="cuda") / H**0.5
    W_out = torch.randn((H // ranks, H), dtype=dtype, device="cuda") / (H // ranks)**0.5
    mcast_flat, local_flat = allocator._allocate_mcast_tensor((M * H,), dtype)
    mcast_C, original_C = mcast_flat.view(M, H), local_flat.view(M, H)
    mcast_signal, local_signal = allocator._allocate_mcast_tensor((tiles,), torch.uint32)
    candidate_mcast_signal, candidate_local_signal = allocator._allocate_mcast_tensor(
        (candidate_tiles,), torch.uint32)
    pre_signal = tilelang.tensor((ranks,), torch.int32, allocator=allocator).zero_()
    post_signal = tilelang.tensor((ranks,), torch.int32, allocator=allocator).zero_()

    def attention_path():
        norm = F.layer_norm(X, (H,))
        qkv = (norm @ W_qkv).view(M, 3, heads_local, head_dim)
        q, k, v = qkv.permute(1, 2, 0, 3).unbind(0)
        context = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        local_out = context.permute(1, 0, 2).contiguous().view(M, H // ranks)
        attention_partial = local_out @ W_out
        dist.all_reduce(attention_partial, group=group)
        return attention_partial + X

    ref_attention = attention_path()
    norm_ref = F.layer_norm(ref_attention, (H,))
    activation_ref = F.gelu(norm_ref @ W_up)
    expected = activation_ref @ W_down
    dist.all_reduce(expected, group=group)
    expected = expected + ref_attention
    dist.barrier(group)

    def body(name):
        attention_out = attention_path()
        norm = F.layer_norm(attention_out, (H,))
        activation = F.gelu(norm @ W_up)
        if name == "original":
            reset(local_signal)
        else:
            reset_candidate(candidate_local_signal)
        barrier(pre_signal)
        if name == "original":
            original(activation, W_down, original_C, mcast_C, mcast_signal, local_signal, 16)
        else:
            candidate(activation, W_down, original_C, mcast_C,
                      candidate_mcast_signal, candidate_local_signal, 16)
        barrier(post_signal)
        return original_C + attention_out

    for name in ("original", "candidate"):
        out = body(name)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, atol=0.5, rtol=0.1)
        dist.barrier(group)
        if rank == 0:
            print(f"{name} decoder block correctness passed", flush=True)

    def run_once(name):
        torch.cuda.synchronize()
        dist.barrier(group)
        start_wall = time.perf_counter()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        body(name)
        end.record()
        end.synchronize()
        wall = (time.perf_counter() - start_wall) * 1000
        measurements = torch.tensor([start.elapsed_time(end), wall], device="cuda")
        dist.all_reduce(measurements, op=dist.ReduceOp.MAX, group=group)
        return measurements.tolist()

    for _ in range(args.warmup):
        run_once("original")
        run_once("candidate")
    ratios = []
    for cycle in range(args.rep):
        b1 = run_once("original")
        d1 = run_once("candidate")
        d2 = run_once("candidate")
        b2 = run_once("original")
        base_ms = min(b1[0], b2[0])
        candidate_ms = min(d1[0], d2[0])
        ratios.append(base_ms / candidate_ms)
        if rank == 0:
            print(f"cycle={cycle} block_gpu_ms original={base_ms:.6f} "
                  f"candidate={candidate_ms:.6f} ratio={ratios[-1]:.3f}", flush=True)
    if rank == 0:
        ordered = sorted(ratios)
        print(f"Block GPU original/candidate ratio median={ordered[len(ordered)//2]:.3f} "
              f"min={ordered[0]:.3f} max={ordered[-1]:.3f}", flush=True)

    allocator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--ffn", type=int, default=2048)
    parser.add_argument("--graph-repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rep", type=int, default=7)
    args = parser.parse_args()
    assert args.ffn % args.num_processes == 0
    assert args.tokens % 128 == 0
    torch.multiprocessing.spawn(rank_main, args=(args,), nprocs=args.num_processes, join=True)

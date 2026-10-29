"""Exact TileScale AR default vs smaller compute/comm tile in fused kernel."""

import argparse
import os
import time

import torch
import torch.distributed as dist

import tilelang
import tilelang.language as T
from tilelang.distributed.allocator import get_allocator
from tilelang.distributed.host import init_dist

from _paths import add_tilescale_example

add_tilescale_example("gemm_allreduce")
from example_gemm_ar_specialized import (  # noqa: E402
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
    rank, num_ranks, group = init_dist(local_rank, args.num_processes)
    assert rank == local_rank
    M, N, K = args.M, args.N, args.K
    num_tiles = (M // 128) * (N // 256)
    candidate_tiles = (M // args.block_m) * (N // args.block_n)
    allocator = get_allocator(
        size=2**30, device=f"cuda:{local_rank}", is_distributed=True,
        local_rank=local_rank, num_local_ranks=num_ranks, group=group,
        mcast_size=M * N * 2 + (num_tiles + candidate_tiles) * 4 + 16384,
        use_vmm=True,
    )
    baseline = gemm_ar_sm_specialized_kernel(
        M, N, K, num_ranks, 128, 256, 64, 256, 12, 4, 4096
    )
    candidate = gemm_ar_sm_specialized_kernel(
        M, N, K, num_ranks, args.block_m, args.block_n, 64, 256, 12, args.stages, 4096
    )
    reset = reset_gemm_ar_specialized_state_kernel(num_tiles, 256)
    reset_candidate = reset_gemm_ar_specialized_state_kernel(candidate_tiles, 256)
    device_barrier = rank_barrier_kernel(num_ranks)
    for kernel in (baseline, candidate, reset, reset_candidate, device_barrier):
        kernel.initialize(allocator=allocator)

    dtype = torch.bfloat16
    torch.manual_seed(42 + local_rank)
    A = tilelang.tensor((M, K), dtype, allocator=allocator).normal_() / (K**0.25)
    B = tilelang.tensor((K, N), dtype, allocator=allocator).normal_() / (K**0.25)
    mcast_C_flat, local_C_flat = allocator._allocate_mcast_tensor((M * N,), dtype)
    mcast_signal, local_signal = allocator._allocate_mcast_tensor((num_tiles,), torch.uint32)
    candidate_mcast_signal, candidate_local_signal = allocator._allocate_mcast_tensor(
        (candidate_tiles,), torch.uint32)
    C_base = local_C_flat.view(M, N)
    mcast_C = mcast_C_flat.view(M, N)
    pre_signal = tilelang.tensor((num_ranks,), torch.int32, allocator=allocator).zero_()
    post_signal = tilelang.tensor((num_ranks,), torch.int32, allocator=allocator).zero_()
    expected = A @ B
    dist.all_reduce(expected, group=group)
    dist.barrier(group)

    def body(name):
        if name == "materialized":
            reset(local_signal)
        else:
            reset_candidate(candidate_local_signal)
        device_barrier(pre_signal)
        if name == "materialized":
            baseline(A, B, C_base, mcast_C, mcast_signal, local_signal, 16)
        else:
            candidate(A, B, C_base, mcast_C, candidate_mcast_signal,
                      candidate_local_signal, 16)
        device_barrier(post_signal)

    for name in ("materialized", "candidate"):
        body(name)
        torch.cuda.synchronize()
        dist.barrier(group)

    graphs = {}
    for name in ("materialized", "candidate"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.graph_repeats):
                body(name)
        graphs[name] = graph
        dist.barrier(group)

    def run_once(name):
        wall_start = time.perf_counter()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graphs[name].replay()
        end.record()
        end.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1000
        times = torch.tensor([start.elapsed_time(end) / args.graph_repeats,
                              wall_ms / args.graph_repeats], device="cuda")
        dist.all_reduce(times, op=dist.ReduceOp.MAX, group=group)
        return times.tolist(), C_base

    for name in ("materialized", "candidate"):
        _, output = run_once(name)
        torch.testing.assert_close(output, expected, atol=0.5, rtol=0.1)
        if local_rank == 0:
            print(f"{name} correctness passed", flush=True)

    for _ in range(args.warmup):
        run_once("materialized")
        run_once("candidate")
    kernel_ratios = []
    wall_ratios = []
    device_ratios = []
    for cycle in range(args.rep):
        first_base = run_once("materialized")[0]
        first_direct = run_once("candidate")[0]
        second_direct = run_once("candidate")[0]
        second_base = run_once("materialized")[0]
        base_kernel = min(first_base[0], second_base[0])
        direct_kernel = min(first_direct[0], second_direct[0])
        base_wall = min(first_base[1], second_base[1])
        direct_wall = min(first_direct[1], second_direct[1])
        base_device = base_kernel
        direct_device = direct_kernel
        kernel_ratios.append(base_kernel / direct_kernel)
        wall_ratios.append(base_wall / direct_wall)
        device_ratios.append(base_device / direct_device)
        if local_rank == 0:
            print(f"cycle={cycle} best_kernel_ms base={base_kernel:.6f} candidate={direct_kernel:.6f} "
                  f"ratio={kernel_ratios[-1]:.3f}; best_wall_ms base={base_wall:.6f} "
                  f"candidate={direct_wall:.6f} ratio={wall_ratios[-1]:.3f}; "
                  f"best_device_ms base={base_device:.6f} candidate={direct_device:.6f} "
                  f"ratio={device_ratios[-1]:.3f}", flush=True)
    if local_rank == 0:
        for label, ratios in (("kernel", kernel_ratios), ("wall", wall_ratios),
                              ("device", device_ratios)):
            ordered = sorted(ratios)
            print(f"paired {label} ratio median={ordered[len(ordered)//2]:.3f} "
                  f"min={ordered[0]:.3f} max={ordered[-1]:.3f}", flush=True)

    allocator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--K", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rep", type=int, default=5)
    parser.add_argument("--graph-repeats", type=int, default=10)
    parser.add_argument("--block-m", type=int, choices=(64, 128, 256), default=64)
    parser.add_argument("--block-n", type=int, choices=(128, 256, 512), default=128)
    parser.add_argument("--stages", type=int, choices=(2, 3, 4), default=4)
    args = parser.parse_args()
    assert args.M % 128 == 0 and args.N % 128 == 0 and args.K % 64 == 0
    torch.multiprocessing.spawn(rank_main, args=(args,), nprocs=args.num_processes, join=True)

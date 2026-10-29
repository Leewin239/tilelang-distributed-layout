"""Exact TileScale cubic RS+residual vs K128/stage2 direct residual writes.

Each replay includes output zeroing, cross-rank barriers, and a TileLang
fused GEMM+remote-atomic ReduceScatter. Graph construction is excluded.
"""

import argparse
import os
import time

import torch
import torch.distributed as dist

import tilelang
import tilelang.language as T
from tilelang.distributed.host import init_dist

from _paths import add_tilescale_example

add_tilescale_example("gemm_reducescatter")
from example_gemm_rs_specialized import gemm_rs_specialized_kernel  # noqa: E402

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
    assert rank == local_rank and num_ranks == 4
    M, N, K = args.M, args.N, args.K
    allocator = tilelang.get_allocator(
        size=max(2**29, 2*(M*(K//num_ranks) + (K//num_ranks)*N
                           + 2*(M//num_ranks)*N) + 2**28),
        device="cuda", is_distributed=True,
        local_rank=local_rank, num_local_ranks=num_ranks, group=group,
    )
    specs = {"tilescale_original": (256, 64, 4),
             "optimized": (128, 128, 2)}
    kernels = {}
    outputs = {}
    for name, (block_n, block_k, stages) in specs.items():
        kernel = gemm_rs_specialized_kernel(
            M, N, K, num_ranks, 128, block_n, block_k, 256, 12, stages, True,
        )
        kernel.initialize(allocator=allocator)
        kernels[name] = kernel
        outputs[name] = tilelang.tensor((M // num_ranks, N), torch.bfloat16, allocator=allocator)
    barrier = rank_barrier_kernel(num_ranks)
    barrier.initialize(allocator=allocator)
    pre_signal = tilelang.tensor((num_ranks,), torch.int32, allocator=allocator).zero_()
    post_signal = tilelang.tensor((num_ranks,), torch.int32, allocator=allocator).zero_()

    local_k = K // num_ranks
    torch.manual_seed(42 + local_rank)
    A = tilelang.tensor((M, local_k), torch.bfloat16, allocator=allocator).normal_() / (local_k**0.25)
    B = tilelang.tensor((local_k, N), torch.bfloat16, allocator=allocator).normal_() / (local_k**0.25)
    residual = tilelang.tensor((M // num_ranks, N), torch.bfloat16, allocator=allocator).normal_()
    final_old = tilelang.tensor((M // num_ranks, N), torch.bfloat16, allocator=allocator)
    expected = torch.empty_like(outputs["tilescale_original"])
    dist.reduce_scatter_tensor(expected, A @ B, group=group)
    expected = expected + residual
    dist.barrier(group)

    def body(name):
        if name == "tilescale_original":
            outputs[name].zero_()
            barrier(pre_signal)
            kernels[name](A, B, outputs[name])
            barrier(post_signal)
            final_old.copy_(outputs[name] + residual)
        else:
            outputs[name].copy_(residual)
            barrier(pre_signal)
            kernels[name](A, B, outputs[name])
            barrier(post_signal)

    for name in specs:
        body(name)
        torch.cuda.synchronize()
        dist.barrier(group)

    graphs = {}
    for name in specs:
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
        wall_ms = (time.perf_counter() - wall_start) * 1000 / args.graph_repeats
        gpu_ms = start.elapsed_time(end) / args.graph_repeats
        times = torch.tensor([gpu_ms, wall_ms], device="cuda")
        dist.all_reduce(times, op=dist.ReduceOp.MAX, group=group)
        return times.tolist()

    for name in specs:
        run_once(name)
        actual = final_old if name == "tilescale_original" else outputs[name]
        torch.testing.assert_close(actual, expected, atol=0.5, rtol=0.1)
        if local_rank == 0:
            print(f"{name} correctness passed", flush=True)

    for _ in range(args.warmup):
        run_once("tilescale_original")
        run_once("optimized")
    gpu_ratios = []
    wall_ratios = []
    for cycle in range(args.rep):
        b0 = run_once("tilescale_original")
        n0 = run_once("optimized")
        n1 = run_once("optimized")
        b1 = run_once("tilescale_original")
        base_gpu, new_gpu = min(b0[0], b1[0]), min(n0[0], n1[0])
        base_wall, new_wall = min(b0[1], b1[1]), min(n0[1], n1[1])
        gpu_ratios.append(base_gpu / new_gpu)
        wall_ratios.append(base_wall / new_wall)
        if local_rank == 0:
            print(f"cycle={cycle} gpu_ms original={base_gpu:.6f} optimized={new_gpu:.6f} "
                  f"ratio={gpu_ratios[-1]:.3f}; wall_ms original={base_wall:.6f} "
                  f"optimized={new_wall:.6f} ratio={wall_ratios[-1]:.3f}", flush=True)
    if local_rank == 0:
        for label, ratios in (("gpu", gpu_ratios), ("wall", wall_ratios)):
            ordered = sorted(ratios)
            print(f"paired {label} ratio median={ordered[len(ordered)//2]:.3f} "
                  f"min={ordered[0]:.3f} max={ordered[-1]:.3f}", flush=True)

    allocator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--M", type=int, default=2048)
    parser.add_argument("--N", type=int, default=2048)
    parser.add_argument("--K", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rep", type=int, default=5)
    parser.add_argument("--graph-repeats", type=int, default=10)
    args = parser.parse_args()
    assert args.num_processes == 4
    assert args.M % (args.num_processes * 128) == 0
    assert args.N % 256 == 0 and args.K % (args.num_processes * 128) == 0
    torch.multiprocessing.spawn(rank_main, args=(args,), nprocs=args.num_processes, join=True)

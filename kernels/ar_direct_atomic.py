"""GEMM+AllReduce without a full local partial-C HBM buffer.

Each rank computes its GEMM partial tile and atomically adds it to every
rank's symmetric output. This trades more peer traffic for no materialized
partial-C buffer and no separate multicast reduction CTA.
"""

import argparse
import os

import torch
import torch.distributed as dist

import tilelang
import tilelang.language as T
from tilelang.distributed.host import init_dist

os.environ.setdefault("NCCL_DEBUG", "ERROR")


@tilelang.jit(compile_once=True)
def direct_atomic_gemm_ar(M, N, K, num_ranks, block_M=128, block_N=128, block_K=64):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.bfloat16),
        B: T.Tensor((K, N), T.bfloat16),
        C: T.Tensor((M, N), T.bfloat16),
    ):
        with T.Kernel(M // block_M, N // block_N, threads=256) as (by, bx):
            A_shared = T.alloc_shared((block_M, block_K), T.bfloat16)
            B_shared = T.alloc_shared((block_K, block_N), T.bfloat16)
            C_shared = T.alloc_shared((block_M, block_N), T.bfloat16)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_local)
            for k in T.Pipelined(K // block_K, num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C_shared)
            T.sync_threads()
            for dst_rank in T.serial(num_ranks):
                T.atomic_add(
                    C[by * block_M : (by + 1) * block_M,
                      bx * block_N : (bx + 1) * block_N],
                    C_shared,
                    use_tma=True,
                    tma_wait_complete=True,
                    dst_pe=dst_rank,
                )

    return main


def rank_main(local_rank, args):
    rank, num_ranks, group = init_dist(local_rank, args.num_processes)
    assert rank == local_rank
    allocator = tilelang.get_allocator(
        size=2**30, device="cuda", is_distributed=True,
        local_rank=local_rank, num_local_ranks=num_ranks, group=group,
    )
    kernel = direct_atomic_gemm_ar(args.M, args.N, args.K, num_ranks)
    kernel.initialize(allocator=allocator)
    dtype = torch.bfloat16
    torch.manual_seed(42 + local_rank)
    A = tilelang.tensor((args.M, args.K), dtype, allocator=allocator).normal_() / (args.K**0.25)
    B = tilelang.tensor((args.K, args.N), dtype, allocator=allocator).normal_() / (args.K**0.25)
    C = tilelang.tensor((args.M, args.N), dtype, allocator=allocator)
    C.zero_()
    torch.cuda.synchronize()
    dist.barrier(group)
    kernel(A, B, C)
    torch.cuda.synchronize()
    dist.barrier(group)
    expected = A @ B
    dist.all_reduce(expected, group=group)
    max_diff = (C.float() - expected.float()).abs().max().item()
    passed = torch.allclose(C, expected, atol=0.5, rtol=0.1)
    print(f"rank {local_rank} check {'passed' if passed else 'failed'} max_diff={max_diff}", flush=True)
    assert passed
    allocator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--K", type=int, default=512)
    args = parser.parse_args()
    assert args.M % 128 == 0 and args.N % 128 == 0 and args.K % 64 == 0
    torch.multiprocessing.spawn(rank_main, args=(args,), nprocs=args.num_processes, join=True)

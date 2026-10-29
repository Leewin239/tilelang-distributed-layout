"""Direct peer AG+GEMM that reuses each remote A tile for two N tiles."""

import argparse
import os

import torch
import torch.distributed as dist

import tilelang
import tilelang.language as T
from tilelang.distributed.host import init_dist

os.environ.setdefault("NCCL_DEBUG", "ERROR")


@tilelang.jit(compile_once=True)
def direct_peer_reuse2_ag_gemm(M, N, K, num_ranks, block_M=64, block_N=128, block_K=64):
    m_per_rank = M // num_ranks
    n_per_rank = N // num_ranks
    m_blocks_per_rank = m_per_rank // block_M

    @T.prim_func
    def main(
        A_local: T.Tensor((m_per_rank, K), T.bfloat16),
        B_local: T.Tensor((K, n_per_rank), T.bfloat16),
        C: T.Tensor((M, n_per_rank), T.bfloat16),
    ):
        with T.Kernel(M // block_M, n_per_rank // (2 * block_N), threads=256) as (by, bx):
            source_rank = by // m_blocks_per_rank
            source_by = by - source_rank * m_blocks_per_rank
            A_shared = T.alloc_shared((block_M, block_K), T.bfloat16)
            B0_shared = T.alloc_shared((block_K, block_N), T.bfloat16)
            B1_shared = T.alloc_shared((block_K, block_N), T.bfloat16)
            C0_local = T.alloc_fragment((block_M, block_N), T.float32)
            C1_local = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C0_local)
            T.clear(C1_local)
            for k in T.Pipelined(K // block_K, num_stages=3):
                T.copy(A_local[source_by * block_M, k * block_K], A_shared, src_pe=source_rank)
                T.copy(B_local[k * block_K, bx * 2 * block_N], B0_shared)
                T.copy(B_local[k * block_K, (bx * 2 + 1) * block_N], B1_shared)
                T.gemm(A_shared, B0_shared, C0_local)
                T.gemm(A_shared, B1_shared, C1_local)
            T.copy(C0_local, C[by * block_M, bx * 2 * block_N])
            T.copy(C1_local, C[by * block_M, (bx * 2 + 1) * block_N])

    return main


def rank_main(local_rank, args):
    rank, num_ranks, group = init_dist(local_rank, args.num_processes)
    assert rank == local_rank
    allocator = tilelang.get_allocator(
        size=2**30, device="cuda", is_distributed=True,
        local_rank=local_rank, num_local_ranks=num_ranks, group=group,
    )
    kernel = direct_peer_reuse2_ag_gemm(args.M, args.N, args.K, num_ranks)
    kernel.initialize(allocator=allocator)
    dtype = torch.bfloat16
    torch.manual_seed(42 + local_rank)
    A = tilelang.tensor((args.M // num_ranks, args.K), dtype, allocator=allocator).normal_()
    B = tilelang.tensor((args.K, args.N // num_ranks), dtype, allocator=allocator).normal_()
    C = tilelang.tensor((args.M, args.N // num_ranks), dtype, allocator=allocator)
    dist.barrier(group)
    kernel(A, B, C)
    torch.cuda.synchronize()
    gathered = torch.empty((args.M, args.K), device="cuda", dtype=dtype)
    dist.all_gather_into_tensor(gathered, A, group=group)
    expected = gathered @ B
    torch.testing.assert_close(C, expected, atol=1e-2, rtol=1e-2)
    if local_rank == 0:
        print("reuse2 correctness passed", flush=True)
    allocator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--K", type=int, default=512)
    args = parser.parse_args()
    assert args.M % (args.num_processes * 64) == 0
    assert args.N % (args.num_processes * 256) == 0
    assert args.K % 64 == 0
    torch.multiprocessing.spawn(rank_main, args=(args,), nprocs=args.num_processes, join=True)

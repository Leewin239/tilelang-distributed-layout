"""Four-rank decoder block: TileScale fused baseline vs boundary-folded writes."""

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

from _paths import add_local_kernels, add_tilescale_example

add_local_kernels()
from ag_mcast_gelu import ag_gemm_gelu_sm_specialized_kernel  # noqa: E402
from ag_direct_peer_reuse2 import direct_peer_reuse2_ag_gemm  # noqa: E402

add_tilescale_example("allgather_gemm")
from example_allgather_gemm_specialized import (  # noqa: E402
    ag_gemm_sm_specialized_kernel,
    reset_ag_gemm_specialized_state_kernel,
)
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
    rank, nranks, group = init_dist(local_rank, args.num_processes)
    assert rank == local_rank and nranks == 4
    M, H, Fdim = args.tokens, args.hidden, args.ffn
    # Original TileScale AG+GEMM requires 2 * block_M rows per rank.
    assert M % (nranks * 256) == 0 and H % 128 == 0
    assert Fdim % (nranks * 512) == 0
    m_local, f_local = M // nranks, Fdim // nranks
    signal_blocks = (M + 255) // 256
    allocator = get_allocator(
        size=2**29, device=f"cuda:{local_rank}", is_distributed=True,
        local_rank=local_rank, num_local_ranks=nranks, group=group,
        use_vmm=True, mcast_size=M * H * 2 + signal_blocks * 4 + 4096,
    )

    ag_old = ag_gemm_sm_specialized_kernel(M, Fdim, H, nranks, 128, 256, 64, 256, 12, 4, True)
    ag_reset = reset_ag_gemm_specialized_state_kernel(signal_blocks, nranks, 256)
    ag_new = ag_gemm_gelu_sm_specialized_kernel(
        M, Fdim, H, nranks, 128, 256, 64, 256, 12, 4, True)
    rs_old = gemm_rs_specialized_kernel(M, H, Fdim, nranks, 128, 256, 64, 256, 12, 4, True)
    rs_new = (rs_old if args.rs_variant in ("original", "attn_k128") else
              gemm_rs_specialized_kernel(M, H, Fdim, nranks, 128, 128, 128, 256, 12, 2, True))
    ag_qkv_old = ag_gemm_sm_specialized_kernel(
        M, 3 * H, H, nranks, 128, 256, 64, 256, 12, 4, True)
    ag_qkv_new = (direct_peer_reuse2_ag_gemm(M, 3 * H, H, nranks)
                  if args.qkv_variant == "peer2" else ag_qkv_old)
    rs_attn_old = gemm_rs_specialized_kernel(
        M, H, H, nranks, 128, 256, 64, 256, 12, 4, True)
    rs_attn = (rs_attn_old if args.rs_variant in ("original", "mlp_k128") else
               gemm_rs_specialized_kernel(
                   M, H, H, nranks, 128, 128, 128, 256, 12, 2, True))
    barrier = rank_barrier_kernel(nranks)
    for kernel in (ag_old, ag_reset, ag_new, rs_old, rs_new,
                   ag_qkv_old, ag_qkv_new, rs_attn_old, rs_attn, barrier):
        kernel.initialize(allocator=allocator)

    dtype = torch.bfloat16
    torch.manual_seed(100 + local_rank)
    x = tilelang.tensor((m_local, H), dtype, allocator=allocator).normal_()
    x_src = x.clone()
    heads_local = H // nranks // 128
    assert heads_local > 0
    w_qkv = torch.randn((H, 3 * H // nranks), dtype=dtype, device="cuda") / H**0.5
    w_out = torch.randn((H // nranks, H), dtype=dtype, device="cuda") / (H // nranks)**0.5
    attn_gathered = torch.empty((M, H), dtype=dtype, device="cuda")
    attn_shard = torch.empty((m_local, H), dtype=dtype, device="cuda")
    attn_norm_sym = tilelang.tensor((m_local, H), dtype, allocator=allocator)
    qkv_original = tilelang.tensor((M, 3 * H // nranks), dtype, allocator=allocator)
    qkv_fused = tilelang.tensor((M, 3 * H // nranks), dtype, allocator=allocator)
    attn_original_shard = tilelang.tensor((m_local, H), dtype, allocator=allocator)
    x_norm = tilelang.tensor((m_local, H), dtype, allocator=allocator)
    w_up = tilelang.tensor((H, f_local), dtype, allocator=allocator).normal_() / H**0.5
    w_down = tilelang.tensor((f_local, H), dtype, allocator=allocator).normal_() / f_local**0.5
    up_old = tilelang.tensor((M, f_local), dtype, allocator=allocator)
    act = tilelang.tensor((M, f_local), dtype, allocator=allocator)
    act_old = tilelang.tensor((M, f_local), dtype, allocator=allocator)
    act_seed = tilelang.tensor((M, f_local), dtype, allocator=allocator)
    down_old = tilelang.tensor((m_local, H), dtype, allocator=allocator)
    final_old = tilelang.tensor((m_local, H), dtype, allocator=allocator)
    final_new = tilelang.tensor((m_local, H), dtype, allocator=allocator)
    ag_barriers = tilelang.tensor((2, nranks), torch.int32, allocator=allocator).zero_()
    mcast_flat, gathered_flat = allocator._allocate_mcast_tensor((M * H,), dtype)
    mcast_signal, local_signal = allocator._allocate_mcast_tensor((signal_blocks,), torch.uint32)
    mcast_x, gathered_x = mcast_flat.view(M, H), gathered_flat.view(M, H)
    pre_signal = tilelang.tensor((nranks,), torch.int32, allocator=allocator).zero_()
    post_signal = tilelang.tensor((nranks,), torch.int32, allocator=allocator).zero_()
    ag_reset(local_signal, ag_barriers, False)
    torch.cuda.synchronize()
    dist.barrier(group)

    def run_attention(name):
        attn_norm_sym.copy_(F.layer_norm(x_src, (H,)))
        barrier(pre_signal)
        if name == "original":
            ag_qkv_old(attn_norm_sym, w_qkv, mcast_x, gathered_x,
                       mcast_signal, local_signal, ag_barriers, qkv_original, 4)
            ag_reset(local_signal, ag_barriers, True)
            qkv = qkv_original.view(M, 3, heads_local, 128)
        else:
            if args.qkv_variant == "peer2":
                ag_qkv_new(attn_norm_sym, w_qkv, qkv_fused)
            else:
                ag_qkv_old(attn_norm_sym, w_qkv, mcast_x, gathered_x,
                           mcast_signal, local_signal, ag_barriers, qkv_fused, 4)
                ag_reset(local_signal, ag_barriers, True)
            qkv = qkv_fused.view(M, 3, heads_local, 128)
        q, k, v = qkv.permute(1, 2, 0, 3).unbind(0)
        context = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        local_out = context.permute(1, 0, 2).contiguous().view(M, H // nranks)
        if name == "original":
            attn_original_shard.zero_()
            barrier(pre_signal)
            rs_attn_old(local_out, w_out, attn_original_shard)
            barrier(post_signal)
            x.copy_(attn_original_shard + x_src)
        else:
            # The RS atomic epilogue adds into the residual buffer itself.
            x.copy_(x_src)
            barrier(pre_signal)
            rs_attn(local_out, w_out, x)
            barrier(post_signal)

    def run_layer(name):
        x_norm.copy_(F.layer_norm(x, (H,)))
        barrier(pre_signal)
        if name == "original":
            ag_old(x_norm, w_up, mcast_x, gathered_x, mcast_signal,
                   local_signal, ag_barriers, up_old, 4)
            ag_reset(local_signal, ag_barriers, True)
            act.copy_(F.gelu(up_old))
            down_old.zero_()
            barrier(pre_signal)
            rs_old(act, w_down, down_old)
            barrier(post_signal)
            final_old.copy_(down_old + x)
            return final_old
        else:
            # AG+GEMM writes GELU output in the consumer's native layout.
            ag_new(x_norm, w_up, mcast_x, gathered_x, mcast_signal,
                   local_signal, ag_barriers, act, 4)
            ag_reset(local_signal, ag_barriers, True)
            # A second RS writes directly into the block's residual output.
            final_new.copy_(x)
            barrier(pre_signal)
            rs_new(act, w_down, final_new)
            barrier(post_signal)
            return final_new

    def run_block(name):
        run_attention(name)
        return run_layer(name)

    def run_ag_region(name):
        barrier(pre_signal)
        if name == "original":
            ag_old(x_norm, w_up, mcast_x, gathered_x, mcast_signal,
                   local_signal, ag_barriers, up_old, 4)
            ag_reset(local_signal, ag_barriers, True)
            act_old.copy_(F.gelu(up_old))
            return act_old
        ag_new(x_norm, w_up, mcast_x, gathered_x, mcast_signal,
               local_signal, ag_barriers, act, 4)
        ag_reset(local_signal, ag_barriers, True)
        return act

    def run_rs_region(name):
        if name == "original":
            down_old.zero_()
            barrier(pre_signal)
            rs_old(act_seed, w_down, down_old)
            barrier(post_signal)
            final_old.copy_(down_old + x)
            return final_old
        final_new.copy_(x)
        barrier(pre_signal)
        rs_new(act_seed, w_down, final_new)
        barrier(post_signal)
        return final_new

    run_block("original")
    torch.cuda.synchronize()
    dist.barrier(group)
    x_attn_old = x.clone()
    run_block("optimized")
    torch.cuda.synchronize()
    dist.barrier(group)
    norm_attn_ref = F.layer_norm(x_src, (H,))
    dist.all_gather_into_tensor(attn_gathered, norm_attn_ref, group=group)
    qkv_ref = (attn_gathered @ w_qkv).view(M, 3, heads_local, 128)
    torch.testing.assert_close(qkv_original.view_as(qkv_ref), qkv_ref, atol=0.1, rtol=0.1)
    torch.testing.assert_close(qkv_fused.view_as(qkv_ref), qkv_ref, atol=0.1, rtol=0.1)
    q_ref, k_ref, v_ref = qkv_ref.permute(1, 2, 0, 3).unbind(0)
    context_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=True)
    local_out_ref = context_ref.permute(1, 0, 2).contiguous().view(M, H // nranks)
    dist.reduce_scatter_tensor(attn_shard, local_out_ref @ w_out, group=group)
    attention_ref = attn_shard + x_src
    torch.testing.assert_close(x_attn_old, attention_ref, atol=0.5, rtol=0.1)
    torch.testing.assert_close(x, attention_ref, atol=0.5, rtol=0.1)
    torch.testing.assert_close(x, x_attn_old, atol=0.5, rtol=0.1)
    x.copy_(attention_ref)
    gathered_stage = torch.empty((M, H), dtype=dtype, device="cuda")
    dist.all_gather_into_tensor(gathered_stage, x_norm, group=group)
    up_ref = gathered_stage @ w_up
    stage_diffs = torch.stack(((up_old.float() - up_ref.float()).abs().max(),
                              (act.float() - F.gelu(up_ref).float()).abs().max()))
    dist.all_reduce(stage_diffs, op=dist.ReduceOp.MAX, group=group)
    if local_rank == 0:
        print(f"stage max_abs_diff old_up/ref={stage_diffs[0].item():.4f} "
              f"fused_act/ref={stage_diffs[1].item():.4f}", flush=True)
    torch.testing.assert_close(final_new, final_old, atol=0.5, rtol=0.1)

    # Independent PyTorch reference for the complete MLP path.
    ref_norm = F.layer_norm(x, (H,))
    gathered_ref = torch.empty((M, H), dtype=dtype, device="cuda")
    dist.all_gather_into_tensor(gathered_ref, ref_norm, group=group)
    ref_up = gathered_ref @ w_up
    ref_partial = F.gelu(ref_up) @ w_down
    ref_down = torch.empty((m_local, H), dtype=dtype, device="cuda")
    dist.reduce_scatter_tensor(ref_down, ref_partial, group=group)
    torch.testing.assert_close(final_old, ref_down + x, atol=0.5, rtol=0.1)
    torch.testing.assert_close(final_new, ref_down + x, atol=0.5, rtol=0.1)
    if args.timing_target == "mlp":
        # Recheck both MLP paths from the same fixed attention output used below.
        run_layer("original")
        run_layer("optimized")
        torch.testing.assert_close(final_old, ref_down + x, atol=0.5, rtol=0.1)
        torch.testing.assert_close(final_new, ref_down + x, atol=0.5, rtol=0.1)
    elif args.timing_target == "ag_region":
        x_norm.copy_(ref_norm)
        run_ag_region("original")
        run_ag_region("optimized")
        torch.testing.assert_close(act_old, F.gelu(ref_up), atol=0.1, rtol=0.1)
        torch.testing.assert_close(act, F.gelu(ref_up), atol=0.1, rtol=0.1)
        torch.testing.assert_close(act, act_old, atol=0.1, rtol=0.1)
    elif args.timing_target == "rs_region":
        act_seed.copy_(F.gelu(ref_up))
        run_rs_region("original")
        run_rs_region("optimized")
        torch.testing.assert_close(final_old, ref_down + x, atol=0.5, rtol=0.1)
        torch.testing.assert_close(final_new, ref_down + x, atol=0.5, rtol=0.1)
    if local_rank == 0:
        print("Decoder block original, optimized, and PyTorch MLP reference correctness passed", flush=True)

    def timed_once(name):
        torch.cuda.synchronize()
        dist.barrier(group)
        wall_start = time.perf_counter()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if args.timing_target == "mlp":
            run_layer(name)
        elif args.timing_target == "ag_region":
            run_ag_region(name)
        elif args.timing_target == "rs_region":
            run_rs_region(name)
        else:
            run_block(name)
        end.record()
        end.synchronize()
        local_ms = torch.tensor([start.elapsed_time(end),
                                 (time.perf_counter() - wall_start) * 1000], device="cuda")
        dist.all_reduce(local_ms, op=dist.ReduceOp.MAX, group=group)
        return local_ms.tolist()

    for _ in range(args.warmup):
        timed_once("original")
        timed_once("optimized")
    ratios = []
    target_label = {"mlp": "MLP", "ag_region": "AG_REGION",
                    "rs_region": "RS_REGION", "block": "Block"}[args.timing_target]
    for cycle in range(args.rep):
        b0 = timed_once("original")
        n0 = timed_once("optimized")
        n1 = timed_once("optimized")
        b1 = timed_once("original")
        old_gpu, new_gpu = min(b0[0], b1[0]), min(n0[0], n1[0])
        old_wall, new_wall = min(b0[1], b1[1]), min(n0[1], n1[1])
        ratios.append(old_gpu / new_gpu)
        if local_rank == 0:
            print(f"cycle={cycle} {target_label.lower()}_gpu_ms original={old_gpu:.6f} optimized={new_gpu:.6f} "
                  f"ratio={ratios[-1]:.3f}; wall_ms original={old_wall:.6f} "
                  f"optimized={new_wall:.6f} ratio={old_wall/new_wall:.3f}", flush=True)
    if local_rank == 0:
        ordered = sorted(ratios)
        print(f"{target_label} GPU ratio median={ordered[len(ordered)//2]:.3f} "
              f"min={ordered[0]:.3f} max={ordered[-1]:.3f}", flush=True)

    allocator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num-processes", type=int, default=4)
    p.add_argument("--tokens", type=int, default=2048)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--ffn", type=int, default=16384)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--rep", type=int, default=3)
    p.add_argument("--rs-variant", choices=("original", "k128", "attn_k128", "mlp_k128"), default="k128")
    p.add_argument("--qkv-variant", choices=("original", "peer2"), default="original")
    p.add_argument("--timing-target", choices=("block", "mlp", "ag_region", "rs_region"), default="block")
    args = p.parse_args()
    torch.multiprocessing.spawn(rank_main, args=(args,), nprocs=args.num_processes, join=True)

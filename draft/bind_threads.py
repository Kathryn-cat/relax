import argparse
from typing import List

import numpy as np
import torch
import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.meta_schedule.postproc import Postproc
from tvm.script import tir as T


@T.prim_func
def transpose(
    rxplaceholder: T.Buffer((T.int64(64), T.int64(32)), "float16"),
    T_transpose: T.Buffer((T.int64(32), T.int64(64)), "float16"),
):
    T.func_attr({"global_symbol": "trans", "op_pattern": 2, "tir.noalias": True})
    # with T.block("root"):
    for ax0, ax1 in T.grid(T.int64(32), T.int64(64)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(rxplaceholder[v_ax1, v_ax0])
            T.writes(T_transpose[v_ax0, v_ax1])
            T_transpose[v_ax0, v_ax1] = rxplaceholder[v_ax1, v_ax0]


def apply_trace(sch: tir.Schedule) -> None:
    b0 = sch.get_block(name="T_transpose", func_name="main")
    l1, l2 = sch.get_loops(block=b0)
    l3 = sch.fuse(l1, l2, preserve_unit_iters=True)
    v4 = sch.sample_categorical(
        candidates=[32, 64, 128, 256, 512, 1024],
        probs=[
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
        ],
        decision=5,
    )
    l5, l6 = sch.split(loop=l3, factors=[None, v4], preserve_unit_iters=True)
    sch.bind(loop=l5, thread_axis="blockIdx.x")
    sch.bind(loop=l6, thread_axis="threadIdx.x")
    sch.enter_postproc()


if __name__ == "__main__":
    """
    target = tvm.target.Target("nvidia/geforce-rtx-3090")
    with ms.Profiler() as profiler:
        sch: tvm.tir.Schedule = ms.tune_tir(
            mod=transpose,
            target=target,
            num_trials_per_iter=32,
            max_trials_global=128,
            work_dir="logs",
        )
    """

    sch = tvm.tir.Schedule(transpose)
    apply_trace(sch)
    sch.mod.show()
    mod = tvm.build(sch.mod, target="cuda")

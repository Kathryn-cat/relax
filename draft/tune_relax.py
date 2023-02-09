import tvm
from tvm import meta_schedule as ms
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

A_TYPE = "float16"
B_TYPE = "float16"
C_TYPE = "float16"


def constructTransGEMMTarget(m, n, k):
    @tvm.script.ir_module
    class TransHGEMMTarget:
        @R.function
        def main(A: R.Tensor((k, m), A_TYPE), B: R.Tensor((k, n), B_TYPE)):
            with R.dataflow():
                A_new: R.Tensor((m, k), A_TYPE) = R.permute_dims(A, [1, 0])
                C: R.Tensor((m, n), C_TYPE) = R.matmul(A_new, B, out_dtype=C_TYPE)
                R.output(C)
            return C

    return TransHGEMMTarget


if __name__ == "__main__":
    mod = constructTransGEMMTarget(32, 128, 64)
    mod.show()
    target = tvm.target.Target("nvidia/geforce-rtx-3090")
    with ms.Profiler() as profiler:
        sch: tvm.tir.Schedule = ms.relax_integration.tune_relax(
            mod=mod,
            params=None,
            target=target,
            num_trials_per_iter=32,
            max_trials_global=128,
            work_dir="logs",
        )
        sch.trace.show()

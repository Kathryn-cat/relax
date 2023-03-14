# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import numpy as np
import pytest
import tvm
import tvm.testing
import tvm.topi.testing
from tvm import relax
from tvm.contrib.cutlass.build import is_shape_valid_for_cutlass_matmul
from tvm.contrib.pickle_memoize import memoize
from tvm.relax.backend import get_patterns_with_prefix
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


def build_and_run(mod, inputs_np, target, legalize=False):
    if legalize:
        mod = relax.transform.LegalizeOps()(mod)

    dev = tvm.device(target, 0)
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, dev)
    f = vm["main"]
    inputs = [tvm.nd.array(inp, dev) for inp in inputs_np]
    return f(*inputs).numpy()


def get_result_with_relax_cutlass_offload(mod, *args, assert_all_bindings_fused=True):
    patterns = [(entry.name, entry.pattern) for entry in get_patterns_with_prefix("cutlass")]
    assert len(patterns) != 0, "Cannot find cutlass patterns"

    mod.show()
    mod = partition_for_cutlass(mod)
    mod.show()
    if assert_all_bindings_fused:
        assert len(mod["main"].body.blocks[0].bindings) == 1

    codegen_pass = relax.transform.RunCodegen({"cutlass": {"sm": 80, "find_first_valid": True}})
    mod = codegen_pass(mod)

    return build_and_run(mod, args, "cuda")


def constructGEMM(m, n, k, dtype="float16"):
    @tvm.script.ir_module
    class HGEMM:
        @T.prim_func
        def hgemm(x: T.handle, y: T.handle, z: T.handle) -> None:
            A = T.match_buffer(x, (m, k), dtype)  # pylint: disable=invalid-name
            B = T.match_buffer(y, (k, n), dtype)  # pylint: disable=invalid-name
            C = T.match_buffer(z, (m, n), dtype)  # pylint: disable=invalid-name
            for l0, l1, l2 in T.grid(m, n, k):
                with T.block("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [l0, l1, l2])
                    T.reads(A[vi, vk], B[vk, vj])
                    T.writes(C[vi, vj])
                    with T.init():
                        C[vi, vj] = T.cast(0.0, dtype)
                    C[vi, vj] += A[vi, vk] * B[vk, vj]

        @R.function
        def main(A: R.Tensor((m, k), dtype), B: R.Tensor((k, n), dtype)):
            with R.dataflow():
                C: R.Tensor((m, n), dtype) = R.call_tir(
                    HGEMM.hgemm, (A, B), R.Tensor((m, n), dtype)
                )
                R.output(C)
            return C

    return HGEMM


@tvm.testing.requires_cutlass
def test_call_tir():
    m, n, k = 32, 64, 128
    mod = constructGEMM(m, n, k)
    mod.show()
    mod = partition_for_cutlass(mod)
    mod.show()
    codegen_pass = relax.transform.RunCodegen({"cutlass": {"sm": 80, "find_first_valid": True}})
    mod = codegen_pass(mod)
    mod.show()


if __name__ == "__main__":
    test_call_tir()

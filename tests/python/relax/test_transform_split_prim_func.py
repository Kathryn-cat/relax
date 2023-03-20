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

import math

import numpy as np
import pytest
import tvm
import tvm.testing
import tvm.topi.testing
from tvm import relax
from tvm.relax.transform import SplitPrimFunc
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

# pylint: disable=invalid-name,missing-function-docstring


def get_test_mod_1(b, s, s_kv, n, h, dtype="float32"):
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def matmul_bias(q_: T.handle, k_: T.handle, bias_: T.handle, score: T.handle) -> None:
            A = T.match_buffer(q_, (b, s, n, h), dtype)
            B = T.match_buffer(k_, (b, s_kv, n, h), dtype)
            bias = T.match_buffer(bias_, (b, s_kv), dtype)
            C = T.match_buffer(score, (b, n, s, s_kv), dtype)
            D = T.alloc_buffer((b, n, s, s_kv), dtype)
            for b0, b1, i, j, k in T.grid(b, n, s, s_kv, h):
                with T.block("matmul"):
                    vb0, vb1, vi, vj, vk = T.axis.remap("SSSSR", [b0, b1, i, j, k])
                    T.reads(A[vb0, vi, vb1, vk], B[vb0, vj, vb1, vk])
                    T.writes(D[vb0, vb1, vi, vj])
                    with T.init():
                        D[vb0, vb1, vi, vj] = T.cast(0.0, dtype)
                    D[vb0, vb1, vi, vj] += A[vb0, vi, vb1, vk] * B[vb0, vj, vb1, vk] / math.sqrt(h)

            for b0, b1, i, j in T.grid(b, n, s, s_kv):
                with T.block("bias"):
                    vb0, vb1, vi, vj = T.axis.remap("SSSS", [b0, b1, i, j])
                    T.reads(D[vb0, vb1, vi, vj], bias[vb0, vj])
                    T.writes(C[vb0, vb1, vi, vj])
                    with T.init():
                        C[vb0, vb1, vi, vj] = T.cast(0.0, dtype)
                    C[vb0, vb1, vi, vj] += D[vb0, vb1, vi, vj] + bias[vb0, vj]

        @T.prim_func
        def matmul_bias_1(q_: T.handle, k_: T.handle, bias_: T.handle, score: T.handle) -> None:
            A = T.match_buffer(q_, (b, s, n, h), dtype)
            B = T.match_buffer(k_, (b, s_kv, n, h), dtype)
            bias = T.match_buffer(bias_, (b, s_kv), dtype)
            C = T.match_buffer(score, (b, n, s, s_kv), dtype)
            D = T.alloc_buffer((b, n, s, s_kv), dtype)
            for b0, b1 in T.grid(b, n):
                with T.block("C"):
                    vb0, vb1 = T.axis.remap("SS", [b0, b1])
                    for i, j, k in T.grid(s, s_kv, h):
                        with T.block("matmul"):
                            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                            T.reads(A[vb0, vi, vb1, vk], B[vb0, vj, vb1, vk])
                            T.writes(D[vb0, vb1, vi, vj])
                            with T.init():
                                D[vb0, vb1, vi, vj] = T.cast(0.0, dtype)
                            D[vb0, vb1, vi, vj] += A[vb0, vi, vb1, vk] * B[vb0, vj, vb1, vk]

            for b0, b1, i, j in T.grid(b, n, s, s_kv):
                with T.block("bias"):
                    vb0, vb1, vi, vj = T.axis.remap("SSSS", [b0, b1, i, j])
                    T.reads(D[vb0, vb1, vi, vj], bias[vb0, vj])
                    T.writes(C[vb0, vb1, vi, vj])
                    with T.init():
                        C[vb0, vb1, vi, vj] = T.cast(0.0, dtype)
                    C[vb0, vb1, vi, vj] += D[vb0, vb1, vi, vj] + bias[vb0, vj]

        @R.function
        def main(
            Q: R.Tensor((b, s, n, h), dtype),
            K: R.Tensor((b, s_kv, n, h), dtype),
            bias: R.Tensor((b, s_kv), dtype),
        ):
            with R.dataflow():
                score: R.Tensor((b, n, s, s_kv), dtype) = R.call_tir(
                    Module.matmul_bias, (Q, K, bias), R.Tensor((b, n, s, s_kv), dtype)
                )
                score1: R.Tensor((b, n, s, s_kv), dtype) = R.call_tir(
                    Module.matmul_bias_1, (Q, K, bias), R.Tensor((b, n, s, s_kv), dtype)
                )
                res: R.Tensor((b, n, s, s_kv), dtype) = R.add(score, score1)
                R.output(res)
            return res

    return Module


def test_split_1():
    params = b, s, s_kv, n, h, dtype = 32, 64, 8, 16, 24, "float32"
    mod = get_test_mod_1(*params)
    mod = SplitPrimFunc()(mod)


if __name__ == "__main__":
    test_split_1()

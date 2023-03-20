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


# test case 1: einsum for matrix product
def get_attention_module_1(b, s, s_kv, n, h, h_v, dtype="float32"):
    @tvm.script.ir_module
    class Attention:
        @T.prim_func
        def compute_score(q: T.handle, k: T.handle, score: T.handle) -> None:
            A = T.match_buffer(q, (b, s, n, h), dtype)
            B = T.match_buffer(k, (b, s_kv, n, h), dtype)
            C = T.match_buffer(score, (b, n, s, s_kv), dtype)
            for b0, b1, i, j, k in T.grid(b, n, s, s_kv, h):
                with T.block("einsum_matmul"):
                    vb0, vb1, vi, vj, vk = T.axis.remap("SSSSR", [b0, b1, i, j, k])
                    T.reads(A[vb0, vi, vb1, vk], B[vb0, vj, vb1, vk])
                    T.writes(C[vb0, vb1, vi, vj])
                    with T.init():
                        C[vb0, vb1, vi, vj] = T.cast(0.0, dtype)
                    C[vb0, vb1, vi, vj] += A[vb0, vi, vb1, vk] * B[vb0, vj, vb1, vk] / math.sqrt(h)

        @T.prim_func
        def compute_result(attn: T.handle, v: T.handle, res: T.handle) -> None:
            A = T.match_buffer(attn, (b, n, s, s_kv), dtype)
            B = T.match_buffer(v, (b, s_kv, n, h_v), dtype)
            C = T.match_buffer(res, (b, s, n, h_v), dtype)
            for b0, b1, i, j, k in T.grid(b, n, s, h_v, s_kv):
                with T.block("einsum_matmul"):
                    vb0, vb1, vi, vj, vk = T.axis.remap("SSSSR", [b0, b1, i, j, k])
                    T.reads(A[vb0, vb1, vi, vk], B[vb0, vk, vb1, vj])
                    T.writes(C[vb0, vi, vb1, vj])
                    with T.init():
                        C[vb0, vi, vb1, vj] = T.cast(0.0, dtype)
                    C[vb0, vi, vb1, vj] += A[vb0, vb1, vi, vk] * B[vb0, vk, vb1, vj]

        @R.function
        def main(
            Q: R.Tensor((b, s, n, h), dtype),
            K: R.Tensor((b, s_kv, n, h), dtype),
            V: R.Tensor((b, s_kv, n, h_v), dtype),
        ):
            with R.dataflow():
                score: R.Tensor((b, n, s, s_kv), dtype) = R.call_tir(
                    Attention.compute_score, (Q, K), R.Tensor((b, n, s, s_kv), dtype)
                )
                attn: R.Tensor((b, n, s, s_kv), dtype) = R.nn.softmax(score, axis=-1)
                res: R.Tensor((b, s, n, h_v), dtype) = R.call_tir(
                    Attention.compute_result, (attn, V), R.Tensor((b, s, n, h_v), dtype)
                )
                R.output(res)
            return res

    return Attention


def get_numpy_attention_ref_1(q, k, v):
    qt = q.transpose(0, 2, 1, 3)  # b, n, s, h
    kt = k.transpose(0, 2, 3, 1)  # b, n, h, s_kv
    score = qt @ kt / np.sqrt(q.shape[-1])  # b, n, s, s_kv
    attn = tvm.topi.testing.softmax_python(score, -1)
    vt = v.transpose(0, 2, 1, 3)  # b, n, s_kv, h_v
    ref = attn @ vt  # b, n, s, h_v
    return ref.transpose(0, 2, 1, 3)  # b, s, n, h_v


def test_correctness_1():
    params = b, s, s_kv, n, h, h_v, dtype = 32, 64, 8, 16, 24, 48, "float32"
    mod = get_attention_module_1(*params)
    mod.show()

    q = np.random.randn(b, s, n, h).astype(dtype)
    k = np.random.randn(b, s_kv, n, h).astype(dtype)
    v = np.random.randn(b, s_kv, n, h_v).astype(dtype)
    args = [q, k, v]
    # compute the mod result
    target = "llvm"
    res1 = build_and_run(mod, args, target, legalize=True)
    # compute the ref result
    res2 = get_numpy_attention_ref_1(*args)
    tvm.testing.assert_allclose(res1, res2, rtol=1e-2, atol=1e-2)


# test case 2: pure high-level ops
def get_attention_module_2():
    @tvm.script.ir_module
    class Attention:
        @R.function
        def main(
            Q: R.Tensor((32, 64, 16, 24), "float32"),
            K: R.Tensor((32, 8, 16, 24), "float32"),
            V: R.Tensor((32, 8, 16, 48), "float32"),
        ):
            with R.dataflow():
                qt: R.Tensor((32, 16, 64, 24), "float32") = R.permute_dims(Q, axes=[0, 2, 1, 3])
                kt: R.Tensor((32, 16, 24, 8), "float32") = R.permute_dims(K, axes=[0, 2, 3, 1])
                score: R.Tensor((32, 16, 64, 8), "float32") = R.matmul(
                    qt, kt, out_dtype="float32"
                ) / R.const(math.sqrt(24), "float32")
                attn: R.Tensor((32, 16, 64, 8), "float32") = R.nn.softmax(score, axis=-1)
                vt: R.Tensor((32, 16, 8, 48), "float32") = R.permute_dims(V, axes=[0, 2, 1, 3])
                r: R.Tensor((32, 16, 64, 48), "float32") = R.matmul(attn, vt, out_dtype="float32")
                rt: R.Tensor((32, 64, 16, 48), "float32") = R.permute_dims(r, axes=[0, 2, 1, 3])
                R.output(rt)
            return rt

    return Attention


def test_correctness_2():
    params = b, s, s_kv, n, h, h_v, dtype = 32, 64, 8, 16, 24, 48, "float32"
    mod = get_attention_module_2()
    mod.show()

    q = np.random.randn(b, s, n, h).astype(dtype)
    k = np.random.randn(b, s_kv, n, h).astype(dtype)
    v = np.random.randn(b, s_kv, n, h_v).astype(dtype)
    args = [q, k, v]
    # compute the mod result
    target = "llvm"
    res1 = build_and_run(mod, args, target, legalize=True)
    # compute the ref result
    res2 = get_numpy_attention_ref_1(*args)
    tvm.testing.assert_allclose(res1, res2, rtol=1e-2, atol=1e-2)


# test case 3: pure high-level ops, permute their order
def get_attention_module_3():
    @tvm.script.ir_module
    class Attention:
        @R.function
        def main(
            Q: R.Tensor((32, 64, 16, 24), "float32"),
            K: R.Tensor((32, 8, 16, 24), "float32"),
            V: R.Tensor((32, 8, 16, 48), "float32"),
        ):
            with R.dataflow():
                kt: R.Tensor((32, 16, 24, 8), "float32") = R.permute_dims(K, axes=[0, 2, 3, 1])
                qt: R.Tensor((32, 16, 64, 24), "float32") = R.permute_dims(Q, axes=[0, 2, 1, 3])
                vt: R.Tensor((32, 16, 8, 48), "float32") = R.permute_dims(V, axes=[0, 2, 1, 3])
                score: R.Tensor((32, 16, 64, 8), "float32") = R.matmul(
                    qt, kt, out_dtype="float32"
                ) / R.const(math.sqrt(24), "float32")
                attn: R.Tensor((32, 16, 64, 8), "float32") = R.nn.softmax(score, axis=-1)
                r: R.Tensor((32, 16, 64, 48), "float32") = R.matmul(attn, vt, out_dtype="float32")
                rt: R.Tensor((32, 64, 16, 48), "float32") = R.permute_dims(r, axes=[0, 2, 1, 3])
                R.output(rt)
            return rt

    return Attention


def test_correctness_3():
    params = b, s, s_kv, n, h, h_v, dtype = 32, 64, 8, 16, 24, 48, "float32"
    mod = get_attention_module_3()
    mod.show()

    q = np.random.randn(b, s, n, h).astype(dtype)
    k = np.random.randn(b, s_kv, n, h).astype(dtype)
    v = np.random.randn(b, s_kv, n, h_v).astype(dtype)
    args = [q, k, v]
    # compute the mod result
    target = "llvm"
    res1 = build_and_run(mod, args, target, legalize=True)
    # compute the ref result
    res2 = get_numpy_attention_ref_1(*args)
    tvm.testing.assert_allclose(res1, res2, rtol=1e-2, atol=1e-2)


# test case 4: with bias
def get_attention_module_4(b, s, s_kv, n, h, h_v, dtype="float32"):
    @tvm.script.ir_module
    class Attention:
        @T.prim_func
        def compute_score(q: T.handle, k: T.handle, bias_: T.handle, score: T.handle) -> None:
            A = T.match_buffer(q, (b, s, n, h), dtype)
            B = T.match_buffer(k, (b, s_kv, n, h), dtype)
            bias = T.match_buffer(bias_, (b, s_kv), dtype)
            C = T.match_buffer(score, (b, n, s, s_kv), dtype)
            D = T.alloc_buffer((b, n, s, s_kv), dtype)
            for b0, b1, i, j, k in T.grid(b, n, s, s_kv, h):
                with T.block("einsum_matmul"):
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
        def compute_result(attn: T.handle, vt: T.handle, res: T.handle) -> None:
            A = T.match_buffer(attn, (b, n, s, s_kv), dtype)
            B = T.match_buffer(vt, (b, n, s_kv, h_v), dtype)
            C = T.match_buffer(res, (b, s, n, h_v), dtype)
            for b0, b1, i, j, k in T.grid(b, n, s, h_v, s_kv):
                with T.block("einsum_matmul"):
                    vb0, vb1, vi, vj, vk = T.axis.remap("SSSSR", [b0, b1, i, j, k])
                    T.reads(A[vb0, vb1, vi, vk], B[vb0, vb1, vk, vj])
                    T.writes(C[vb0, vi, vb1, vj])
                    with T.init():
                        C[vb0, vi, vb1, vj] = T.cast(0.0, dtype)
                    C[vb0, vi, vb1, vj] += A[vb0, vb1, vi, vk] * B[vb0, vb1, vk, vj]

        @R.function
        def main(
            Q: R.Tensor((b, s, n, h), dtype),
            K: R.Tensor((b, s_kv, n, h), dtype),
            V: R.Tensor((b, s_kv, n, h_v), dtype),
            bias: R.Tensor((b, s_kv), dtype),
        ):
            with R.dataflow():
                vt: R.Tensor((b, n, s_kv, h_v)) = R.permute_dims(V, axes=[0, 2, 1, 3])
                score: R.Tensor((b, n, s, s_kv), dtype) = R.call_tir(
                    Attention.compute_score, (Q, K, bias), R.Tensor((b, n, s, s_kv), dtype)
                )
                attn: R.Tensor((b, n, s, s_kv), dtype) = R.nn.softmax(score, axis=-1)
                res: R.Tensor((b, s, n, h_v), dtype) = R.call_tir(
                    Attention.compute_result, (attn, vt), R.Tensor((b, s, n, h_v), dtype)
                )
                R.output(res)
            return res

    return Attention


def get_numpy_attention_ref_4(q, k, v, bias):
    qt = q.transpose(0, 2, 1, 3)  # b, n, s, h
    kt = k.transpose(0, 2, 3, 1)  # b, n, h, s_kv
    score = qt @ kt / np.sqrt(q.shape[-1])  # b, n, s, s_kv
    score = score + bias[None, None, ...].transpose(2, 0, 1, 3)
    attn = tvm.topi.testing.softmax_python(score, -1)
    vt = v.transpose(0, 2, 1, 3)  # b, n, s_kv, h_v
    ref = attn @ vt  # b, n, s, h_v
    return ref.transpose(0, 2, 1, 3)  # b, s, n, h_v


def test_correctness_4():
    params = b, s, s_kv, n, h, h_v, dtype = 32, 64, 8, 16, 24, 48, "float32"
    mod = get_attention_module_4(*params)
    mod.show()

    q = np.random.randn(b, s, n, h).astype(dtype)
    k = np.random.randn(b, s_kv, n, h).astype(dtype)
    v = np.random.randn(b, s_kv, n, h_v).astype(dtype)
    bias = np.random.rand(b, s_kv).astype(dtype)
    args = [q, k, v, bias]
    # compute the mod result
    target = "llvm"
    res1 = build_and_run(mod, args, target, legalize=True)
    # compute the ref result
    res2 = get_numpy_attention_ref_4(*args)
    tvm.testing.assert_allclose(res1, res2, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test_correctness_1()

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

from __future__ import annotations

import tempfile

import numpy as np
import tvm
import tvm.relax.cutlass.pattern
import tvm.testing
from tvm import relax, runtime
from tvm.relax.vm import build as relax_build

PKG_FILE = "/tmp/test_transform_cutlass_codegen.so"
GLOBAL_SYMBOL = "HGEMM"
A_TYPE = "float16"
B_TYPE = "float16"
C_TYPE = "float16"

target = "cuda"


def f_run(rt_mod: runtime.Module, device: runtime.ndarray.Device, *input):
    vm = relax.vm.VirtualMachine(exec=rt_mod, device=device)
    return vm["main"](*input)


def build(mod):
    print("original module:")
    mod.show()
    mod = relax.transform.SplitCutlass()(mod)
    print("after SplitCutlass:")
    mod.show()
    mod = relax.transform.CutlassCodegen()(mod)
    print("after CutlassCodegen:")
    mod.show()
    try:
        executable = relax_build(mod, target)
    except tvm._ffi.base.TVMError:
        return False
    executable.mod.export_library(PKG_FILE, cc="nvcc")
    return True


# basic tests


def constructGEMM(m, n, k, GLOBAL_SYMBOL="HGEMM"):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg("A", T.buffer_decl((m, k), A_TYPE))  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((k, n), B_TYPE))  # pylint: disable=invalid-name
                D = T.alloc_buffer((m, n), C_TYPE)
                with T.grid(m, n, k) as (l0, l1, l2):
                    with T.block("dense_row_row_row"):
                        vi, vj, vk = T.axis.remap("SSR", [l0, l1, l2])
                        T.reads(A[vi, vk], B[vk, vj])
                        T.writes(D[vi, vj])
                        with T.init():
                            T.buffer_store(D, T.cast(0.0, C_TYPE), [vi, vj])
                        T.buffer_store(D, D[vi, vj] + A[vi, vk] * B[vk, vj], [vi, vj])
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((m, k), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((k, n), B_TYPE))  # pylint: disable=invalid-name
                C = R.call_tir(
                    frame.global_vars[GLOBAL_SYMBOL], args=[A, B], shape=(m, n), dtype=C_TYPE
                )
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def test_cutlass_dense():
    m, n, k = 128, 128, 128
    assert build(constructGEMM(m, n, k)), "build failure on CUDA"
    dev = tvm.cuda()
    A = np.random.rand(m, k).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B, rtol=1e-2)


def constructGEMM_bias(m, n, k, GLOBAL_SYMBOL="HGEMM_bias"):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg("A", T.buffer_decl((m, k), A_TYPE))  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((k, n), B_TYPE))  # pylint: disable=invalid-name
                bias = T.arg("bias", T.buffer_decl((1, n), A_TYPE))  # pylint: disable=invalid-name
                C = T.arg("C", T.buffer_decl((m, n), C_TYPE))  # pylint: disable=invalid-name
                D = T.alloc_buffer((m, n), C_TYPE)
                with T.grid(m, n, k) as (l0, l1, l2):
                    with T.block("dense_row_row_row"):
                        vi, vj, vk = T.axis.remap("SSR", [l0, l1, l2])
                        T.reads(A[vi, vk], B[vk, vj])
                        T.writes(D[vi, vj])
                        with T.init():
                            T.buffer_store(D, T.cast(0.0, C_TYPE), [vi, vj])
                        T.buffer_store(D, D[vi, vj] + A[vi, vk] * B[vk, vj], [vi, vj])
                with T.grid(m, n) as (i, j):
                    with T.block("bias"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        T.reads(D[vi, vj], bias[0, vj])
                        T.writes(C[vi, vj])
                        T.buffer_store(C, D[vi, vj] + bias[0, vj], [vi, vj])
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((m, k), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((k, n), B_TYPE))  # pylint: disable=invalid-name
                bias = R.arg("bias", R.tensor((1, n), A_TYPE))  # pylint: disable=invalid-name
                C = R.call_tir(
                    frame.global_vars[GLOBAL_SYMBOL], args=[A, B, bias], shape=(m, n), dtype=C_TYPE
                )
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def test_cutlass_dense_bias():
    m, n, k = 128, 128, 128
    assert build(constructGEMM_bias(m, n, k)), "build failure on CUDA"
    dev = tvm.cuda()
    A = np.random.rand(m, k).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    bias = np.random.rand(1, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B + bias, rtol=1e-2)


def constructGEMM_bias_relu(m, n, k, GLOBAL_SYMBOL="HGEMM"):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg("A", T.buffer_decl((m, k), A_TYPE))  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((k, n), B_TYPE))  # pylint: disable=invalid-name
                bias = T.arg("bias", T.buffer_decl((1, n), A_TYPE))  # pylint: disable=invalid-name
                C = T.arg("C", T.buffer_decl((m, n), C_TYPE))  # pylint: disable=invalid-name
                D = T.alloc_buffer((m, n), C_TYPE)
                E = T.alloc_buffer((m, n), C_TYPE)
                with T.grid(m, n, k) as (l0, l1, l2):
                    with T.block("dense_row_row_row"):
                        vi, vj, vk = T.axis.remap("SSR", [l0, l1, l2])
                        T.reads(A[vi, vk], B[vk, vj])
                        T.writes(D[vi, vj])
                        with T.init():
                            T.buffer_store(D, T.cast(0.0, C_TYPE), [vi, vj])
                        T.buffer_store(D, D[vi, vj] + A[vi, vk] * B[vk, vj], [vi, vj])
                with T.grid(m, n) as (i, j):
                    with T.block("bias"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        T.reads(D[vi, vj], bias[0, vj])
                        T.writes(E[vi, vj])
                        T.buffer_store(E, D[vi, vj] + bias[0, vj], [vi, vj])
                with T.grid(m, n) as (i, j):
                    with T.block("relu"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        T.reads(E[vi, vj])
                        T.writes(C[vi, vj])
                        T.buffer_store(C, T.max(E[vi, vj], T.cast(0.0, C_TYPE)), [vi, vj])
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((m, k), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((k, n), B_TYPE))  # pylint: disable=invalid-name
                bias = R.arg("bias", R.tensor((1, n), A_TYPE))  # pylint: disable=invalid-name
                C = R.call_tir(
                    frame.global_vars[GLOBAL_SYMBOL], args=[A, B, bias], shape=(m, n), dtype=C_TYPE
                )
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def test_cutlass_dense_bias_relu():
    m, n, k = 128, 128, 128
    assert build(constructGEMM_bias_relu(m, n, k)), "build failure on CUDA"
    dev = tvm.cuda()
    A = np.random.rand(m, k).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    bias = np.random.rand(1, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
    np.testing.assert_allclose(result.numpy(), np.maximum(A @ B + bias, 0), rtol=1e-2)


def constructBatchGEMM(b, m, n, k, GLOBAL_SYMBOL="BatchHGEMM"):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg("A", T.buffer_decl((b, m, k), A_TYPE))  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((k, n), B_TYPE))  # pylint: disable=invalid-name
                C = T.arg("C", T.buffer_decl((b, m, n), C_TYPE))  # pylint: disable=invalid-name
                with T.grid(b, m, n, k) as (lb, l0, l1, l2):
                    with T.block("batch_dense_row_row_row"):
                        vb, vi, vj, vk = T.axis.remap("SSSR", [lb, l0, l1, l2])
                        T.reads(A[vb, vi, vk], B[vk, vj])
                        T.writes(C[vb, vi, vj])
                        with T.init():
                            T.buffer_store(C, T.cast(0.0, C_TYPE), [vb, vi, vj])
                        T.buffer_store(C, C[vb, vi, vj] + A[vb, vi, vk] * B[vk, vj], [vb, vi, vj])
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((b, m, k), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((k, n), B_TYPE))  # pylint: disable=invalid-name
                C = R.call_tir(
                    frame.global_vars[GLOBAL_SYMBOL], args=[A, B], shape=(b, m, n), dtype=C_TYPE
                )
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def test_cutlass_batch_dense():
    b, m, n, k = 2, 128, 128, 128
    assert build(constructBatchGEMM(b, m, n, k)), "build failure on CUDA"
    dev = tvm.cuda()
    A = np.random.rand(b, m, k).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B, rtol=1e-2)


def constructBatchGEMM2(b, m, n, k, GLOBAL_SYMBOL="BatchHGEMM2"):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg("A", T.buffer_decl((b, m, k), A_TYPE))  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((b, k, n), B_TYPE))  # pylint: disable=invalid-name
                D = T.alloc_buffer((b, m, n), C_TYPE)
                with T.grid(b, m, n, k) as (lb, l0, l1, l2):
                    with T.block("batch_dense_row_row_row"):
                        vb, vi, vj, vk = T.axis.remap("SSSR", [lb, l0, l1, l2])
                        T.reads(A[vb, vi, vk], B[vb, vk, vj])
                        T.writes(D[vb, vi, vj])
                        with T.init():
                            T.buffer_store(D, T.cast(0.0, C_TYPE), [vb, vi, vj])
                        T.buffer_store(
                            D, D[vb, vi, vj] + A[vb, vi, vk] * B[vb, vk, vj], [vb, vi, vj]
                        )
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((b, m, k), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((b, k, n), B_TYPE))  # pylint: disable=invalid-name
                C = R.call_tir(
                    frame.global_vars[GLOBAL_SYMBOL], args=[A, B], shape=(b, m, n), dtype=C_TYPE
                )
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def test_cutlass_batch_dense2():
    b, m, n, k = 2, 128, 128, 128
    assert build(constructBatchGEMM2(b, m, n, k)), "build failure on CUDA"
    dev = tvm.cuda()
    A = np.random.rand(b, m, k).astype("float16") * 5
    B = np.random.rand(b, k, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B, rtol=1e-2)


def constructBatchGEMM_bias(b, m, n, k, GLOBAL_SYMBOL="BatchHGEMM_bias"):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg("A", T.buffer_decl((b, m, k), A_TYPE))  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((k, n), B_TYPE))  # pylint: disable=invalid-name
                bias = T.arg("bias", T.buffer_decl((1, n), C_TYPE))  # pylint: disable=invalid-name
                C = T.arg("C", T.buffer_decl((b, m, n), C_TYPE))  # pylint: disable=invalid-name

                D = T.alloc_buffer((b, m, n), C_TYPE)
                with T.grid(b, m, n, k) as (lb, l0, l1, l2):
                    with T.block("batch_dense_row_row_row"):
                        vb, vi, vj, vk = T.axis.remap("SSSR", [lb, l0, l1, l2])
                        T.reads(A[vb, vi, vk], B[vk, vj])
                        T.writes(D[vb, vi, vj])
                        with T.init():
                            T.buffer_store(D, T.cast(0.0, C_TYPE), [vb, vi, vj])
                        T.buffer_store(D, D[vb, vi, vj] + A[vb, vi, vk] * B[vk, vj], [vb, vi, vj])
                with T.grid(b, m, n) as (lb, i, j):
                    with T.block("bias"):
                        vb, vi, vj = T.axis.remap("SSS", [lb, i, j])
                        T.reads(D[vb, vi, vj], bias[0, vj])
                        T.writes(C[vb, vi, vj])
                        T.buffer_store(C, D[vb, vi, vj] + bias[0, vj], [vb, vi, vj])
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((b, m, k), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((k, n), B_TYPE))  # pylint: disable=invalid-name
                bias = R.arg("bias", R.tensor((1, n), C_TYPE))  # pylint: disable=invalid-name
                C = R.call_tir(
                    frame.global_vars[GLOBAL_SYMBOL],
                    args=[A, B, bias],
                    shape=(b, m, n),
                    dtype=C_TYPE,
                )
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def test_cutlass_batch_dense_bias():
    b, m, n, k = 2, 128, 128, 128
    assert build(constructBatchGEMM_bias(b, m, n, k)), "build failure on CUDA"
    dev = tvm.cuda()
    A = np.random.rand(b, m, k).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    bias = np.random.rand(1, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B + bias, rtol=1e-2)


def constructBatchGEMM2_bias(b, m, n, k, GLOBAL_SYMBOL="BatchHGEMM2_bias"):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg("A", T.buffer_decl((b, m, k), A_TYPE))  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((b, k, n), B_TYPE))  # pylint: disable=invalid-name
                bias = T.arg("bias", T.buffer_decl((1, n), C_TYPE))  # pylint: disable=invalid-name
                C = T.arg("C", T.buffer_decl((b, m, n), C_TYPE))  # pylint: disable=invalid-name

                D = T.alloc_buffer((b, m, n), C_TYPE)
                with T.grid(b, m, n, k) as (lb, l0, l1, l2):
                    with T.block("batch_dense_row_row_row2"):
                        vb, vi, vj, vk = T.axis.remap("SSSR", [lb, l0, l1, l2])
                        T.reads(A[vb, vi, vk], B[vb, vk, vj])
                        T.writes(D[vb, vi, vj])
                        with T.init():
                            T.buffer_store(D, T.cast(0.0, C_TYPE), [vb, vi, vj])
                        T.buffer_store(
                            D, D[vb, vi, vj] + A[vb, vi, vk] * B[vb, vk, vj], [vb, vi, vj]
                        )
                with T.grid(b, m, n) as (lb, i, j):
                    with T.block("bias"):
                        vb, vi, vj = T.axis.remap("SSS", [lb, i, j])
                        T.reads(D[vb, vi, vj], bias[0, vj])
                        T.writes(C[vb, vi, vj])
                        T.buffer_store(C, D[vb, vi, vj] + bias[0, vj], [vb, vi, vj])
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((b, m, k), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((b, k, n), B_TYPE))  # pylint: disable=invalid-name
                bias = R.arg("bias", R.tensor((1, n), C_TYPE))  # pylint: disable=invalid-name
                C = R.call_tir(
                    frame.global_vars[GLOBAL_SYMBOL],
                    args=[A, B, bias],
                    shape=(b, m, n),
                    dtype=C_TYPE,
                )
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def test_cutlass_batch_dense2_bias():
    b, m, n, k = 2, 128, 128, 128
    assert build(constructBatchGEMM2_bias(b, m, n, k)), "build failure on CUDA"
    dev = tvm.cuda()
    A = np.random.rand(b, m, k).astype("float16") * 5
    B = np.random.rand(b, k, n).astype("float16") * 5
    bias = np.random.rand(1, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B + bias, rtol=1e-2)


# manually constructed test cases for multi-dim batch workloads and loop permutation

# einsum "ghij, ghjk -> ghik"
def constructMultiBatchGEMM(b1, b2, m, n, k, GLOBAL_SYMBOL="MultiBatchHGEMM"):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg(
                    "A", T.buffer_decl((b1, b2, m, k), A_TYPE)
                )  # pylint: disable=invalid-name
                B = T.arg(
                    "B", T.buffer_decl((b1, b2, k, n), B_TYPE)
                )  # pylint: disable=invalid-name
                D = T.alloc_buffer((b1, b2, m, n), C_TYPE)
                with T.grid(b1, b2, m, n, k) as (lb1, lb2, l0, l1, l2):
                    with T.block("multi_batch_dense_row_row_row"):
                        vb1, vb2, vi, vj, vk = T.axis.remap("SSSSR", [lb1, lb2, l0, l1, l2])
                        T.reads(A[vb1, vb2, vi, vk], B[vb1, vb2, vk, vj])
                        T.writes(D[vb1, vb2, vi, vj])
                        with T.init():
                            T.buffer_store(D, T.cast(0.0, C_TYPE), [vb1, vb2, vi, vj])
                        T.buffer_store(
                            D,
                            D[vb1, vb2, vi, vj] + A[vb1, vb2, vi, vk] * B[vb1, vb2, vk, vj],
                            [vb1, vb2, vi, vj],
                        )
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((b1, b2, m, k), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((b1, b2, k, n), B_TYPE))  # pylint: disable=invalid-name
                C = R.call_tir(
                    frame.global_vars[GLOBAL_SYMBOL],
                    args=[A, B],
                    shape=(b1, b2, m, n),
                    dtype=C_TYPE,
                )
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def test_cutlass_multi_batch_dense():
    b1, b2, m, n, k = 2, 3, 128, 128, 128
    assert build(constructMultiBatchGEMM(b1, b2, m, n, k)), "build failure on CUDA"
    dev = tvm.cuda()
    A = np.random.rand(b1, b2, m, k).astype("float16") * 5
    B = np.random.rand(b1, b2, k, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B, rtol=1e-2)


# einsum "ghij, hjk -> ghik"
def constructMultiBatchGEMM2(b1, b2, m, n, k, GLOBAL_SYMBOL="MultiBatchHGEMM2"):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg(
                    "A", T.buffer_decl((b1, b2, m, k), A_TYPE)
                )  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((b2, k, n), B_TYPE))  # pylint: disable=invalid-name
                D = T.alloc_buffer((b1, b2, m, n), C_TYPE)
                with T.grid(b1, b2, m, n, k) as (lb1, lb2, l0, l1, l2):
                    with T.block("multi_batch_dense_row_row_row"):
                        vb1, vb2, vi, vj, vk = T.axis.remap("SSSSR", [lb1, lb2, l0, l1, l2])
                        T.reads(A[vb1, vb2, vi, vk], B[vb2, vk, vj])
                        T.writes(D[vb1, vb2, vi, vj])
                        with T.init():
                            T.buffer_store(D, T.cast(0.0, C_TYPE), [vb1, vb2, vi, vj])
                        T.buffer_store(
                            D,
                            D[vb1, vb2, vi, vj] + A[vb1, vb2, vi, vk] * B[vb2, vk, vj],
                            [vb1, vb2, vi, vj],
                        )
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((b1, b2, m, k), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((b2, k, n), B_TYPE))  # pylint: disable=invalid-name
                C = R.call_tir(
                    frame.global_vars[GLOBAL_SYMBOL],
                    args=[A, B],
                    shape=(b1, b2, m, n),
                    dtype=C_TYPE,
                )
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def test_cutlass_multi_batch_dense2():
    b1, b2, m, n, k = 2, 3, 128, 128, 128
    assert build(constructMultiBatchGEMM2(b1, b2, m, n, k)), "build failure on CUDA"
    dev = tvm.cuda()
    A = np.random.rand(b1, b2, m, k).astype("float16") * 5
    B = np.random.rand(b2, k, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B, rtol=1e-2)


# einsum "ghij, gjk -> ghik"
def constructMultiBatchGEMM3(b1, b2, m, n, k, GLOBAL_SYMBOL="MultiBatchHGEMM3"):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg(
                    "A", T.buffer_decl((b1, b2, m, k), A_TYPE)
                )  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((b1, k, n), B_TYPE))  # pylint: disable=invalid-name
                D = T.alloc_buffer((b1, b2, m, n), C_TYPE)
                with T.grid(b1, b2, m, n, k) as (lb1, lb2, l0, l1, l2):
                    with T.block("multi_batch_dense_row_row_row"):
                        vb1, vb2, vi, vj, vk = T.axis.remap("SSSSR", [lb1, lb2, l0, l1, l2])
                        T.reads(A[vb1, vb2, vi, vk], B[vb1, vk, vj])
                        T.writes(D[vb1, vb2, vi, vj])
                        with T.init():
                            T.buffer_store(D, T.cast(0.0, C_TYPE), [vb1, vb2, vi, vj])
                        T.buffer_store(
                            D,
                            D[vb1, vb2, vi, vj] + A[vb1, vb2, vi, vk] * B[vb1, vk, vj],
                            [vb1, vb2, vi, vj],
                        )
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((b1, b2, m, k), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((b1, k, n), B_TYPE))  # pylint: disable=invalid-name
                C = R.call_tir(
                    frame.global_vars[GLOBAL_SYMBOL],
                    args=[A, B],
                    shape=(b1, b2, m, n),
                    dtype=C_TYPE,
                )
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def test_cutlass_multi_batch_dense3():
    b1, b2, m, n, k = 2, 3, 128, 128, 128
    assert build(constructMultiBatchGEMM3(b1, b2, m, n, k)), "build failure on CUDA"
    dev = tvm.cuda()
    A = np.random.rand(b1, b2, m, k).astype("float16") * 5
    B = np.random.rand(b1, k, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), np.einsum("ghij, gjk->ghik", A, B), rtol=1e-2)


# einsum "ij, ik -> jk"
def constructTransGEMM(m, n, k, GLOBAL_SYMBOL="TransHGEMM"):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg("A", T.buffer_decl((k, m), A_TYPE))  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((k, n), B_TYPE))  # pylint: disable=invalid-name
                D = T.alloc_buffer((m, n), C_TYPE)
                with T.grid(m, n, k) as (l0, l1, l2):
                    with T.block("trans_dense_row_row_row"):
                        vi, vj, vk = T.axis.remap("SSR", [l0, l1, l2])
                        T.reads(A[vk, vi], B[vk, vj])
                        T.writes(D[vi, vj])
                        with T.init():
                            T.buffer_store(D, T.cast(0.0, C_TYPE), [vi, vj])
                        T.buffer_store(
                            D,
                            D[vi, vj] + A[vk, vi] * B[vk, vj],
                            [vi, vj],
                        )
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((k, m), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((k, n), B_TYPE))  # pylint: disable=invalid-name
                C = R.call_tir(
                    frame.global_vars[GLOBAL_SYMBOL],
                    args=[A, B],
                    shape=(m, n),
                    dtype=C_TYPE,
                )
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def test_cutlass_trans_dense():
    m, n, k = 128, 128, 128
    assert build(constructTransGEMM(m, n, k)), "build failure on CUDA"
    dev = tvm.cuda()
    A = np.random.rand(k, m).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), A.T @ B, rtol=1e-2)


# einsum "ik, jk -> ij"
def constructTransGEMM2(m, n, k, GLOBAL_SYMBOL="TransHGEMM2"):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg("A", T.buffer_decl((m, k), A_TYPE))  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((n, k), B_TYPE))  # pylint: disable=invalid-name
                D = T.alloc_buffer((m, n), C_TYPE)
                with T.grid(m, n, k) as (l0, l1, l2):
                    with T.block("trans_dense_row_row_row"):
                        vi, vj, vk = T.axis.remap("SSR", [l0, l1, l2])
                        T.reads(A[vi, vk], B[vj, vk])
                        T.writes(D[vi, vj])
                        with T.init():
                            T.buffer_store(D, T.cast(0.0, C_TYPE), [vi, vj])
                        T.buffer_store(
                            D,
                            D[vi, vj] + A[vi, vk] * B[vj, vk],
                            [vi, vj],
                        )
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((m, k), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((n, k), B_TYPE))  # pylint: disable=invalid-name
                C = R.call_tir(
                    frame.global_vars[GLOBAL_SYMBOL],
                    args=[A, B],
                    shape=(m, n),
                    dtype=C_TYPE,
                )
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def test_cutlass_trans_dense2():
    m, n, k = 128, 128, 128
    assert build(constructTransGEMM2(m, n, k)), "build failure on CUDA"
    dev = tvm.cuda()
    A = np.random.rand(m, k).astype("float16") * 5
    B = np.random.rand(n, k).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B.T, rtol=1e-2)


# einsum "hij, hjk -> ik"
def constructReductionGEMM(b, m, n, k, GLOBAL_SYMBOL="ReductionHGEMM"):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg("A", T.buffer_decl((b, m, k), A_TYPE))  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((b, k, n), B_TYPE))  # pylint: disable=invalid-name
                D = T.alloc_buffer((m, n), C_TYPE)
                with T.grid(b, m, n, k) as (lb, l0, l1, l2):
                    with T.block("reduction_dense_row_row_row"):
                        vb, vi, vj, vk = T.axis.remap("RSSR", [lb, l0, l1, l2])
                        T.reads(A[vb, vi, vk], B[vb, vk, vj])
                        T.writes(D[vi, vj])
                        with T.init():
                            T.buffer_store(D, T.cast(0.0, C_TYPE), [vi, vj])
                        T.buffer_store(
                            D,
                            D[vi, vj] + A[vb, vi, vk] * B[vb, vk, vj],
                            [vi, vj],
                        )
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((b, m, k), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((b, k, n), B_TYPE))  # pylint: disable=invalid-name
                C = R.call_tir(
                    frame.global_vars[GLOBAL_SYMBOL],
                    args=[A, B],
                    shape=(m, n),
                    dtype=C_TYPE,
                )
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def test_cutlass_reduction_dense():
    b, m, n, k = 2, 128, 128, 128
    assert build(constructReductionGEMM(b, m, n, k)), "build failure on CUDA"
    dev = tvm.cuda()
    A = np.random.rand(b, m, k).astype("float16") * 5
    B = np.random.rand(b, k, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), np.einsum("hij, hjk->ik", A, B), rtol=1e-2)


# einsum "ghij, hjk -> gik"
def constructReductionGEMM2(b1, b2, m, n, k, GLOBAL_SYMBOL="ReductionHGEMM2"):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg(
                    "A", T.buffer_decl((b1, b2, m, k), A_TYPE)
                )  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((b2, k, n), B_TYPE))  # pylint: disable=invalid-name
                D = T.alloc_buffer((b1, m, n), C_TYPE)
                with T.grid(b1, b2, m, n, k) as (lb1, lb2, l0, l1, l2):
                    with T.block("reduction_dense_row_row_row"):
                        vb1, vb2, vi, vj, vk = T.axis.remap("SRSSR", [lb1, lb2, l0, l1, l2])
                        T.reads(A[vb1, vb2, vi, vk], B[vb2, vk, vj])
                        T.writes(D[vb1, vi, vj])
                        with T.init():
                            T.buffer_store(D, T.cast(0.0, C_TYPE), [vb1, vi, vj])
                        T.buffer_store(
                            D,
                            D[vb1, vi, vj] + A[vb1, vb2, vi, vk] * B[vb2, vk, vj],
                            [vb1, vi, vj],
                        )
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((b1, b2, m, k), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((b2, k, n), B_TYPE))  # pylint: disable=invalid-name
                C = R.call_tir(
                    frame.global_vars[GLOBAL_SYMBOL],
                    args=[A, B],
                    shape=(b1, m, n),
                    dtype=C_TYPE,
                )
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def test_cutlass_reduction_dense2():
    b1, b2, m, n, k = 2, 3, 128, 128, 128
    assert build(constructReductionGEMM2(b1, b2, m, n, k)), "build failure on CUDA"
    dev = tvm.cuda()
    A = np.random.rand(b1, b2, m, k).astype("float16") * 5
    B = np.random.rand(b2, k, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), np.einsum("ghij, hjk->gik", A, B), rtol=1e-2)


# einsum "ghij, gjk -> hik"
def constructReductionGEMM3(b1, b2, m, n, k, GLOBAL_SYMBOL="ReductionHGEMM3"):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg(
                    "A", T.buffer_decl((b1, b2, m, k), A_TYPE)
                )  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((b1, k, n), B_TYPE))  # pylint: disable=invalid-name
                D = T.alloc_buffer((b2, m, n), C_TYPE)
                with T.grid(b1, b2, m, n, k) as (lb1, lb2, l0, l1, l2):
                    with T.block("reduction_dense_row_row_row"):
                        vb1, vb2, vi, vj, vk = T.axis.remap("RSSSR", [lb1, lb2, l0, l1, l2])
                        T.reads(A[vb1, vb2, vi, vk], B[vb1, vk, vj])
                        T.writes(D[vb2, vi, vj])
                        with T.init():
                            T.buffer_store(D, T.cast(0.0, C_TYPE), [vb2, vi, vj])
                        T.buffer_store(
                            D,
                            D[vb2, vi, vj] + A[vb1, vb2, vi, vk] * B[vb1, vk, vj],
                            [vb2, vi, vj],
                        )
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((b1, b2, m, k), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((b1, k, n), B_TYPE))  # pylint: disable=invalid-name
                C = R.call_tir(
                    frame.global_vars[GLOBAL_SYMBOL],
                    args=[A, B],
                    shape=(b2, m, n),
                    dtype=C_TYPE,
                )
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def test_cutlass_reduction_dense3():
    b1, b2, m, n, k = 2, 3, 128, 128, 128
    assert build(constructReductionGEMM3(b1, b2, m, n, k)), "build failure on CUDA"
    dev = tvm.cuda()
    A = np.random.rand(b1, b2, m, k).astype("float16") * 5
    B = np.random.rand(b1, k, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), np.einsum("ghij, gjk->hik", A, B), rtol=1e-2)


# einsum "ghij, ghjk -> ik"
def constructReductionGEMM4(b1, b2, m, n, k, GLOBAL_SYMBOL="ReductionHGEMM4"):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg(
                    "A", T.buffer_decl((b1, b2, m, k), A_TYPE)
                )  # pylint: disable=invalid-name
                B = T.arg(
                    "B", T.buffer_decl((b1, b2, k, n), B_TYPE)
                )  # pylint: disable=invalid-name
                D = T.alloc_buffer((m, n), C_TYPE)
                with T.grid(b1, b2, m, n, k) as (lb1, lb2, l0, l1, l2):
                    with T.block("reduction_dense_row_row_row"):
                        vb1, vb2, vi, vj, vk = T.axis.remap("RRSSR", [lb1, lb2, l0, l1, l2])
                        T.reads(A[vb1, vb2, vi, vk], B[vb1, vb2, vk, vj])
                        T.writes(D[vi, vj])
                        with T.init():
                            T.buffer_store(D, T.cast(0.0, C_TYPE), [vi, vj])
                        T.buffer_store(
                            D,
                            D[vi, vj] + A[vb1, vb2, vi, vk] * B[vb1, vb2, vk, vj],
                            [vi, vj],
                        )
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((b1, b2, m, k), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((b1, b2, k, n), B_TYPE))  # pylint: disable=invalid-name
                C = R.call_tir(
                    frame.global_vars[GLOBAL_SYMBOL],
                    args=[A, B],
                    shape=(m, n),
                    dtype=C_TYPE,
                )
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def test_cutlass_reduction_dense4():
    b1, b2, m, n, k = 2, 3, 128, 128, 128
    assert build(constructReductionGEMM4(b1, b2, m, n, k)), "build failure on CUDA"
    dev = tvm.cuda()
    A = np.random.rand(b1, b2, m, k).astype("float16") * 5
    B = np.random.rand(b1, b2, k, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), np.einsum("ghij, ghjk->ik", A, B), rtol=1e-2)


# einsum "hij, ijk -> hik"
def constructPermutationGEMM(b, m, n, k, GLOBAL_SYMBOL="PermutationHGEMM"):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg("A", T.buffer_decl((b, m, k), A_TYPE))  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((m, k, n), B_TYPE))  # pylint: disable=invalid-name
                D = T.alloc_buffer((b, m, n), C_TYPE)
                with T.grid(b, m, n, k) as (lb, l0, l1, l2):
                    with T.block("permutation_dense_row_row_row"):
                        vb, vi, vj, vk = T.axis.remap("SSSR", [lb, l0, l1, l2])
                        T.reads(A[vb, vi, vk], B[vi, vk, vj])
                        T.writes(D[vb, vi, vj])
                        with T.init():
                            T.buffer_store(D, T.cast(0.0, C_TYPE), [vb, vi, vj])
                        T.buffer_store(
                            D,
                            D[vb, vi, vj] + A[vb, vi, vk] * B[vi, vk, vj],
                            [vb, vi, vj],
                        )
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((b, m, k), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((m, k, n), B_TYPE))  # pylint: disable=invalid-name
                C = R.call_tir(
                    frame.global_vars[GLOBAL_SYMBOL],
                    args=[A, B],
                    shape=(b, m, n),
                    dtype=C_TYPE,
                )
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def test_cutlass_permutation_dense():
    b, m, n, k = 2, 128, 128, 128
    assert build(constructPermutationGEMM(b, m, n, k)), "build failure on CUDA"
    dev = tvm.cuda()
    A = np.random.rand(b, m, k).astype("float16") * 5
    B = np.random.rand(m, k, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), np.einsum("hij, ijk->hik", A, B), rtol=1e-2)


if __name__ == "__main__":
    # test_cutlass_dense()
    # test_cutlass_dense_bias()
    # test_cutlass_dense_bias_relu()
    # test_cutlass_batch_dense()
    # test_cutlass_batch_dense2()
    # test_cutlass_batch_dense_bias()
    # test_cutlass_batch_dense2_bias()
    # test_cutlass_multi_batch_dense()
    # test_cutlass_multi_batch_dense2()
    # test_cutlass_multi_batch_dense3()
    # test_cutlass_trans_dense()
    # test_cutlass_trans_dense2()
    # test_cutlass_reduction_dense()
    # test_cutlass_reduction_dense2()
    # test_cutlass_reduction_dense3()
    # test_cutlass_reduction_dense4()
    test_cutlass_permutation_dense()
    print("passed test")

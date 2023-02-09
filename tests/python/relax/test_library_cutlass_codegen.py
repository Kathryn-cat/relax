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
import tvm.testing
from tvm import relax, runtime
from tvm.relax.library import cutlass_codegen_with_match_results, get_cutlass_pattern
from tvm.relax.transform import LegalizeOps
from tvm.relax.vm import build as relax_build

"""
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I
from tvm.script.ir_builder import relax as R
from tvm.script.ir_builder import tir as T
"""

PKG_FILE = "/tmp/test_transform_cutlass_codegen.so"
GLOBAL_SYMBOL = "HGEMM"
A_TYPE = "float16"
B_TYPE = "float16"
C_TYPE = "float16"

target = "cuda"


def f_run(rt_mod: runtime.Module, device: runtime.ndarray.Device, *input):
    vm = relax.vm.VirtualMachine(exec=rt_mod, device=device)
    return vm["main"](*input)


def build(mod, file_name=PKG_FILE):
    mod = relax.transform.LegalizeOps()(mod)
    mod = relax.transform.AnnotateTIROpPattern()(mod)
    mod = relax.transform.FuseOps()(mod)
    mod = relax.transform.FuseTIR()(mod)
    mod = relax.transform.PreProcess()(mod)
    mod.show()
    mod = relax.transform.SplitCallTIRByPattern(
        get_cutlass_pattern(), cutlass_codegen_with_match_results
    )(mod)
    mod.show()
    executbale = relax_build(mod, target)
    executbale.mod.export_library(file_name, cc="nvcc")
    return executbale


def constructGEMM(M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.matmul(A, B, out_dtype=C_TYPE))
                    R.output(C)
                (C,) = df.output_vars
                R.func_ret_value(C)
    relax_mod = ib.get()
    return relax_mod


@tvm.testing.requires_cutlass
def test_cutlass_dense():
    m, n, k = 128, 128, 128
    build(constructGEMM(m, n, k))
    dev = tvm.cuda()
    A = np.random.rand(m, k).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B, rtol=1e-2)


def constructGEMM_bias(M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                bias = R.arg(
                    "bias", relax.TensorStructInfo((1, N), A_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.matmul(A, B, out_dtype=C_TYPE))
                    D = R.emit(R.add(C, bias))
                    R.output(D)
                (D,) = df.output_vars
                R.func_ret_value(D)
    relax_mod = ib.get()
    return relax_mod


@tvm.testing.requires_cutlass
def test_cutlass_dense_bias():
    m, n, k = 128, 128, 128
    build(constructGEMM_bias(m, n, k))
    dev = tvm.cuda()
    A = np.random.rand(m, k).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    bias = np.random.rand(1, n).astype("float16") * 20
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B + bias, rtol=1e-2)


def constructGEMM_bias_relu(M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                bias = R.arg(
                    "bias", relax.TensorStructInfo((1, N), A_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.matmul(A, B, out_dtype=C_TYPE))
                    D = R.emit(R.add(C, bias))
                    E = R.emit(R.nn.relu(D))
                    R.output(E)
                (E,) = df.output_vars
                R.func_ret_value(E)
    relax_mod = ib.get()
    return relax_mod


@tvm.testing.requires_cutlass
def test_cutlass_dense_bias_relu():
    m, n, k = 128, 128, 128
    build(constructGEMM_bias_relu(m, n, k))
    dev = tvm.cuda()
    A = np.random.rand(m, k).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    bias = np.random.rand(1, n).astype("float16") * 20
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
    np.testing.assert_allclose(result.numpy(), np.maximum(A @ B + bias, 0), rtol=1e-2)


def constructBatchGEMM(batch, M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((batch, M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.matmul(A, B, out_dtype=C_TYPE))
                    R.output(C)
                (C,) = df.output_vars
                R.func_ret_value(C)
    relax_mod = ib.get()
    return relax_mod


@tvm.testing.requires_cutlass
def test_cutlass_batch_dense():
    b, m, n, k = 2, 128, 128, 128
    build(constructBatchGEMM(b, m, n, k))
    dev = tvm.cuda()
    A = np.random.rand(b, m, k).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B, rtol=1e-2)


def constructBatchGEMM2(batch, M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((batch, M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((batch, K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.matmul(A, B, out_dtype=C_TYPE))
                    R.output(C)
                (C,) = df.output_vars
                R.func_ret_value(C)
    relax_mod = ib.get()
    return relax_mod


@tvm.testing.requires_cutlass
def test_cutlass_batch_dense2():
    b, m, n, k = 2, 128, 128, 128
    build(constructBatchGEMM2(b, m, n, k))
    dev = tvm.cuda()
    A = np.random.rand(b, m, k).astype("float16") * 5
    B = np.random.rand(b, k, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B, rtol=1e-2)


def constructBatchGEMM_bias(batch, M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((batch, M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                bias = R.arg(
                    "bias", relax.TensorStructInfo((1, N), A_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.matmul(A, B, out_dtype=C_TYPE))
                    D = R.emit(R.add(C, bias))
                    R.output(D)
                (D,) = df.output_vars
                R.func_ret_value(D)
    relax_mod = ib.get()
    return relax_mod


@tvm.testing.requires_cutlass
def test_cutlass_batch_dense_bias():
    b, m, n, k = 2, 128, 128, 128
    build(constructBatchGEMM_bias(b, m, n, k))
    dev = tvm.cuda()
    A = np.random.rand(b, m, k).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    bias = np.random.rand(1, n).astype("float16") * 20
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B + bias, rtol=1e-2)


def constructBatchGEMM2_bias(batch, M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((batch, M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((batch, K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                bias = R.arg(
                    "bias", relax.TensorStructInfo((1, N), A_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.matmul(A, B, out_dtype=C_TYPE))
                    D = R.emit(R.add(C, bias))
                    R.output(D)
                (D,) = df.output_vars
                R.func_ret_value(D)
    relax_mod = ib.get()
    return relax_mod


@tvm.testing.requires_cutlass
def test_cutlass_batch_dense2_bias():
    b, m, n, k = 2, 128, 128, 128
    build(constructBatchGEMM2_bias(b, m, n, k))
    dev = tvm.cuda()
    A = np.random.rand(b, m, k).astype("float16") * 5
    B = np.random.rand(b, k, n).astype("float16") * 5
    bias = np.random.rand(1, n).astype("float16") * 20
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B + bias, rtol=1e-2)


def constructConv2D(N, C, H, W, KH, KW, O, strides, padding, dilation):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                x = R.arg(
                    "x", relax.TensorStructInfo((N, H, W, C), A_TYPE)
                )  # pylint: disable=invalid-name
                w = R.arg(
                    "w", relax.TensorStructInfo((O, KH, KW, C), B_TYPE)
                )  # pylint: disable=invalid-name
                C = R.nn.conv2d(
                    x,
                    w,
                    strides=strides,
                    padding=padding,
                    dilation=dilation,
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype=C_TYPE,
                )
                R.func_ret_value(C)
    mod = ib.get()
    return mod


@tvm.testing.requires_cutlass
def test_cutlass_conv2d():
    import torch

    n, c, h, w = 1, 3, 224, 224
    kh, kw, o = 3, 3, 64
    # strides = (1, 1)
    # padding = (3, 3)
    # dilation = (1, 1)
    counter = 0
    for strides in [(1, 1), (2, 2)]:
        for padding in [(0, 0), (3, 3)]:
            for dilation in [(1, 1), (4, 4)]:
                filename = "/tmp/" + "test_transform_cutlass_codegen" + str(counter) + ".so"
                build(constructConv2D(n, c, h, w, kh, kw, o, strides, padding, dilation), filename)
                dev = tvm.cuda()
                np.random.seed(0)
                A = np.random.rand(n, h, w, c).astype("float16") * 5
                B = np.random.rand(o, kh, kw, c).astype("float16") * 5
                A_tvm = tvm.nd.array(A, dev)
                B_tvm = tvm.nd.array(B, dev)
                executable = tvm.runtime.load_module(filename)
                result = f_run(executable, dev, A_tvm, B_tvm)
                A_torch = torch.from_numpy(np.transpose(A, (0, 3, 1, 2))).cuda()
                B_torch = torch.from_numpy(np.transpose(B, (0, 3, 1, 2))).cuda()
                C_torch = torch.nn.functional.conv2d(
                    A_torch, B_torch, stride=strides, padding=padding, dilation=dilation
                )
                np.testing.assert_allclose(
                    np.transpose(result.numpy(), (0, 3, 1, 2)), C_torch.cpu().numpy(), rtol=1e-2
                )
                counter += 1


# ------------------------ einsum tests below ------------------------
# -------------- (change from IR Builder to tvm script) --------------


from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


def constructTransGEMM(m, n, k):
    @tvm.script.ir_module
    class TransHGEMM:
        @T.prim_func
        def hgemm(x: T.handle, y: T.handle, z: T.handle) -> None:
            A = T.match_buffer(x, (k, m), A_TYPE)  # pylint: disable=invalid-name
            B = T.match_buffer(y, (k, n), B_TYPE)  # pylint: disable=invalid-name
            C = T.match_buffer(z, (m, n), C_TYPE)  # pylint: disable=invalid-name
            for l0, l1, l2 in T.grid(m, n, k):
                with T.block("dense_row_row_row"):
                    vi, vj, vk = T.axis.remap("SSR", [l0, l1, l2])
                    T.reads(A[vk, vi], B[vk, vj])
                    T.writes(C[vi, vj])
                    with T.init():
                        C[vi, vj] = T.cast(0.0, C_TYPE)
                    C[vi, vj] += A[vk, vi] * B[vk, vj]

        @R.function
        def main(A: R.Tensor((k, m), A_TYPE), B: R.Tensor((k, n), B_TYPE)):
            with R.dataflow():
                C: R.Tensor((m, n), C_TYPE) = R.call_tir(hgemm, (A, B), R.Tensor((m, n), C_TYPE))
                R.output(C)
            return C

    return TransHGEMM


def constructTransGEMMTarget(m, n, k):
    @tvm.script.ir_module
    class TransHGEMMTarget:
        @T.prim_func
        def trans(
            rxplaceholder: T.Buffer((T.int64(64), T.int64(32)), "float16"),
            T_transpose: T.Buffer((T.int64(32), T.int64(64)), "float16"),
        ):
            T.func_attr({"global_symbol": "trans", "op_pattern": 2, "tir.noalias": True})
            for ax0_ax1_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
                for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                    with T.block("T_transpose"):
                        v_ax0 = T.axis.spatial(
                            T.int64(32),
                            (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(64),
                        )
                        v_ax1 = T.axis.spatial(
                            T.int64(64),
                            (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(64),
                        )
                        T.reads(rxplaceholder[v_ax1, v_ax0])
                        T.writes(T_transpose[v_ax0, v_ax1])
                        T_transpose[v_ax0, v_ax1] = rxplaceholder[v_ax1, v_ax0]

        @R.function
        def main(A: R.Tensor((k, m), A_TYPE), B: R.Tensor((k, n), B_TYPE)):
            with R.dataflow():
                A_new: R.Tensor((m, k), A_TYPE) = R.call_tir(trans, A, R.Tensor((m, k), A_TYPE))
                C: R.Tensor((m, n), C_TYPE) = R.matmul(A_new, B, out_dtype=C_TYPE)
                R.output(C)
            return C

    return TransHGEMMTarget


# debug access to host memory issue
def constructTrans(m, k):
    @tvm.script.ir_module
    class Trans:
        @T.prim_func
        def trans(
            rxplaceholder: T.Buffer((T.int64(64), T.int64(32)), "float16"),
            T_transpose: T.Buffer((T.int64(32), T.int64(64)), "float16"),
        ):
            T.func_attr({"global_symbol": "trans", "op_pattern": 2, "tir.noalias": True})
            for ax0_ax1_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
                for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                    with T.block("T_transpose"):
                        v_ax0 = T.axis.spatial(
                            T.int64(32),
                            (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(64),
                        )
                        v_ax1 = T.axis.spatial(
                            T.int64(64),
                            (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(64),
                        )
                        T.reads(rxplaceholder[v_ax1, v_ax0])
                        T.writes(T_transpose[v_ax0, v_ax1])
                        T_transpose[v_ax0, v_ax1] = rxplaceholder[v_ax1, v_ax0]

        @R.function
        def main(A: R.Tensor((k, m), A_TYPE)):
            with R.dataflow():
                A_new: R.Tensor((m, k), A_TYPE) = R.call_tir(trans, A, R.Tensor((m, k), A_TYPE))
                R.output(A_new)
            return A_new

    return Trans


@tvm.testing.requires_cutlass
def test_cutlass_dense_trans():
    m, n, k = 32, 128, 64
    # mod = constructTransGEMM(m, n, k)
    mod = constructTransGEMMTarget(m, n, k)
    build(mod)
    dev = tvm.cuda()
    A = np.random.rand(k, m).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), A.T @ B, rtol=1e-2)


def test_trans():
    m, k = 32, 64
    mod = constructTrans(m, k)
    build(mod)
    dev = tvm.cuda()
    A = np.random.rand(k, m).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm)
    np.testing.assert_allclose(result.numpy(), A.T, rtol=1e-2)


if __name__ == "__main__":
    """
    test_cutlass_dense()
    test_cutlass_dense_bias()
    test_cutlass_dense_bias_relu()
    test_cutlass_batch_dense()
    test_cutlass_batch_dense2()
    test_cutlass_batch_dense_bias()
    test_cutlass_batch_dense2_bias()
    test_cutlass_conv2d()
    """
    test_cutlass_dense_trans()

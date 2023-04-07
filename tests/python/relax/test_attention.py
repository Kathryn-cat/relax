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
from tvm.relax.transform import FuseOps, LegalizeOps
from tvm.script import relax as R
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import relax as relax_builder


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

    mod = partition_for_cutlass(mod)
    print("after fuse ops by pattern:")
    mod.show()

    if assert_all_bindings_fused:
        assert len(mod["main"].body.blocks[0].bindings) == 1

    codegen_pass = relax.transform.RunCodegen({"cutlass": {"sm": 80, "find_first_valid": True}})
    mod = codegen_pass(mod)

    return build_and_run(mod, args, "cuda")


def get_relax_attention_module(q, k, v, bias=None, qk_scale=None):
    dtype = str(q.dtype)

    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import relax as relax_builder
    from tvm.script.ir_builder import tir as T

    if qk_scale is not None:
        qk_scale = T.FloatImm("float32", qk_scale)

    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name("main")
            q = R.arg("q", R.Tensor(q.shape, dtype))
            k = R.arg("k", R.Tensor(k.shape, dtype))
            v = R.arg("v", R.Tensor(v.shape, dtype))
            if bias is not None:
                bias = R.arg("bias", R.Tensor(bias.shape, dtype))
            with R.dataflow() as frame:
                result = R.emit(R.nn.attention(q, k, v, bias, qk_scale))
                R.output(result)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm.IRModule({"main": func})


@memoize("topi.tests.test_codegen_cutlass.test_attention_offload")
def get_numpy_attention_ref(b, s, s_kv, n, h, h_v, bias_shape, bias_reshape, qk_scale, dtype):
    q = np.random.randn(b, s, n, h).astype(dtype)
    k = np.random.randn(b, s_kv, n, h).astype(dtype)
    v = np.random.randn(b, s_kv, n, h_v).astype(dtype)
    qt = q.transpose(0, 2, 1, 3)  # b, n, s, h
    kt = k.transpose(0, 2, 3, 1)  # b, n, h, s_kv
    if not qk_scale == "none":
        score = qt @ kt * qk_scale  # b, n, s, s_kv
    else:
        score = qt @ kt / np.sqrt(q.shape[-1])  # b, n, s, s_kv
    if not bias_shape == "none":
        bias = np.random.randn(*bias_shape).astype(dtype)
        score = score + bias.reshape(*bias_reshape)  # b, n, s, s_kv
    else:
        bias = None
    attn = tvm.topi.testing.softmax_python(score, -1)
    vt = v.transpose(0, 2, 1, 3)  # b, n, s_kv, h_v
    ref = attn @ vt  # b, n, s, h_v
    return q, k, v, bias, ref.transpose(0, 2, 1, 3)  # b, s, n, h_v


def test_attention_offload():
    b, (s, s_kv), n, (h, h_v) = 4, (16, 8), 32, (8, 16)
    q, k, v, _, ref = get_numpy_attention_ref(
        b, s, s_kv, n, h, h_v, "none", "none", "none", "float32"
    )

    mod = get_relax_attention_module(q, k, v)
    print("original mod:")
    mod.show()
    out = get_result_with_relax_cutlass_offload(mod, q, k, v)

    tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


def get_relax_attention_module_sd(b, s, s_kv, n, h, h_v):
    dtype = "float32"

    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import relax as relax_builder
    from tvm.script.ir_builder import tir as T

    scale = R.const((1 / np.sqrt(h)).astype(dtype))

    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name("main")
            q = R.arg("q", R.Tensor((b, s, n, h), dtype))
            k = R.arg("k", R.Tensor((b, s_kv, n, h), dtype))
            v = R.arg("v", R.Tensor((b, s_kv, n, h_v), dtype))
            with R.dataflow() as frame:
                qt = R.emit(R.permute_dims(q, axes=[0, 2, 1, 3]))
                qt = R.emit(R.reshape(qt, (b * n, s, h)))
                kt = R.emit(R.permute_dims(k, axes=[0, 2, 1, 3]))
                kt = R.emit(R.reshape(kt, (b * n, s_kv, h)))
                vt = R.emit(R.permute_dims(v, axes=[0, 2, 1, 3]))
                vt = R.emit(R.reshape(vt, (b * n, s_kv, h_v)))
                kt = R.emit(R.permute_dims(kt, axes=[0, 2, 1]))
                score = R.emit(R.matmul(qt, kt, out_dtype=dtype))
                score = R.emit(R.multiply(score, scale))
                attn = R.emit(R.nn.softmax(score, axis=-1))
                result = R.emit(R.matmul(attn, vt, out_dtype=dtype))
                result = R.emit(R.reshape(result, (b, n, s, h_v)))
                result = R.emit(R.permute_dims(result, axes=[0, 2, 1, 3]))
                R.output(result)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm.IRModule({"main": func})


def test_attention_offload_sd():
    b, (s, s_kv), n, (h, h_v) = 2, (4096, 4096), 8, (40, 40)
    q, k, v, _, ref = get_numpy_attention_ref(
        b, s, s_kv, n, h, h_v, "none", "none", "none", "float32"
    )

    mod = get_relax_attention_module_sd(b, s, s_kv, n, h, h_v)
    print("original mod:")
    mod.show()

    # ----- change the mod to packed version -----

    patterns = [(entry.name, entry.pattern) for entry in get_patterns_with_prefix("cutlass")]
    assert len(patterns) != 0, "Cannot find cutlass patterns"
    print(f"number of cutlass patterns: {len(patterns)}")
    for i, (name, pattern) in enumerate(patterns):
        print(f"{i}: name: {name}\npattern: {pattern}\n")
    print("--------------------------------")

    print("after fused ops:")
    mod = partition_for_cutlass(mod)
    mod.show()

    print("begin codegen:")
    codegen_pass = relax.transform.RunCodegen({"cutlass": {"sm": 80, "find_first_valid": True}})
    mod = codegen_pass(mod)

    print(f"begin legalize ops:")
    mod = LegalizeOps()(mod)

    print(f"begin fusing ops:")
    mod = FuseOps()(mod)

    out = build_and_run(mod, [q, k, v], "cuda", legalize=True)
    tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


def get_relax_attention_module_sd_2(b, s, s_kv, n, h, h_v):
    dtype = "float32"

    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import relax as relax_builder
    from tvm.script.ir_builder import tir as T

    scale = R.const((1 / np.sqrt(h)).astype(dtype))

    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name("main")
            q = R.arg("q", R.Tensor((b, s, n, h), dtype))
            k = R.arg("k", R.Tensor((b, s_kv, n, h), dtype))
            v = R.arg("v", R.Tensor((b, s_kv, n, h_v), dtype))
            with R.dataflow() as frame:
                qt = R.emit(R.permute_dims(q, axes=[0, 2, 1, 3]))
                qt = R.emit(R.reshape(qt, (b * n, s, h)))
                kt = R.emit(R.permute_dims(k, axes=[0, 2, 1, 3]))
                kt = R.emit(R.reshape(kt, (b * n, s_kv, h)))
                vt = R.emit(R.permute_dims(v, axes=[0, 2, 1, 3]))
                vt = R.emit(R.reshape(vt, (b * n, s_kv, h_v)))
                kt = R.emit(R.permute_dims(kt, axes=[0, 2, 1]))
                score = R.emit(R.matmul(qt, kt, out_dtype=dtype))
                score = R.emit(R.multiply(score, scale))
                score = R.emit(R.astype(score, dtype=dtype))
                attn = R.emit(R.nn.softmax(score, axis=-1))
                attn = R.emit(R.astype(attn, dtype=dtype))
                result = R.emit(R.matmul(attn, vt, out_dtype=dtype))
                result = R.emit(R.reshape(result, (b, n, s, h_v)))
                result = R.emit(R.permute_dims(result, axes=[0, 2, 1, 3]))
                R.output(result)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm.IRModule({"main": func})


def test_attention_offload_sd_2():
    b, (s, s_kv), n, (h, h_v) = 1, (4096, 4096), 1, (512, 512)
    q, k, v, _, ref = get_numpy_attention_ref(
        b, s, s_kv, n, h, h_v, "none", "none", "none", "float32"
    )

    mod = get_relax_attention_module_sd_2(b, s, s_kv, n, h, h_v)
    print("original mod:")
    mod.show()

    # ----- change the mod to packed version -----

    patterns = [(entry.name, entry.pattern) for entry in get_patterns_with_prefix("cutlass")]
    assert len(patterns) != 0, "Cannot find cutlass patterns"
    print(f"number of cutlass patterns: {len(patterns)}")
    for i, (name, pattern) in enumerate(patterns):
        print(f"{i}: name: {name}\npattern: {pattern}\n")
    print("--------------------------------")

    print("after fused ops:")
    mod = partition_for_cutlass(mod)
    mod.show()

    print("begin codegen:")
    codegen_pass = relax.transform.RunCodegen({"cutlass": {"sm": 80, "find_first_valid": True}})
    mod = codegen_pass(mod)

    print(f"begin legalize ops:")
    mod = LegalizeOps()(mod)

    print(f"begin fusing ops:")
    mod = FuseOps()(mod)

    out = build_and_run(mod, [q, k, v], "cuda", legalize=True)
    tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    # test_attention_offload()
    # test_attention_offload_sd()
    test_attention_offload_sd_2()

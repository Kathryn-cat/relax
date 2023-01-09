/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/struct_info.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tir/expr.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <iterator>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace tvm;
using namespace tvm::runtime;
using namespace tvm::relax;

TEST(NestedMsg, Basic) {
  // start with no annotation
  relax::Var x("x", NullOpt), y("y", NullOpt);

  // constructor from array, T and nullopt.
  NestedMsg<relax::Expr> msg({x, NullOpt, x});

  EXPECT_TRUE(msg.IsNested());
  EXPECT_FALSE(msg.IsLeaf());
  EXPECT_TRUE(msg != nullptr);

  EXPECT_ANY_THROW(msg.LeafValue());

  auto arr = msg.NestedArray();
  EXPECT_TRUE(arr[0].same_as(x));
  EXPECT_TRUE(arr[1] == nullptr);
  EXPECT_TRUE(arr[1].IsNull());

  EXPECT_TRUE(arr[2].LeafValue().same_as(x));

  auto a0 = arr[0];
  EXPECT_TRUE(a0.IsLeaf());

  // assignment
  // assign null
  a0 = NullOpt;
  EXPECT_TRUE(a0 == nullptr);

  // assign array
  a0 = {x, {x, NullOpt, y}};
  EXPECT_TRUE(a0.IsNested());
  auto t0 = a0.NestedArray()[1];
  EXPECT_TRUE(t0.IsNested());
  EXPECT_TRUE(t0.NestedArray()[2].same_as(y));

  // assign leaf
  a0 = x;

  EXPECT_TRUE(a0.IsLeaf());
  EXPECT_TRUE(a0.same_as(x));
}

TEST(NestedMsg, ForEachLeaf) {
  relax::Var x("x", NullOpt), y("y", NullOpt);
  NestedMsg<Expr> msg = {x, {x, y}, NullOpt, {x, {x, y}}};

  int x_count = 0, y_count = 0;

  ForEachLeaf(msg, [&](const Expr& v) {
    if (v.same_as(x)) ++x_count;
    if (v.same_as(y)) ++y_count;
  });
  EXPECT_EQ(x_count, 4);
  EXPECT_EQ(y_count, 2);
}

TEST(NestedMsg, Equal) {
  relax::Var x("x", NullOpt), y("y", NullOpt);
  relax::Var z("z", NullOpt);

  auto fequal = [](Expr lhs, Expr rhs) { return lhs.same_as(rhs); };

  using M = NestedMsg<relax::Expr>;

  EXPECT_TRUE(Equal(M(NullOpt), M(NullOpt), fequal));

  EXPECT_TRUE(Equal(M(x), M(x), fequal));

  EXPECT_TRUE(Equal(M({x, y}), M({x, y}), fequal));

  EXPECT_TRUE(Equal(M({x, NullOpt}), M({x, NullOpt}), fequal));

  EXPECT_TRUE(Equal(M({x, {NullOpt, y}}), M({x, {NullOpt, y}}), fequal));

  EXPECT_TRUE(Equal(M({x, {NullOpt, y}, {x, z}}), M({x, {NullOpt, y}, {x, z}}), fequal));

  // type mismatch
  EXPECT_FALSE(Equal(M({x, {NullOpt, y}, x}), M({x, {NullOpt, y}, {x, z}}), fequal));

  EXPECT_FALSE(Equal(M({x, {NullOpt, y}, {x, NullOpt}}), M({x, {NullOpt, y}, {x, z}}), fequal));

  EXPECT_FALSE(Equal(M({x, {NullOpt, y}}), M({x, {NullOpt, y}, {x, z}}), fequal));

  EXPECT_FALSE(Equal(M(x), M(NullOpt), fequal));

  EXPECT_FALSE(Equal(M(NullOpt), M(x), fequal));

  EXPECT_FALSE(Equal(M(x), M({x}), fequal));

  EXPECT_FALSE(Equal(M({x}), M(x), fequal));
}

TEST(NestedMsg, MapAndDecompose) {
  relax::Var x("x", PrimStructInfo(runtime::DataType::Int(16)));
  relax::Var y("y", PrimStructInfo(runtime::DataType::Int(32)));
  relax::Var z("z", PrimStructInfo(runtime::DataType::Int(64)));

  BlockBuilder bb = BlockBuilder::Create(NullOpt);
  relax::Expr t0 = bb->Normalize(Tuple({x, y}));
  relax::Expr t1 = bb->Normalize(Tuple({t0, x, z, t0}));

  auto c0 = Integer(0);
  auto c1 = Integer(1);
  auto c2 = Integer(2);

  auto output = MapToNestedMsg<Integer>(t1, [&](Expr value) {
    if (value.same_as(x)) return c0;
    if (value.same_as(y)) return c1;
    return c2;
  });

  NestedMsg<Integer> expected = {{c0, c1}, c0, c2, {c0, c1}};

  EXPECT_TRUE(Equal(output, expected,
                    [](Integer lhs, Integer rhs) -> bool { return lhs->value == rhs->value; }));

  auto output2 =
      MapToNestedMsg<Integer>(GetStructInfo(t1), [&](StructInfo sinfo) -> NestedMsg<Integer> {
        const auto* prim_sinfo = sinfo.as<PrimStructInfoNode>();
        if (prim_sinfo == nullptr) return NullOpt;
        int bits = prim_sinfo->dtype.bits();
        if (bits == 16) return c0;
        if (bits == 32) return c1;
        if (bits == 64) return c2;
        return NullOpt;
      });

  EXPECT_TRUE(Equal(output2, expected,
                    [](Integer lhs, Integer rhs) -> bool { return lhs->value == rhs->value; }));

  int x_count = 0, y_count = 0, z_count = 0;

  DecomposeNestedMsg(t1, expected, [&](Expr value, NestedMsg<Integer> msg) {
    if (value.same_as(x)) {
      EXPECT_TRUE(msg.same_as(c0));
      ++x_count;
    } else if (value.same_as(y)) {
      EXPECT_TRUE(msg.same_as(c1));
      ++y_count;
    } else {
      EXPECT_TRUE(msg.same_as(c2));
      ++z_count;
    }
  });
  EXPECT_EQ(x_count, 3);
  EXPECT_EQ(y_count, 2);
  EXPECT_EQ(z_count, 1);
}

TEST(NestedMsg, CombineNestedMsg) {
  auto c0 = Integer(0);
  auto c1 = Integer(1);
  auto c2 = Integer(2);

  NestedMsg<Integer> lhs = {c0, {c0, c1}, NullOpt, {c0, {c1, c2}}};
  NestedMsg<Integer> rhs = {c1, {c2, NullOpt}, NullOpt, {c1, {c2, c2}}};
  NestedMsg<Integer> expected = {c1, {c2, c1}, NullOpt, {c1, {c2, c2}}};

  auto output = CombineNestedMsg(lhs, rhs, [](Integer x, Integer y) {
    if (x->value > y->value) return x;
    return y;
  });

  EXPECT_TRUE(Equal(output, expected,
                    [](Integer lhs, Integer rhs) -> bool { return lhs->value == rhs->value; }));
}

TEST(NestedMsg, TransformTupleLeaf) {
  auto c0 = Integer(0);
  auto c1 = Integer(1);
  auto c2 = Integer(2);
  using NInt = NestedMsg<Integer>;

  NInt msg1 = {c0, {c0, c1}, c2, {c0, {c1, c2}}};
  NInt msg2 = {c1, {c2, c0}, c2, {c1, {c2, c0}}};

  PrimStructInfo s = PrimStructInfo(runtime::DataType::Int(32));
  relax::Var x("x", s), y("y", s), z("z", s);
  BlockBuilder bb = BlockBuilder::Create(NullOpt);
  Expr expr = bb->Normalize(Tuple({x, Tuple({x, x}), x, Tuple({x, Tuple({x, x})})}));

  auto ftransleaf = [&](Expr value, std::array<NInt, 2> msgs) -> Expr {
    int lhs = Downcast<Integer>(msgs[0].LeafValue())->value;
    int rhs = Downcast<Integer>(msgs[1].LeafValue())->value;
    if (lhs > rhs)
      return z;
    else if (lhs == rhs)
      return value;
    else
      return y;
  };

  Expr expected = Tuple({y, Tuple({y, z}), x, Tuple({y, Tuple({y, z})})});

  EXPECT_TRUE(StructuralEqual()(
      TransformTupleLeaf(expr, std::array<NInt, 2>({msg1, msg2}), ftransleaf), expected));
}

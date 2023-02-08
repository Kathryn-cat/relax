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
/*!
 * \file src/relax/transform/split_cutlass.cc
 * \brief Dispatch graph-level tir to cutlass.
 */
#include "./pattern_matching.h"
#include "./split_functions.h"

namespace tvm {
namespace relax {

namespace transform {
Pass DispatchCutlass() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
      [=](IRModule m, PassContext pc) {
        // TODO: add pre- and post- transform function calls here
        // in the pre-splitting stage, our first goal is to categorize the axes in the module
        IRModule split_mod = SplitMutator::Transform(/*mod=*/m, /*vendor_type=*/"cutlass");
        return split_mod;
      };
  return CreateModulePass(/*pass_function=*/pass_func,      //
                          /*opt_level=*/0,                  //
                          /*pass_name=*/"DispatchCutlass",  //
                          /*required=*/{});
}
TVM_REGISTER_GLOBAL("relax.transform.DispatchCutlass").set_body_typed(DispatchCutlass);

}  // namespace transform

}  // namespace relax
}  // namespace tvm

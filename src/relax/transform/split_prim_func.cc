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
 * \file src/relax/transform/split_prim_func.cc
 * \brief Transform all dataflow structure to non-dataflow version.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ir/module.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/tir_pattern.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {

namespace tir {

/*! \brief helper class to partition a function into 2 parts. Return function information which we
 * can use to construct the two partitioned parts.*/
class FunctionPartitioner : public StmtExprVisitor {
 public:
  explicit FunctionPartitioner(int num_for_loops) : num_for_loops_(num_for_loops) {}

  /*! \brief array of partitioned blocks */
  std::vector<Block> block_partition;
  /*! \brief alloc_buffers for each function */
  std::vector<std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>> allocs;
  /*! \brief input buffers for each function */
  std::vector<std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>> inputs;
  /*! \brief output buffers for each function */
  std::vector<Buffer> outputs;

 private:
  void VisitStmt_(const BlockNode* op) final { block_partition.push_back(GetRef<Block>(op)); }

  size_t num_for_loops_;
};

/* find out the number of outer for-loops in the body.*/
int CountForLoops(Stmt body) {
  if (body->IsInstance<ForNode>()) {
    return 1;
  } else if (const SeqStmtNode* seq = body.as<SeqStmtNode>()) {
    int count = 0;
    // we have to check every stmt is for loop, cuz we can't handle other cases for now
    for (auto stmt : seq->seq) {
      if (stmt->IsInstance<ForNode>()) {
        count += 1;
      } else {
        return -1;
      }
    }
    return count;
  }
  return -1;
}

/* if the prim_func contains more than one for loop, split it into multiple prim_func,
 * each contain one for-loop, and update the corresponding call_tirs. */
bool SplitForLoops(PrimFunc func) {
  // Step 1. Find out number of outer for-loops in the func.
  Stmt body = func->body.as<BlockRealizeNode>()->block->body;
  int count = CountForLoops(body);
  if (count <= 1) return false;
  // Step 2. Split the for loops.
  FunctionPartitioner partitioner(1);
  partitioner(body);
  return true;
}

}  // namespace tir

namespace relax {
/* split prim_func such that each prim_func contains only one for loop. */
class SplitMutator : public ExprMutator {
 public:
  SplitMutator(const tvm::IRModule& mod) : ExprMutator(mod), mod_(mod) {}
  static IRModule Transform(const IRModule& mod) {
    SplitMutator mutator(mod);
    // collect string to function mappings
    mutator.ConstructFunctionMap();
    for (auto& kv : mutator.function_map_) {
      if (auto* func = kv.second.as<FunctionNode>()) {
        mutator(GetRef<Function>(func));
      }
    }
    return mutator.builder_->GetContextIRModule();
  }

 private:
  using ExprMutator::VisitExpr_;

  void ConstructFunctionMap() {
    for (auto& kv : mod_->functions) {
      function_map_[kv.first->name_hint] = kv.second;
    }
  }

  Expr VisitExpr_(const CallNode* op) final {
    Call call = Downcast<Call>(ExprMutator::VisitExpr_(op));
    static const Op& call_tir_op_ = Op::Get("relax.call_tir");
    // we only handle call_tir
    if (!call->op.same_as(call_tir_op_)) return call;
    // retrieve the function from the module
    const auto* gv_ptr = call->args[0].as<GlobalVarNode>();
    if (gv_ptr == nullptr) return call;
    GlobalVar gv = GetRef<GlobalVar>(gv_ptr);
    auto kv = function_map_.find(gv->name_hint);
    if (kv == function_map_.end()) return call;
    tir::PrimFunc func = Downcast<tir::PrimFunc>(kv->second);
    bool splitted = tir::SplitForLoops(func);
    if (!splitted) return GetRef<Call>(op);
    return call;
  }

  tvm::IRModule mod_;
  std::unordered_map<std::string, BaseFunc> function_map_;
};

namespace transform {

Pass SplitPrimFunc() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
      [=](IRModule m, PassContext pc) { return SplitMutator::Transform(m); };
  return CreateModulePass(/*pass_function=*/pass_func,    //
                          /*opt_level=*/0,                //
                          /*pass_name=*/"SplitPrimFunc",  //
                          /*required=*/{});
}
TVM_REGISTER_GLOBAL("relax.transform.SplitPrimFunc").set_body_typed(SplitPrimFunc);

}  // namespace transform
}  // namespace relax
}  // namespace tvm

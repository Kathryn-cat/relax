/*
 * Licensed to the Apache Software Foundation(ASF) under one
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
 * \file src/relax/transform/pattern_matching.h
 * \brief Pattern matching context for the dispatch algorithm.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ir/module.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {

// TODO: write the categorization code here
// can make a new class, subclass of StmtExprVisitor/Mutator
// for now, we only focus on the categorization part
class PatternMatcher : public StmtExprVisitor {
 public:
  void Categorize(Stmt body) { this->VisitStmt(body); }

  // some public vars here, like loop categorization
  // temporarily this is a set, but will change into hash values of 0, 1 later
  // TODO: add categorized vars of matmul blocks (might be many) here

 private:
  // helper function goes here

  void VisitStmt_(const BlockNode* op) final {
    // perhaps can categorize loop bindings based on the read / write pattern

    if (op->reads.size() != 2 || op->writes.size() != 1) {
      // only categorize and transform matmul patterns
      // this check does not exclude bias
      return;
    }
    // this mapping should be per block
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> all_index_vars;
    const BufferRegion region_A = op->reads[0];
    const BufferRegion region_B = op->reads[1];
    const BufferRegion region_C = op->writes[0];

    // step 0. summary of all indices vars that appeared
    auto f_get_index_vars =
        [](const BufferRegion region,
           std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>* all_index_vars) {
          for (const Range range : region->region) {
            // the indices of region, might contain 2,3, or many
            const VarNode* index_ptr = range->min.as<VarNode>();
            ICHECK(index_ptr != nullptr);
            const Var index_var = GetRef<Var>(index_ptr);
            all_index_vars->insert(index_var);
            std::cout << "inserted var to set" << std::endl;
          }
        };
    f_get_index_vars(region_A, &all_index_vars);
    f_get_index_vars(region_B, &all_index_vars);
    f_get_index_vars(region_C, &all_index_vars);
  }

  // some private vars here
};

}  // namespace tir

namespace relax {

// TODO: write the function to be called in pass
// function MergeLoops / ReorderLoops should be in class subscribing to ExprMutator
// for now, to test categorization, use a dummy function

void PreProcessModule(const tvm::IRModule& mod) {
  tir::PatternMatcher matcher;
  for (auto& kv : mod->functions) {
    if (auto* func = kv.second.as<tir::PrimFuncNode>()) {
      tir::Stmt body = func->body.as<tir::BlockRealizeNode>()->block->body;
      matcher.Categorize(body);
    }
  }
}

}  // namespace relax
}  // namespace tvm

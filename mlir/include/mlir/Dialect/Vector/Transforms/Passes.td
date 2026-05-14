//===-- Passes.td - Vector pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_TRANSFORMS_PASSES
#define MLIR_DIALECT_VECTOR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def LowerVectorMaskPass : Pass<"lower-vector-mask", "func::FuncOp"> {
  let summary = "Lower 'vector.mask' operations";
  let constructor = "mlir::vector::createLowerVectorMaskPass()";
}

def LowerVectorMultiReduction : Pass<"lower-vector-multi-reduction", "func::FuncOp"> {
  let summary = "Lower 'vector.multi_reduction' operations";
  let constructor = "mlir::vector::createLowerVectorMultiReductionPass()";
  let options = [
    Option<"loweringStrategy", "lowering-strategy", "mlir::vector::VectorMultiReductionLowering",
           /*default=*/"mlir::vector::VectorMultiReductionLowering::InnerParallel",
           "Select the strategy to control how multi_reduction is lowered.",
           [{::llvm::cl::values(
            clEnumValN(mlir::vector::VectorMultiReductionLowering::InnerParallel,
                       "inner-parallel",
                       "Lower multi_reduction into outer-reduction and inner-parallel ops."),
            clEnumValN(mlir::vector::VectorMultiReductionLowering::InnerReduction,
                       "inner-reduction",
                       "Lower multi_reduction into outer-parallel and inner-reduction ops.")
        )}]>
  ];
}

def LowerVectorToFromElementsToShuffleTree
  : Pass<"lower-vector-to-from-elements-to-shuffle-tree", "func::FuncOp"> {
  let summary = "Lower `vector.to_elements` and `vector.from_elements` to a tree of `vector.shuffle` operations";
}

def VectorMultiReductionToTreeReduction
  : Pass<"vector-multi-reduction-to-tree-reduction"> {
  let summary = "Convert chained vector.multi_reduction ops to tree reduction";
  let description = [{
    This experimental pass rewrites chains of vector.multi_reduction operations
    into a tree-reduction pattern. It matches chains where extract_strided_slice
    operations feed into sequential multi_reduction ops that accumulate results.
    
    For a chain of N reductions (where N is even and >= 2), it replaces them with
    (N-1) element-wise arithmetic operations (arith.maximumf or arith.addf) arranged
    in a binary tree, followed by a single final vector.multi_reduction.
    
    Example: 4 chained reductions become 3 arith ops + 1 final reduction.
    Example: 8 chained reductions become 7 arith ops + 1 final reduction.
    
    Currently supports only <maximumf> and <add> reduction kinds.
  }];
  let constructor = "mlir::vector::createVectorMultiReductionToTreeReductionPass()";
}

#endif // MLIR_DIALECT_VECTOR_TRANSFORMS_PASSES

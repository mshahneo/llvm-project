//===- VectorMultiReductionToTreeReduction.cpp - Tree reduction opt -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an experimental optimization pass that rewrites chains
// of vector.multi_reduction operations into tree-reduction patterns using
// element-wise arithmetic operations followed by a single vector.multi_reduction.
//
// Pattern matched (for chain length = 4):
//   %0 = vector.extract_strided_slice ... -> vector<1x16xf32>
//   %1 = vector.multi_reduction <op>, %0, %acc [dim] -> vector<1xf32>
//   %2 = vector.extract_strided_slice ... -> vector<1x16xf32>
//   %3 = vector.multi_reduction <op>, %2, %1 [dim] -> vector<1xf32>
//   %4 = vector.extract_strided_slice ... -> vector<1x16xf32>
//   %5 = vector.multi_reduction <op>, %4, %3 [dim] -> vector<1xf32>
//   %6 = vector.extract_strided_slice ... -> vector<1x16xf32>
//   %7 = vector.multi_reduction <op>, %6, %5 [dim] -> vector<1xf32>
//
// Transformed to:
//   %0 = vector.extract_strided_slice ...
//   %2 = vector.extract_strided_slice ...
//   %4 = vector.extract_strided_slice ...
//   %6 = vector.extract_strided_slice ...
//   %a0 = arith.<op> %0, %2 : vector<1x16xf32>
//   %a1 = arith.<op> %4, %6 : vector<1x16xf32>
//   %a2 = arith.<op> %a0, %a1 : vector<1x16xf32>
//   %7 = vector.multi_reduction <op>, %a2, %acc [dim] -> vector<1xf32>
//
// Supports: <maximumf> and <add> reductions only.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cstdlib>
#include <cerrno>

namespace mlir {
namespace vector {
#define GEN_PASS_DEF_VECTORMULTIREDUCTIONTOTREEREDUCTION
#include "mlir/Dialect/Vector/Transforms/Passes.h.inc"
} // namespace vector
} // namespace mlir

using namespace mlir;
using namespace mlir::vector;

namespace {

static int getEnvInt(const char* name, int defaultVal) {
    const char* val = std::getenv(name);
    if (!val) return defaultVal;

    char* end = nullptr;
    errno = 0;

    long result = std::strtol(val, &end, 10);

    // check errors:
    // 1. no digits parsed
    // 2. extra garbage after number
    // 3. overflow / underflow
    if (end == val || *end != '\0' || errno == ERANGE) {
        return defaultVal;
    }

    return static_cast<int>(result);
}

// Helper function to build a binary tree reduction
static Value buildBinaryTreeReduction(PatternRewriter &rewriter, Location loc,
                                       SmallVectorImpl<Value> &values,
                                       VectorType vectorType,
                                       CombiningKind kind) {
  // Base case: if only one value left, return it
  if (values.size() == 1)
    return values[0];
  
  SmallVector<Value, 4> nextLevel;
  
  // Combine pairs of values
  for (size_t i = 0; i + 1 < values.size(); i += 2) {
    Value combined;
    if (kind == CombiningKind::MAXIMUMF) {
      combined = rewriter.create<arith::MaximumFOp>(loc, vectorType,
                                                     values[i], values[i + 1]);
    } else { // CombiningKind::ADD
      combined = rewriter.create<arith::AddFOp>(loc, vectorType,
                                                 values[i], values[i + 1]);
    }
    nextLevel.push_back(combined);
  }
  
  // Recursively build the tree
  return buildBinaryTreeReduction(rewriter, loc, nextLevel, vectorType, kind);
}

// Pattern to convert chains of vector.multi_reduction into tree reductions.
class ChainedMultiReductionToTreePattern
    : public OpRewritePattern<MultiDimReductionOp> {
public:
  using OpRewritePattern<MultiDimReductionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MultiDimReductionOp finalReductionOp,
                                PatternRewriter &rewriter) const override {
    if (!getEnvInt("GC_TREE_REDUCTION", 0)) {
      return failure();
    }
    llvm::dbgs() << "Trying to match multi_reduction: " << finalReductionOp << "\n";
    // Match only maximumf and add reductions
    auto kind = finalReductionOp.getKind();
    if (kind != CombiningKind::MAXIMUMF && kind != CombiningKind::ADD) {
      llvm::dbgs() << "Skipping reduction with unsupported kind: " << kind << "\n";
      return failure();
    }

    // Walk back through the chain to find all chained reductions
    SmallVector<MultiDimReductionOp, 8> reductionChain;
    SmallVector<Value, 8> extractedVectors;
    Value currentAcc = finalReductionOp.getAcc();
    MultiDimReductionOp currentReduction = finalReductionOp;
    
    reductionChain.push_back(finalReductionOp);
    
    // Get the extracted vector for the final reduction
    auto extractOp = finalReductionOp.getSource()
                         .getDefiningOp<ExtractStridedSliceOp>();
    if (!extractOp) {
      llvm::dbgs() << "Skipping reduction with unsupported source: "
                   << finalReductionOp.getSource() << "\n";
      return failure();
    }
    extractedVectors.push_back(finalReductionOp.getSource());
    
    // Store reduction dimensions and type info from the first op
    ArrayRef<int64_t> reductionDims = finalReductionOp.getReductionDims();
    VectorType vectorType = cast<VectorType>(extractOp.getType());
    
    // Walk back through the chain to collect all chained reductions    
    while (true) {
      auto prevReduction = dyn_cast_or_null<MultiDimReductionOp>(
          currentAcc.getDefiningOp());
      if (!prevReduction)
        break;
        
      // Verify same reduction kind and dimensions
      if (prevReduction.getKind() != kind)
        break;
      if (prevReduction.getReductionDims() != reductionDims)
        break;
      
      // Check that the source comes from extract_strided_slice
      auto prevExtract = prevReduction.getSource()
                             .getDefiningOp<ExtractStridedSliceOp>();
      if (!prevExtract)
        break;
      
      // Verify same extracted vector type
      if (prevExtract.getType() != vectorType)
        break;
      
      reductionChain.push_back(prevReduction);
      extractedVectors.push_back(prevReduction.getSource());
      currentAcc = prevReduction.getAcc();
      currentReduction = prevReduction;
    }
    
    // We need at least 2 reductions and an even number for tree reduction
    size_t chainLength = reductionChain.size();
    if (chainLength < 2) {
      llvm::dbgs() << "Skipping reduction chain of length " << chainLength
                   << " (need at least 2)\n";
      return failure();
    }
    
    if (chainLength % 2 != 0) {
      llvm::dbgs() << "Skipping reduction chain of length " << chainLength
                   << " (need even number)\n";
      return failure();
    }
    
    llvm::dbgs() << "Matched chain of " << chainLength << " reductions\n";
    
    // Reverse to get them in execution order (oldest first)
    std::reverse(reductionChain.begin(), reductionChain.end());
    std::reverse(extractedVectors.begin(), extractedVectors.end());
    
    // Get the initial accumulator (from the first reduction in the chain)
    Value initialAcc = reductionChain[0].getAcc();
    
    Location loc = finalReductionOp.getLoc();
    
    // Build a binary tree reduction over all extracted vectors
    Value combined = buildBinaryTreeReduction(rewriter, loc, extractedVectors,
                                               vectorType, kind);
    
    // Create the final multi_reduction with the combined vector
    auto newReduction = rewriter.create<MultiDimReductionOp>(
        loc, finalReductionOp.getType(), kind, combined, initialAcc,
        reductionDims);
    
    llvm::dbgs() << "Successfully transformed chain into tree reduction\n";
    
    // Replace the final reduction with our new one
    rewriter.replaceOp(finalReductionOp, newReduction.getResult());
    
    // Erase the intermediate reductions (but not the extracts, they might be used)
    for (size_t i = 0; i < chainLength - 1; ++i) {
      if (reductionChain[i].use_empty())
        rewriter.eraseOp(reductionChain[i]);
    }
    
    return success();
  }
};

struct VectorMultiReductionToTreeReductionPass
    : public vector::impl::VectorMultiReductionToTreeReductionBase<
          VectorMultiReductionToTreeReductionPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ChainedMultiReductionToTreePattern>(&getContext());
    
    if (failed(applyPatternsGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::vector::createVectorMultiReductionToTreeReductionPass() {
  return std::make_unique<VectorMultiReductionToTreeReductionPass>();
}

//===- OptimizeShuffleReductions.cpp - Optimize shuffle reductions -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements optimization of shuffle-based reductions into hardware
// group reduction intrinsics like __spirv_GroupNonUniformFMax/FAdd.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <cstdlib>
#include <cerrno>

#define DEBUG_TYPE "optimize-shuffle-reductions"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_GPUOPTIMIZESHUFFLEREDUCTIONSPASS
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace mlir

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

// Helper to lookup or create SPIRV function declarations
static LLVM::LLVMFuncOp lookupOrCreateSPIRVGroupFn(Operation *symbolTable,
                                                    StringRef name, Type type,
                                                    LLVM::CConv cconv) {
  auto func = dyn_cast_or_null<LLVM::LLVMFuncOp>(
      SymbolTable::lookupSymbolIn(symbolTable, name));
  if (!func) {
    OpBuilder b(symbolTable->getRegion(0));
    auto i32Type = b.getI32Type();
    auto funcType = LLVM::LLVMFunctionType::get(type, {i32Type, i32Type, type});
    func = LLVM::LLVMFuncOp::create(b, symbolTable->getLoc(), name, funcType);
    func.setCConv(cconv);
    func.setConvergent(true);
    func.setNoUnwind(true);
    func.setWillReturn(true);
  }
  return func;
}

// Pattern to match and replace shuffle-based FMax reductions
// Matches: shuffle_xor + maximum pattern sequence -> GroupNonUniformFMax
struct OptimizeShuffleFMaxReduction : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    // Match call to sub_group_shuffle_xor
    auto callee = callOp.getCallee();
    if (!callee || *callee != "_Z21sub_group_shuffle_xorfj") {
      llvm::dbgs() << "[FMax] Not a shuffle_xor call, callee: "
                   << (callee ? *callee : "null") << "\n";
      return failure();
    }

    // Check if this is part of a reduction pattern
    // Pattern: %val -> shuffle_xor -> maximum(%val, shuffled) -> ...
    auto shuffleResult = callOp.getResult();
    if (!shuffleResult.hasOneUse()) {
      llvm::dbgs() << "[FMax] Shuffle result has multiple uses: "
                   << shuffleResult.getUses().begin()->getOperandNumber() << "\n";
      shuffleResult.dump();
      return failure();
    }

    auto *maxOp = *shuffleResult.getUsers().begin();
    auto maximumOp = dyn_cast<arith::MaximumFOp>(maxOp);
    if (!maximumOp) {
      llvm::dbgs() << "[FMax] User is not arith.maximumf:\n";
      maxOp->dump();
      return failure();
    }

    // Check if one operand is the original value and other is shuffle result
    Value origValue = callOp.getOperand(0);
    if ((maximumOp.getOperand(0) != origValue ||
         maximumOp.getOperand(1) != shuffleResult) &&
        (maximumOp.getOperand(1) != origValue ||
         maximumOp.getOperand(0) != shuffleResult)) {
      llvm::dbgs() << "[FMax] Operands don't match pattern\n";
      llvm::dbgs() << "  origValue: "; origValue.dump();
      llvm::dbgs() << "  shuffleResult: "; shuffleResult.dump();
      llvm::dbgs() << "  max op[0]: "; maximumOp.getOperand(0).dump();
      llvm::dbgs() << "  max op[1]: "; maximumOp.getOperand(1).dump();
      return failure();
    }

    // Get the shuffle offset to determine if this is part of full reduction
    Value offset = callOp.getOperand(1);
    auto constOp = offset.getDefiningOp<arith::ConstantOp>();
    if (!constOp) {
      llvm::dbgs() << "[FMax] Offset is not arith.constant:\n";
      offset.dump();
      return failure();
    }

    auto offsetAttr = cast<IntegerAttr>(constOp.getValue());
    int64_t offsetValue = offsetAttr.getInt();

    // Check if this is the start of a reduction sequence (offset = 1)
    if (offsetValue != 1) {
      llvm::dbgs() << "[FMax] Not starting offset (expected 1, got "
                   << offsetValue << ")\n";
      return failure();
    }

    // Look for the complete reduction chain (1, 2, 4, 8)
    Value currentVal = maximumOp.getResult();
    SmallVector<Operation *> reductionOps;
    reductionOps.push_back(callOp.getOperation());
    reductionOps.push_back(maximumOp.getOperation());

    llvm::dbgs() << "[FMax] Found offset=1 reduction start, checking chain...\n";

    for (int64_t expectedOffset : {2, 4, 8}) {
      llvm::dbgs() << "[FMax] Looking for offset=" << expectedOffset << "\n";
      
      // currentVal should be used exactly twice: once in shuffle, once in max
      auto uses = currentVal.getUses();
      if (std::distance(uses.begin(), uses.end()) != 2) {
        llvm::dbgs() << "[FMax] currentVal doesn't have exactly 2 uses at offset="
                     << expectedOffset << " (has " << std::distance(uses.begin(), uses.end()) << ")\n";
        currentVal.dump();
        return failure();
      }

      // Find the shuffle and max operations among the two users
      LLVM::CallOp nextShuffle;
      arith::MaximumFOp nextMax;
      for (auto &use : currentVal.getUses()) {
        Operation *user = use.getOwner();
        if (auto call = dyn_cast<LLVM::CallOp>(user)) {
          if (!nextShuffle)
            nextShuffle = call;
        } else if (auto max = dyn_cast<arith::MaximumFOp>(user)) {
          if (!nextMax)
            nextMax = max;
        }
      }

      if (!nextShuffle) {
        llvm::dbgs() << "[FMax] Next user is not CallOp at offset="
                     << expectedOffset << ":\n";
        llvm::dbgs() << "Users:\n";
        for (auto &use : currentVal.getUses()) {
          use.getOwner()->dump();
        }
        return failure();
      }

      if (!nextMax) {
        llvm::dbgs() << "[FMax] Couldn't find maximumf among users at offset="
                     << expectedOffset << "\n";
        return failure();
      }

      auto nextCallee = nextShuffle.getCallee();
      if (!nextCallee || *nextCallee != "_Z21sub_group_shuffle_xorfj") {
        llvm::dbgs() << "[FMax] Next call is not shuffle_xor at offset="
                     << expectedOffset << ", callee: "
                     << (nextCallee ? *nextCallee : "null") << "\n";
        return failure();
      }

      if (nextShuffle.getOperand(0) != currentVal) {
        llvm::dbgs() << "[FMax] Shuffle operand doesn't match at offset="
                     << expectedOffset << "\n";
        llvm::dbgs() << "  expected: "; currentVal.dump();
        llvm::dbgs() << "  got: "; nextShuffle.getOperand(0).dump();
        return failure();
      }

      auto nextOffsetOp = nextShuffle.getOperand(1).getDefiningOp<arith::ConstantOp>();
      if (!nextOffsetOp) {
        llvm::dbgs() << "[FMax] Offset is not arith.constant at offset="
                     << expectedOffset << ":\n";
        nextShuffle.getOperand(1).dump();
        return failure();
      }

      auto nextOffsetAttr = cast<IntegerAttr>(nextOffsetOp.getValue());
      if (nextOffsetAttr.getInt() != expectedOffset) {
        llvm::dbgs() << "[FMax] Offset mismatch: expected "
                     << expectedOffset << ", got "
                     << nextOffsetAttr.getInt() << "\n";
        return failure();
      }

      // Check that nextMax uses both currentVal and shuffle result
      Value shuffleResult = nextShuffle.getResult();
      if ((nextMax.getOperand(0) != currentVal ||
           nextMax.getOperand(1) != shuffleResult) &&
          (nextMax.getOperand(1) != currentVal ||
           nextMax.getOperand(0) != shuffleResult)) {
        llvm::dbgs() << "[FMax] MaximumF operands don't match pattern at offset="
                     << expectedOffset << "\n";
        llvm::dbgs() << "  currentVal: "; currentVal.dump();
        llvm::dbgs() << "  shuffleResult: "; shuffleResult.dump();
        llvm::dbgs() << "  max op[0]: "; nextMax.getOperand(0).dump();
        llvm::dbgs() << "  max op[1]: "; nextMax.getOperand(1).dump();
        return failure();
      }

      reductionOps.push_back(nextShuffle.getOperation());
      reductionOps.push_back(nextMax.getOperation());
      currentVal = nextMax.getResult();
      llvm::dbgs() << "[FMax] Found valid reduction at offset=" << expectedOffset << "\n";
    }

    // We have a complete reduction chain! Replace with GroupNonUniformFMax
    llvm::dbgs() << "[FMax] SUCCESS! Complete reduction chain found, replacing with SPIRV intrinsic\n";
    Operation *moduleOp = callOp->getParentWithTrait<OpTrait::SymbolTable>();
    assert(moduleOp && "Expected module");

    Type f32Type = rewriter.getF32Type();
    LLVM::LLVMFuncOp groupMaxFunc = lookupOrCreateSPIRVGroupFn(
        moduleOp, "_Z27__spirv_GroupNonUniformFMaxiif", f32Type,
        LLVM::CConv::SPIR_FUNC);

    // Create the group reduction call
    // Args: scope=3 (Subgroup), operation=0 (Reduce), value
    Value scope = LLVM::ConstantOp::create(
        rewriter, callOp.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(3));
    Value operation = LLVM::ConstantOp::create(
        rewriter, callOp.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));

    auto groupCall = LLVM::CallOp::create(
        rewriter, callOp.getLoc(), groupMaxFunc, ValueRange{scope, operation, origValue});
    groupCall.setCConv(LLVM::CConv::SPIR_FUNC);
    groupCall.setConvergent(true);
    groupCall.setNoUnwindAttr(rewriter.getUnitAttr());
    groupCall.setWillReturnAttr(rewriter.getUnitAttr());

    // Replace the final result
    currentVal.replaceAllUsesWith(groupCall.getResult());

    // Erase ops in reverse order
    for (auto it = reductionOps.rbegin(); it != reductionOps.rend(); ++it) {
      rewriter.eraseOp(*it);
    }

    return success();
  }
};

// Pattern to match and replace shuffle-based FAdd reductions
// Similar to FMax but uses fadd instead of maximum
struct OptimizeShuffleFAddReduction : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    // Match call to sub_group_shuffle_xor
    auto callee = callOp.getCallee();
    if (!callee || *callee != "_Z21sub_group_shuffle_xorfj") {
      llvm::dbgs() << "[FAdd] Not a shuffle_xor call, callee: "
                   << (callee ? *callee : "null") << "\n";
      return failure();
    }

    // Check if this is part of a reduction pattern
    auto shuffleResult = callOp.getResult();
    if (!shuffleResult.hasOneUse()) {
      llvm::dbgs() << "[FAdd] Shuffle result has multiple uses\n";
      shuffleResult.dump();
      return failure();
    }

    auto *addOp = *shuffleResult.getUsers().begin();
    auto faddOp = dyn_cast<arith::AddFOp>(addOp);
    if (!faddOp) {
      llvm::dbgs() << "[FAdd] User is not arith.addf:\n";
      addOp->dump();
      return failure();
    }

    // Check if one operand is the original value and other is shuffle result
    Value origValue = callOp.getOperand(0);
    if ((faddOp.getOperand(0) != origValue ||
         faddOp.getOperand(1) != shuffleResult) &&
        (faddOp.getOperand(1) != origValue ||
         faddOp.getOperand(0) != shuffleResult)) {
      llvm::dbgs() << "[FAdd] Operands don't match pattern\n";
      llvm::dbgs() << "  origValue: "; origValue.dump();
      llvm::dbgs() << "  shuffleResult: "; shuffleResult.dump();
      llvm::dbgs() << "  add op[0]: "; faddOp.getOperand(0).dump();
      llvm::dbgs() << "  add op[1]: "; faddOp.getOperand(1).dump();
      return failure();
    }

    // Get the shuffle offset
    Value offset = callOp.getOperand(1);
    auto constOp = offset.getDefiningOp<arith::ConstantOp>();
    if (!constOp) {
      llvm::dbgs() << "[FAdd] Offset is not arith.constant:\n";
      offset.dump();
      return failure();
    }

    auto offsetAttr = cast<IntegerAttr>(constOp.getValue());
    int64_t offsetValue = offsetAttr.getInt();

    // Check if this is the start of a reduction sequence (offset = 1)
    if (offsetValue != 1) {
      llvm::dbgs() << "[FAdd] Not starting offset (expected 1, got "
                   << offsetValue << ")\n";
      return failure();
    }

    // Look for the complete reduction chain
    Value currentVal = faddOp.getResult();
    SmallVector<Operation *> reductionOps;
    reductionOps.push_back(callOp.getOperation());
    reductionOps.push_back(faddOp.getOperation());

    llvm::dbgs() << "[FAdd] Found offset=1 reduction start, checking chain...\n";

    for (int64_t expectedOffset : {2, 4, 8}) {
      llvm::dbgs() << "[FAdd] Looking for offset=" << expectedOffset << "\n";
      
      // currentVal should be used exactly twice: once in shuffle, once in addf
      auto uses = currentVal.getUses();
      if (std::distance(uses.begin(), uses.end()) != 2) {
        llvm::dbgs() << "[FAdd] currentVal doesn't have exactly 2 uses at offset="
                     << expectedOffset << " (has " << std::distance(uses.begin(), uses.end()) << ")\n";
        currentVal.dump();
        return failure();
      }

      // Find the shuffle and addf operations among the two users
      LLVM::CallOp nextShuffle;
      arith::AddFOp nextAdd;
      for (auto &use : currentVal.getUses()) {
        Operation *user = use.getOwner();
        if (auto call = dyn_cast<LLVM::CallOp>(user)) {
          if (!nextShuffle)
            nextShuffle = call;
        } else if (auto add = dyn_cast<arith::AddFOp>(user)) {
          if (!nextAdd)
            nextAdd = add;
        }
      }

      if (!nextShuffle) {
        llvm::dbgs() << "[FAdd] Next user is not CallOp at offset="
                     << expectedOffset << ":\n";
        llvm::dbgs() << "Users:\n";
        for (auto &use : currentVal.getUses()) {
          use.getOwner()->dump();
        }
        return failure();
      }

      if (!nextAdd) {
        llvm::dbgs() << "[FAdd] Couldn't find addf among users at offset="
                     << expectedOffset << "\n";
        return failure();
      }

      auto nextCallee = nextShuffle.getCallee();
      if (!nextCallee || *nextCallee != "_Z21sub_group_shuffle_xorfj") {
        llvm::dbgs() << "[FAdd] Next call is not shuffle_xor at offset="
                     << expectedOffset << ", callee: "
                     << (nextCallee ? *nextCallee : "null") << "\n";
        return failure();
      }

      if (nextShuffle.getOperand(0) != currentVal) {
        llvm::dbgs() << "[FAdd] Shuffle operand doesn't match at offset="
                     << expectedOffset << "\n";
        llvm::dbgs() << "  expected: "; currentVal.dump();
        llvm::dbgs() << "  got: "; nextShuffle.getOperand(0).dump();
        return failure();
      }

      auto nextOffsetOp =
          nextShuffle.getOperand(1).getDefiningOp<arith::ConstantOp>();
      if (!nextOffsetOp) {
        llvm::dbgs() << "[FAdd] Offset is not arith.constant at offset="
                     << expectedOffset << ":\n";
        nextShuffle.getOperand(1).dump();
        return failure();
      }

      auto nextOffsetAttr = cast<IntegerAttr>(nextOffsetOp.getValue());
      if (nextOffsetAttr.getInt() != expectedOffset) {
        llvm::dbgs() << "[FAdd] Offset mismatch: expected "
                     << expectedOffset << ", got "
                     << nextOffsetAttr.getInt() << "\n";
        return failure();
      }

      // Check that nextAdd uses both currentVal and shuffle result
      Value shuffleResult = nextShuffle.getResult();
      if ((nextAdd.getOperand(0) != currentVal ||
           nextAdd.getOperand(1) != shuffleResult) &&
          (nextAdd.getOperand(1) != currentVal ||
           nextAdd.getOperand(0) != shuffleResult)) {
        llvm::dbgs() << "[FAdd] AddF operands don't match pattern at offset="
                     << expectedOffset << "\n";
        llvm::dbgs() << "  currentVal: "; currentVal.dump();
        llvm::dbgs() << "  shuffleResult: "; shuffleResult.dump();
        llvm::dbgs() << "  add op[0]: "; nextAdd.getOperand(0).dump();
        llvm::dbgs() << "  add op[1]: "; nextAdd.getOperand(1).dump();
        return failure();
      }

      reductionOps.push_back(nextShuffle.getOperation());
      reductionOps.push_back(nextAdd.getOperation());
      currentVal = nextAdd.getResult();
      llvm::dbgs() << "[FAdd] Found valid reduction at offset=" << expectedOffset << "\n";
    }

    // Replace with GroupNonUniformFAdd
    llvm::dbgs() << "[FAdd] SUCCESS! Complete reduction chain found, replacing with SPIRV intrinsic\n";
    Operation *moduleOp = callOp->getParentWithTrait<OpTrait::SymbolTable>();
    assert(moduleOp && "Expected module");

    Type f32Type = rewriter.getF32Type();
    LLVM::LLVMFuncOp groupAddFunc = lookupOrCreateSPIRVGroupFn(
        moduleOp, "_Z27__spirv_GroupNonUniformFAddiif", f32Type,
        LLVM::CConv::SPIR_FUNC);

    // Create the group reduction call
    Value scope = LLVM::ConstantOp::create(
        rewriter, callOp.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(3));
    Value operation = LLVM::ConstantOp::create(
        rewriter, callOp.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));

    auto groupCall = LLVM::CallOp::create(
        rewriter, callOp.getLoc(), groupAddFunc, ValueRange{scope, operation, origValue});
    groupCall.setCConv(LLVM::CConv::SPIR_FUNC);
    groupCall.setConvergent(true);
    groupCall.setNoUnwindAttr(rewriter.getUnitAttr());
    groupCall.setWillReturnAttr(rewriter.getUnitAttr());

    // Replace the final result
    currentVal.replaceAllUsesWith(groupCall.getResult());

    // Erase ops in reverse order
    for (auto it = reductionOps.rbegin(); it != reductionOps.rend(); ++it) {
      rewriter.eraseOp(*it);
    }

    return success();
  }
};

// Pattern to optimize vector element-wise GroupNonUniform calls into tree reduction
// Matches: multiple extract → GroupNonUniform calls from same vector
// Replaces with: local tree reduction → single GroupNonUniform
struct VectorReductionToGroupUniform : public OpRewritePattern<vector::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    // Check if this extract feeds into a GroupNonUniform call
    if (!extractOp.getResult().hasOneUse())
      return failure();

    auto *user = *extractOp.getResult().getUsers().begin();
    auto groupCall = dyn_cast<LLVM::CallOp>(user);
    if (!groupCall)
      return failure();

    auto callee = groupCall.getCallee();
    if (!callee || (*callee != "_Z27__spirv_GroupNonUniformFMaxiif" &&
                    *callee != "_Z27__spirv_GroupNonUniformFAddiif"))
      return failure();

    llvm::dbgs() << "[VectorReduce] Found extract → GroupNonUniform pattern\n";

    // Get the source vector
    Value sourceVector = extractOp.getSource();
    auto vectorType = dyn_cast<VectorType>(sourceVector.getType());
    if (!vectorType) {
      llvm::dbgs() << "[VectorReduce] Source is not a vector type\n";
      return failure();
    }

    int64_t vectorSize = vectorType.getNumElements();
    llvm::dbgs() << "[VectorReduce] Vector size: " << vectorSize << "\n";

    // Find all similar extract → GroupNonUniform patterns from the same vector
    SmallVector<vector::ExtractOp> extractOps;
    SmallVector<LLVM::CallOp> groupCalls;
    
    // Collect all users of the source vector
    for (auto &use : sourceVector.getUses()) {
      if (auto extract = dyn_cast<vector::ExtractOp>(use.getOwner())) {
        if (!extract.getResult().hasOneUse())
          continue;
        
        auto *extractUser = *extract.getResult().getUsers().begin();
        if (auto call = dyn_cast<LLVM::CallOp>(extractUser)) {
          auto callCallee = call.getCallee();
          if (callCallee && *callCallee == *callee) {
            // Check that it's the same operation (same scope and operation args)
            if (call.getNumOperands() >= 3 &&
                groupCall.getNumOperands() >= 3 &&
                call.getOperand(0) == groupCall.getOperand(0) &&
                call.getOperand(1) == groupCall.getOperand(1)) {
              extractOps.push_back(extract);
              groupCalls.push_back(call);
            }
          }
        }
      }
    }

    llvm::dbgs() << "[VectorReduce] Found " << extractOps.size() 
                 << " extract → GroupNonUniform patterns from same vector\n";

    // Need at least 2 to make tree reduction worthwhile (ideally all elements)
    if (extractOps.size() < 2) {
      llvm::dbgs() << "[VectorReduce] Not enough patterns to optimize\n";
      return failure();
    }

    // Safety check: ensure all operations are in the same block
    Block *commonBlock = extractOps[0]->getBlock();
    for (auto extract : extractOps) {
      if (extract->getBlock() != commonBlock) {
        llvm::dbgs() << "[VectorReduce] Extracts in different blocks, skipping\n";
        return failure();
      }
    }
    for (auto call : groupCalls) {
      if (call->getBlock() != commonBlock) {
        llvm::dbgs() << "[VectorReduce] GroupCalls in different blocks, skipping\n";
        return failure();
      }
    }

    // Check if we're covering most of the vector (at least half)
    if (static_cast<int64_t>(extractOps.size()) < vectorSize / 2) {
      llvm::dbgs() << "[VectorReduce] Only " << extractOps.size() 
                   << " out of " << vectorSize << " elements, skipping\n";
      return failure();
    }

    llvm::dbgs() << "[VectorReduce] Optimizing " << extractOps.size() 
                 << " GroupNonUniform calls into tree reduction\n";

    // Determine operation type (max or add)
    bool isFMax = (*callee == "_Z27__spirv_GroupNonUniformFMaxiif");

    // Find the last extract operation in program order to set insertion point
    vector::ExtractOp lastExtract = extractOps[0];
    for (auto extract : extractOps) {
      if (extract->isBeforeInBlock(lastExtract))
        continue;
      lastExtract = extract;
    }

    llvm::dbgs() << "[VectorReduce] Last extract operation identified\n";
    llvm::dbgs() << "[VectorReduce] Setting insertion point after last extract\n";

    // Set insertion point after the last extract to ensure dominance
    rewriter.setInsertionPointAfter(lastExtract);

    // Create tree reduction
    // Collect elements from extracts, ensuring they're all before insertion point
    SmallVector<Value> elements;
    SmallVector<int64_t> indices;
    
    for (auto extract : extractOps) {
      // Safety check: extract should be before or at the insertion point
      if (!extract->isBeforeInBlock(lastExtract) && extract != lastExtract) {
        llvm::dbgs() << "[VectorReduce] WARNING: Extract after insertion point, skipping\n";
        return failure();
      }
      elements.push_back(extract.getResult());
      // Get the static position (indices)
      auto position = extract.getStaticPosition();
      if (!position.empty()) {
        // For 1D vectors, we just take the first (and only) index
        indices.push_back(position[0]);
      }
    }
    
    // Perform tree reduction on extracted elements
    SmallVector<Value> currentLevel = elements;
    Location loc = extractOp.getLoc();
    
    llvm::dbgs() << "[VectorReduce] Building tree reduction for " 
                 << currentLevel.size() << " elements\n";

    while (currentLevel.size() > 1) {
      SmallVector<Value> nextLevel;
      for (unsigned i = 0; i + 1 < currentLevel.size(); i += 2) {
        Value reduced;
        if (isFMax) {
          reduced = arith::MaximumFOp::create(
            rewriter, loc, currentLevel[i], currentLevel[i + 1]).getResult();
        } else {
          reduced = arith::AddFOp::create(
            rewriter, loc, currentLevel[i], currentLevel[i + 1]).getResult();
        }
        nextLevel.push_back(reduced);
      }
      // Handle odd element
      if (currentLevel.size() % 2 == 1) {
        nextLevel.push_back(currentLevel.back());
      }
      currentLevel = std::move(nextLevel);
    }

    Value localReduced = currentLevel[0];
    llvm::dbgs() << "[VectorReduce] Tree reduction complete, creating single GroupNonUniform\n";

    // Create single GroupNonUniform call with the reduced value
    // Use LLVM::CallOp::create with proper signature
    auto calleeAttr = rewriter.getStringAttr(*groupCall.getCallee());
    auto newGroupCall = LLVM::CallOp::create(
      rewriter, loc, groupCall.getResultTypes(), calleeAttr,
      ValueRange{groupCall.getOperand(0), groupCall.getOperand(1), localReduced});
    newGroupCall.setCConv(LLVM::CConv::SPIR_FUNC);
    newGroupCall.setConvergent(true);
    newGroupCall.setNoUnwindAttr(rewriter.getUnitAttr());
    newGroupCall.setWillReturnAttr(rewriter.getUnitAttr());

    // Replace all GroupNonUniform calls with the single result
    for (auto call : groupCalls) {
      rewriter.replaceOp(call, newGroupCall.getResult());
    }

    // Note: extracts will be cleaned up by DCE if they have no other users

    llvm::dbgs() << "[VectorReduce] SUCCESS! Replaced " << groupCalls.size() 
                 << " GroupNonUniform calls with 1\n";

    return success();
  }
};

struct OptimizeShuffleReductionsPass
    : public impl::GpuOptimizeShuffleReductionsPassBase<
          OptimizeShuffleReductionsPass> {
  using impl::GpuOptimizeShuffleReductionsPassBase<
      OptimizeShuffleReductionsPass>::GpuOptimizeShuffleReductionsPassBase;

  void runOnOperation() override {
    if (!getEnvInt("GC_REDUCE_INTRINSIC", 0)) {
      return;
    }
    llvm::dbgs() << "\n=== OptimizeShuffleReductions pass started ===\n";
    RewritePatternSet patterns(&getContext());
    // Only enable shuffle reduction patterns for now
    // VectorReductionToGroupUniform disabled due to dominance issues
    patterns.add<OptimizeShuffleFMaxReduction, OptimizeShuffleFAddReduction>(
        &getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      llvm::dbgs() << "=== OptimizeShuffleReductions pass FAILED ===\n";
      signalPassFailure();
    } else {
      llvm::dbgs() << "=== OptimizeShuffleReductions pass completed ===\n";
    }
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createOptimizeShuffleReductionsPass() {
  return std::make_unique<OptimizeShuffleReductionsPass>();
}
} // namespace mlir

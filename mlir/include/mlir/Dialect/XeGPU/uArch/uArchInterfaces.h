//===--- uArchInterfaces.h ---*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines the utility interfaces that are implemented by individual
/// instructions.
///
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_XEGPU_UTILS_UARCH_INTERFACES_H
#define MLIR_DIALECT_XEGPU_UTILS_UARCH_INTERFACES_H

#include "mlir/Dialect/XeGPU/uArch/uArchBase.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include <map>
#include <string>
#include <vector>

namespace mlir {
namespace xegpu {
namespace uArch {

// Create a BlockIOOp Interface
struct BlockIO2DOpInterface {
  // Get the supported shapes for the specific data type.
  // Can provide load/store/prefetch ops supported shapes for a specific
  // uarch
  virtual std::vector<std::vector<uint32_t>> getSupportedShapes(
      mlir::Type type, bool isTrnasform = false /*VNNI transform bit*/,
      bool isTranspose = false /*transpose bit*/,
      uint32_t transpose_bitwidth = 32 /*transpose bitwidth */) = 0;

  // Get supported types
  virtual std::vector<mlir::Type> getSupportedTypes(MLIRContext &context) = 0;
  // Checks if a shape is supported
  virtual bool checkSupportedShapesAndTypes(std::vector<uint32_t> shape,
                                            mlir::Type type) = 0;
  // Checks if a type is type is supported
  virtual bool checkSupportedTypes(mlir::Type type) = 0;

  // Validate the BlockIO2D ops restrictions
  // @param blockSize, size of the load/store/prefetch block
  // @param surfaceSize, size of the load/store/prefetch surface
  // @param dataType, data type of the data
  // @param alignment, alignment
  // @param surface_pitch, suface pitch
  // @param array_len, array length
  virtual bool validate(std::vector<uint32_t> blockSize,
                        std::vector<uint32_t> surfaceSize, mlir::Type dataType,
                        uint32_t alignment, uint32_t surface_pitch,
                        uint32_t array_len = 1) = 0;
  virtual ~BlockIO2DOpInterface() = default;
};

enum class MMAOpndEnum { MatrixA, MatrixB, MatrixC, MatrixD };
struct MMAOpInterface {
  // Get supported Matrix type
  // @param dataType, data type of the matrix
  // @param matrixType, Matrix type (Matrix A, B, C, or D)
  virtual std::vector<std::pair<uint32_t, uint32_t>>
  getSupportedShapes(mlir::Type dataType, MMAOpndEnum matrixType) = 0;

  // @TODO: This method takes an context object as a parameter, this is to
  // create the type objects from the same context. Since type objects are
  // uniqued in a specific context, to do things like "aType == bType" (where
  // aType and bType are both same type) kind of checks, the both types should
  // be from the same context.
  //
  // One alternative to this is to create enum to represent each types, but this
  // adds an extra burden to user to convert these enums to specific types. In
  // fact the utility that would convert enumToType() and vice versa would still
  // have to use the context object.
  //
  // Untill we have a better solution, we stick to passing context object to
  // this method.
  virtual std::vector<mlir::Type> getSupportedTypes(MLIRContext &context,
                                                    MMAOpndEnum matrixType) = 0;
  virtual bool
  checkSupportedShapesAndTypes(std::pair<uint32_t, uint32_t> AShape,
                               std::pair<uint32_t, uint32_t> BShape,
                               std::pair<uint32_t, uint32_t> CShape,
                               std::pair<uint32_t, uint32_t> DShape,
                               mlir::Type AType, mlir::Type BType,
                               mlir::Type CType, mlir::Type DType) = 0;
  virtual bool checkSupportedTypes(mlir::Type AType, mlir::Type BType,
                                   mlir::Type CType, mlir::Type DType) = 0;
  virtual bool validate(std::pair<uint32_t, uint32_t> AShape,
                        std::pair<uint32_t, uint32_t> BShape,
                        std::pair<uint32_t, uint32_t> CShape,
                        std::pair<uint32_t, uint32_t> DShape, mlir::Type AType,
                        mlir::Type BType, mlir::Type CType,
                        mlir::Type DType) = 0;
  virtual std::vector<uint32_t> getSupportedM(mlir::Type type) = 0;
  virtual std::vector<uint32_t> getSupportedK(mlir::Type type) = 0;
  virtual std::vector<uint32_t> getSupportedN(mlir::Type type) = 0;

  virtual ~MMAOpInterface() = default;
};

} // namespace uArch
} // namespace xegpu
} // namespace mlir
#endif // MLIR_DIALECT_XEGPU_UTILS_UARCH_INTERFACES_H

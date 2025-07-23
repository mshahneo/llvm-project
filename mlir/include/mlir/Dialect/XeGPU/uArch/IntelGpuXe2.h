//===--- IntelGpuXe2.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Xe2 uArch definition.
///
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_XEGPU_UTILS_INTEL_GPU_XE2_H
#define MLIR_DIALECT_XEGPU_UTILS_INTEL_GPU_XE2_H

#include "mlir/Dialect/XeGPU/uArch/uArchInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include <map>
#include <string>
#include <vector>

namespace mlir {
namespace xegpu {
namespace uArch {
namespace Xe2Plus {
struct XeCoreInfo {
  uint32_t num_threads;
  SharedMemory shared_memory;
  uint32_t num_vector_units;
  uint32_t num_matrix_units;

  // Constructor
  XeCoreInfo(uint32_t num_threads, const SharedMemory &shared_memory,
             uint32_t num_vector_units, uint32_t num_matrix_units)
      : num_threads(num_threads), shared_memory(shared_memory),
        num_vector_units(num_vector_units), num_matrix_units(num_matrix_units) {
  }
};

struct Xe2Plus : public uArch {
  XeCoreInfo xe_core;
  Xe2Plus(
      const std::string &archName, const std::string &archDescription,
      const XeCoreInfo &xeCore,
      const std::vector<uArchHierarchyComponent> &hierarchy = {},
      const std::map<std::string, RegisterFileInfo> &regInfo = {},
      const std::vector<CacheInfo> &cacheInfo = {},
      const std::map<std::string, std::shared_ptr<Instruction>> &instrs = {})
      : uArch(archName, archDescription, hierarchy, regInfo, cacheInfo, instrs),
        xe_core(xeCore) {}
};

// struct to represent DPAS instruction
struct DPASInstruction : public Instruction, public MMAOpInterface {
  DPASInstruction()
      : Instruction("dpas",                   // name
                    "Dot Product Accumulate") // description
  {}

  // Override all virtuals from MatrixOpInterface
  virtual std::vector<std::pair<uint32_t, uint32_t>>
  getSupportedShapes(mlir::Type dataType, MMAOpndEnum matrixType) override;
  virtual std::vector<mlir::Type>
  getSupportedTypes(MLIRContext &context, MMAOpndEnum matrixType) override;
  virtual bool
  checkSupportedShapesAndTypes(std::pair<uint32_t, uint32_t> AShape,
                               std::pair<uint32_t, uint32_t> BShape,
                               std::pair<uint32_t, uint32_t> CShape,
                               std::pair<uint32_t, uint32_t> DShape,
                               mlir::Type AType, mlir::Type BType,
                               mlir::Type CType, mlir::Type DType) override;
  virtual bool checkSupportedTypes(mlir::Type AType, mlir::Type BType,
                                   mlir::Type CType, mlir::Type DType) override;
  virtual bool validate(std::pair<uint32_t, uint32_t> AShape,
                        std::pair<uint32_t, uint32_t> BShape,
                        std::pair<uint32_t, uint32_t> CShape,
                        std::pair<uint32_t, uint32_t> DShape, mlir::Type AType,
                        mlir::Type BType, mlir::Type CType,
                        mlir::Type DType) override;
  virtual std::vector<uint32_t> getSupportedM(mlir::Type type) override;
  virtual std::vector<uint32_t> getSupportedK(mlir::Type type) override;
  virtual std::vector<uint32_t> getSupportedN(mlir::Type type) override;
};

// struct to represent Load2D/Store2D/Prefetch instruction
struct LoadStorePrefetch2DInstruction : public Instruction {
  MemoryType memory_type;
  // MemoryAccessType memory_access_type;
  //   std::vector<std::string> supported_types;
  std::vector<uint32_t> supported_types_bitwidth;
  std::map<std::string, uint32_t> alignment;
  std::vector<std::vector<uint32_t>> supported_tile_sizes;
  uint32_t min_surface_pitch;

  // Validate Array length restriction on a given tile
  bool validateArrayLenRestriction(std::vector<uint32_t> tile,
                                   uint32_t array_len, mlir::Type dataType) {

    Restriction<std::vector<uint32_t>, uint32_t, mlir::Type>
        width_array_len_restriction(
            tile, array_len, dataType,
            [](std::vector<uint32_t> tile, uint32_t array_len,
               mlir::Type dataType) {
              assert(tile.size() == 2);
              return tile[1] * array_len *
                         (dataType.getIntOrFloatBitWidth() / 8) <=
                     64;
            });
    return width_array_len_restriction.validate();
  }

  // Validate Surface Pitch restriction on a given tile
  bool validateSurfacePitchRestriction(std::vector<uint32_t> tile,
                                       uint32_t surfacePitch /*in bytes*/) {
    Restriction<std::vector<uint32_t>, uint32_t> surface_pitch_restriction(
        tile, surfacePitch,
        [](std::vector<uint32_t> tile, uint32_t surfacePitch) {
          assert(tile.size() == 2);
          return surfacePitch >= 64;
        });
    return surface_pitch_restriction.validate();
  }
};

namespace PVCuArch {
struct PVCuArch : public Xe2Plus {
  // Maintaines ownership of the instructions owned by PVUarch
  std::vector<std::shared_ptr<Instruction>> owned_instructions;
  PVCuArch()
      : Xe2Plus("pvc",                        // archName
                "Ponte Vecchio Architecture", // archDescription
                XeCoreInfo(8, SharedMemory(512 * 1024, 4), 8, 8), // xeCore
                {/* register_file_info */}, // Optional: empty
                {/* cache_info */},         // Optional: empty
                {/* instructions */},       // Optional: empty
                {/* restrictions */}        // Optional: empty
        ) {
    // Initialize uArchHierarchy
    this->uArch_hierarchy.push_back(uArchHierarchyComponent("thread", 0));
    this->uArch_hierarchy.push_back(uArchHierarchyComponent("XeCore", 8));
    this->uArch_hierarchy.push_back(uArchHierarchyComponent("XeSlice", 16));
    this->uArch_hierarchy.push_back(uArchHierarchyComponent("XeStack", 4));
    this->uArch_hierarchy.push_back(uArchHierarchyComponent("gpu", 2));
    // Intialize register file info
    // GRF
    this->register_file_info.emplace(
        "GRF",
        RegisterFileInfo(64 * 1024,          // size in bits
                         {"small", "large"}, // GRF modes
                         {128, 256},         // registers per thread per mode
                         0,                  // number of banks
                         0                   // bank size
                         ));
    // Initialize cache info
    // L1 cache, XeCore level
    this->cache_info.push_back(
        CacheInfo(512 * 1024, 64, this->uArch_hierarchy[1]));
    // L3 cache, XeStack level
    this->cache_info.push_back(
        CacheInfo(512 * 1024, 64, this->uArch_hierarchy[3]));

    // Add the instructions
    auto dpas = std::make_shared<DPASInstruction>();
    instructions.emplace(dpas->name, dpas);
    // instructions[dpas->name] = dpas.get();
    owned_instructions.push_back(dpas);
  }
};
} // namespace PVCuArch

namespace BMGuArch {
struct BMGuArch : public Xe2Plus {
  // Maintaines ownership of the instructions owned by PVUarch
  std::vector<std::shared_ptr<Instruction>> owned_instructions;
  BMGuArch()
      : Xe2Plus("bmg",                     // archName
                "Battlemage Architecture", // archDescription
                XeCoreInfo(8, SharedMemory(256 * 1024, 4), 8, 8), // xeCore
                {/* register_file_info */}, // Optional: empty
                {/* cache_info */},         // Optional: empty
                {/* instructions */},       // Optional: empty
                {/* restrictions */}        // Optional: empty
        ) {
    // Initialize uArchHierarchy
    this->uArch_hierarchy.push_back(uArchHierarchyComponent("thread", 0));
    this->uArch_hierarchy.push_back(uArchHierarchyComponent("XeCore", 8));
    this->uArch_hierarchy.push_back(uArchHierarchyComponent("XeSlice", 4));
    this->uArch_hierarchy.push_back(uArchHierarchyComponent("XeStack", 5));
    this->uArch_hierarchy.push_back(uArchHierarchyComponent("gpu", 1));
    // Intialize register file info
    // GRF
    this->register_file_info["GRF"] =
        RegisterFileInfo(64 * 1024,          // size in bits
                         {"small", "large"}, // GRF modes
                         {128, 256},         // registers per thread per mode
                         0,                  // number of banks
                         0                   // bank size
        );
    // Initialize cache info
    // L1 cache, XeCore level
    this->cache_info.push_back(
        CacheInfo(256 * 1024, 64, this->uArch_hierarchy[1]));
    // L3 cache, XeStack level
    this->cache_info.push_back(
        CacheInfo(18 * 1024 * 1024, 256, this->uArch_hierarchy[3]));

    // Add the instructions
    auto dpas = std::make_shared<DPASInstruction>();
    instructions.emplace(dpas->name, dpas);
    // instructions[dpas->name] = dpas.get();
    owned_instructions.push_back(dpas);
  }
};
} // namespace BMGuArch

} // namespace Xe2Plus
} // namespace uArch
} // namespace xegpu
} // namespace mlir

#endif // MLIR_DIALECT_XEGPU_UTILS_INTEL_GPU_XE2_H

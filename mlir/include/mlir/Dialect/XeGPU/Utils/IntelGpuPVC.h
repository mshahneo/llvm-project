//===--- uArch.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// PVC uArch definition.
///
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_XEGPU_UTILS_INTEL_GPU_PVC_H
#define MLIR_DIALECT_XEGPU_UTILS_INTEL_GPU_PVC_H

#include "mlir/Dialect/XeGPU/Utils/uArch.h"
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
  uint num_threads;
  SharedMemory shared_memory;
  uint num_vector_units;
  uint num_matrix_units;

  // Constructor
  XeCoreInfo(uint num_threads, const SharedMemory &shared_memory,
             uint num_vector_units, uint num_matrix_units)
      : num_threads(num_threads), shared_memory(shared_memory),
        num_vector_units(num_vector_units), num_matrix_units(num_matrix_units) {
  }
};

struct Xe2Plus : public uArch {
  XeCoreInfo xe_core;
  Xe2Plus(const std::string &archName, const std::string &archDescription,
          const XeCoreInfo &xeCore,
          const std::vector<uArchHierarchyComponent> &hierarchy = {},
          const std::map<std::string, RegisterFileInfo> &regInfo = {},
          const std::vector<CacheInfo> &cacheInfo = {},
          const std::map<std::string, Instruction *> &instrs = {},
          const std::vector<Restriction<> *> &restrs = {})
      : uArch(archName, archDescription, hierarchy, regInfo, cacheInfo, instrs,
              restrs),
        xe_core(xeCore) {}
};

// struct to represent DPAS instruction
struct DPASInstruction : public Instruction, public MatrixOpInterface {
  // Range systolic_depth;
  // Range repreat_count;
  // Range execution_size;
  // std::map<std::string, uint> ops_per_channel;
  // std::vector<std::vector<std::string>> supported_types;
  // std::map<std::string, std::map<std::string, std::vector<std::string>>>
  //     matrix_size;

  // bool checkSupportedDPASTypes(mlir::Type dstType, mlir::Type src0Type,
  //                              mlir::Type src1Type, mlir::Type src2Type);

  DPASInstruction()
      : Instruction("dpas",                     // name
                    "Dot Product Accumulate",   // description
                    "0xABCD",                   // opcode
                    FunctionalUnit::Matrix,     // functional_unit
                    InstructionType::SIMD,      // type
                    InstructionScope::Subgroup, // scope
                    UnitOfComputation::Matrix)  // unit_of_computation
  {}

  // Override all virtuals from MatrixOpInterface
  virtual bool checkSupportedMMATypes(mlir::Type AType, mlir::Type BType,
                                      mlir::Type CType,
                                      mlir::Type DType) override;
  virtual std::vector<uint> getSupportedM(mlir::Type type) override;
  virtual std::vector<uint> getSupportedK(mlir::Type type) override;
  virtual std::vector<uint> getSupportedN(mlir::Type type) override;
  virtual std::vector<std::pair<unsigned, unsigned>>
  getSupportedMatrix(mlir::Type type, MatrixType matrixType) override;
};

struct LoadStore2DTileInfo : public RangeTile {
  std::vector<uint> array_len;
};

// struct to represent Load2D/Store2D/Prefetch instruction
struct LoadStorePrefetch2DInstruction : public Instruction {
  MemoryType memory_type;
  MemoryAccessType memory_access_type;
  //   std::vector<std::string> supported_types;
  std::vector<uint> supported_types_bitwidth;
  std::map<std::string, uint> alignment;
  LoadStore2DTileInfo supported_tile_sizes;
  uint min_surface_pitch;

  // Validate Array length restriction on a given tile
  bool validateArrayLenRestriction(Tile tile, uint array_len,
                                   mlir::Type dataType) {

    Restriction<Tile, uint, mlir::Type> width_array_len_restriction(
        tile, array_len, dataType,
        [](Tile tile, uint array_len, mlir::Type dataType) {
          assert(tile.no_of_dims == 2);
          return tile.dims[1] * array_len *
                     (dataType.getIntOrFloatBitWidth() / 8) <=
                 64;
        });
    return width_array_len_restriction.validate();
  }

  // Validate Surface Pitch restriction on a given tile
  bool validateSurfacePitchRestriction(Tile tile,
                                       uint surfacePitch /*in bytes*/) {
    Restriction<Tile, uint> surface_pitch_restriction(
        tile, surfacePitch, [](Tile tile, uint surfacePitch) {
          assert(tile.no_of_dims == 2);
          return surfacePitch >= 64;
        });
    return surface_pitch_restriction.validate();
  }
};

namespace PVCuArch {
struct PVCuArch : public Xe2Plus {
  // Maintaines ownership of the instructions owned by PVUarch
  std::vector<std::unique_ptr<Instruction>> owned_instructions;
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
        CacheInfo(512 * 1024, 64, this->uArch_hierarchy[1]));
    // L3 cache, XeStack level
    this->cache_info.push_back(
        CacheInfo(512 * 1024, 64, this->uArch_hierarchy[3]));

    // Add the instructions
    auto dpas = std::make_unique<DPASInstruction>();
    instructions[dpas->name] = dpas.get();
    owned_instructions.push_back(std::move(dpas));
  }
};
} // namespace PVCuArch

} // namespace Xe2Plus
} // namespace uArch
} // namespace xegpu
} // namespace mlir

#endif // MLIR_DIALECT_XEGPU_UTILS_INTEL_GPU_PVC_H

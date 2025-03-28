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
namespace PVCuArch {
struct XeCoreInfo {
  uint num_threads;
  SharedMemory shared_memory;
  uint num_vector_units;
  uint num_matrix_units;
};

struct Xe2Plus : public uArch {
  XeCoreInfo xe_core;
};

// struct to represent DPAS instruction
struct DPASInstruction : public Instruction {
  Range systolic_depth;
  Range repreat_count;
  Range execution_size;
  std::map<std::string, uint> ops_per_channel;
  std::vector<std::vector<std::string>> supported_types;
  std::map<std::string, std::map<std::string, std::vector<std::string>>>
      matrix_size;

  bool checkSupportedDPASTypes(mlir::Type dstType, mlir::Type src0Type,
                               mlir::Type src1Type, mlir::Type src2Type);
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

} // namespace PVCuArch
} // namespace uArch
} // namespace xegpu
} // namespace mlir

#endif // MLIR_DIALECT_XEGPU_UTILS_INTEL_GPU_PVC_H
//===--- IntelGpuPVC.h ---------------------------------------*- C++ -*-===//

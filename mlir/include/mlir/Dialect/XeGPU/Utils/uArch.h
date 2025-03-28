//===--- uArch.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Base uArch definition for different architectures.
///
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_XEGPU_UTILS_UARCH_H
#define MLIR_DIALECT_XEGPU_UTILS_UARCH_H

#include <functional>
#include <iostream>
#include <tuple>
namespace mlir {
namespace xegpu {
namespace uArch {

// Data types we need for YAML to uArch translation
struct Range {
  int start;
  int end;
};

// Tile can be multi-dimensional
// For example, a 2D tile can be represented as:
// Tile:
//   no_of_dims: 2
//   dim: [2, 2]
// This represents a 2x2 tile
struct Tile {
  uint no_of_dims;
  std::vector<uint> dims;
};

// RangeTile represents a range of tiles instead of a single tile
// RangeTile essentially provides a way represent the supported range of values
// in each dimension For each dimension, the range of values is represented as a
// Range For example, a 2D RangeTile can be represented as: RangeTile:
//   no_of_dims: 2
//   dims:
//     - [1, 32]
//     - [2, 16]
// This represents a 2x2 RangeTile where the first dimension can have values
// from 1 to 32 and the second dimension can have values from 2 to 16
struct RangeTile {
  uint no_of_dims;
  std::vector<Range> dims;
};

// DiscreteTile represents a set of tiles instead of a single tile
// DiscreteTile essentially provides a way represent the supported set of values
// in each dimension For each dimension, the set of values is represented as a
// vector of integers For example, a 2D DiscreteTile can be represented as:
// DiscreteTile:
//   no_of_dims: 2
//   dims:
//     - [1, 2, 4, 8, 16, 32]
//     - [2, 4, 8, 16]
// This represents a 2x2 DiscreteTile where the first dimension can have values
// 1, 2, 4, 8, 16, 32 and the second dimension can have values 2, 4, 8, 16
struct DiscreteTile {
  uint no_of_dims;
  std::vector<std::vector<uint>> dims;
};

// Restriction struct
// This struct is used to represent a restriction on the uArch
// The restriction is represented as a range of necessary parameters (template
// arguments) and a lambda function (validate()) that takes the same number of
// arguments as the number of template arguments The lambda function returns
// true if the arguments satisfy the restriction The lambda function returns
// false if the arguments do not satisfy the restriction

// For example, a restriction that checks if the number of dimensions in a
// RangeTile is 2 can be represented as: RangeTile rt = {2, {{1, 32}, {2, 16}}};
// Restriction<RangeTile> r1(rt, [](RangeTile t) { return t.no_of_dims == 2; });
// r1.validate() will return true if the number of dimensions in the RangeTile
// is 2 r1.validate() will return false if the number of dimensions in the
// RangeTile is not 2

// The primary purpose of Restriction struct is to provide a generic way to
// represent restrictions on the uArch and to validate if the uArch satisfies
// the restrictions
template <typename... Args>
struct Restriction {
  std::tuple<Args...> data;
  std::function<void(Args...)> func;

  Restriction(Args... args, std::function<void(Args...)> f)
      : data(args...), func(f) {}

  bool validate() { return std::apply(func, data); }
  std::any apply() { return std::apply(func, data); }
};

// An enum class to represent the functional unit of an instruction
enum class FunctionalUnit {
  ALU,
  Tensor,
  Matrix,
  Load,
  Store,
  Branch,
  Barrier,
  Memory,
  Atomic,
  Interconnect,
  Other
};

// An enum class to represent the type of memory
enum class MemoryType { Shared, Local, Global, Constant, Texture, Other };

// An enum class to represent the memory access type
enum class MemoryAccessType { Read, Write, ReadWrite, Other };

// An enum class to represent the type of an instruction
enum class InstructionType { SIMT, SIMD, SPMD, MIMD, Other };

// An enum class to represent the scope of an instruction
enum class InstructionScope {
  WorkItem,
  Subgroup,
  Workgroup,
  Cluster,
  Thread, // For CPU
  Core,   // For CPU
  Other
};

// An enum class to represent the unit of computation of an instruction
enum class UnitOfComputation {
  Scalar,
  Vector, // 1-D vector
  Matrix,
  Tile,
  Other
};

// A struct to represent basic information about an instruction
// This struct is used to represent the information about an instruction in the
// uArch The information includes:
// - the name of the instruction,
// - the opcode,
// - the functional unit,
// - the type of the instruction,
// - the scope of the instruction,
// - the unit of computation,
// - the description of the instruction
// The information is represented as strings
// For example, the information about an instruction can be represented as:
// Instruction info = {"dpas", "0x83", "matrix", "simd", "subgroup", "tile",
// "Dot Product Accumulate Systolic  (DPAS) is a matrix multiply-add
// operation"};

// The primary purpose of Instruction struct is to provide a generic way to
// represent information about an instruction and to use this information to
// generate the uArch. Specifc instruction in a uArch can inherit from this
// struct and add more fields as needed

struct Instruction {
  std::string name;
  std::string description;
  std::string opcode;
  FunctionalUnit functional_unit;
  InstructionType type;
  InstructionScope scope;
  UnitOfComputation unit_of_computation;

  // @TODO: Add more fields as needed
  // std::string latency;
  // std::string throughput;
  // std::string pipeline;
  // std::string resource;
  // std::string comment;
};

// A struct to represent register file information
struct RegisterFileInfo {
  uint size;                     // size per register in bits
  std::vector<std::string> mode; // e.g., "small", "large" GRF modes
  std::vector<uint>
      num_regs_per_thread_per_mode; // number of registers per thread per mode
  uint num_banks;
  uint bank_size;
};

// A struct to represent cache information
struct CacheInfo {
  uint size;
  uint associativity;
  uint line_size;
  uint num_banks;
  uint bank_size;
  uint num_ports;
  uint port_width;
  uint bank_conflicts;
};

// A struct to represent the uArch
// This struct is used to represent the microarchitecture of a target device
// The uArch includes:
// - the name of the uArch,
// - the description of the uArch,
// - the range of tiles supported by the uArch,
// - the set of tiles supported by the uArch,
// - the set of instructions supported by the uArch,
// - the set of restrictions on the uArch
// The information is represented as strings, RangeTile, DiscreteTile,
// Instruction and Restriction structs For example, the information about a
// uArch can be represented as: uArch uarch = {"XeHPG", "Intel Xe HPG
// microarchitecture", {2, {{1, 32}, {1, 32}}}, {2, {{1, 2, 4, 8, 16, 32}, {1,
// 2, 4, 8, 16, 32}}}, {{"dpas", "0x83", "matrix", "simd", "subgroup", "tile",
// "Dot Product Accumulate Systolic  (DPAS) is a matrix multiply-add
// operation"}}, {r1, r2, r3}}; This represents a uArch named "XeHPG" with
// description "Intel Xe HPG microarchitecture" that supports 2x2 tiles with
// dimensions ranging from 1 to 32, 1 to 32, supports a DPAS instruction and has
// 3 restrictions r1, r2, r3 on the uArch
struct uArch {
  std::string name; // similar to target triple
  std::string description;
  // Different kind of regiger file information (e.g., GRF, ARF, etc.)
  std::vector<RegisterFileInfo> register_file_info;
  // Each level of cache is indexed lower to higher in the vector
  // (e.g., L1 indexed at 0, L2 at 1 and so on) L1, L2, L3, etc.
  std::vector<CacheInfo> cache_info;
  std::vector<Instruction *> instructions;
  std::vector<Restriction<> *> restrictions;
};

// A struct to represent shared memory information
struct SharedMemory {
  uint size;
  uint alignment;
  // @TODO: Add more fields as needed
  // uint latency;
  // uint throughput;
  // uint bandwidth;
  // uint num_ports;
  // uint port_width;
  // uint bank_size;
  // uint bank_conflicts;
  // uint num_banks;
};

// For future use case in Xe4+

// struct EUInfo {
//     uint num_eu_threads;
//     SharedMemory shared_memory;
// };

//     uint num_simd_units;
//     uint num_spus;
//     uint num_smt;
//     uint num_hardware_threads;
//     uint num_threads_per_spu;
//     uint num_threads_per_simd_unit;
//     uint num_threads_per_hardware_thread;
//     uint num_threads_per_smt;
//     SharedMemory shared_memory;
// };

// A struct to represent a GPU uArch
// This struct is used to represent the GPU microarchitecture of a target device
// struct GPUuArch : public uArch {
//     uint num_compute_units;
//     uint num_vector_units;
//     uint num_scalar_units;
//     uint num_tensor_units;
//     uint num_matrix_units;
//     SharedMemory shared_memory;
// };
} // namespace uArch
} // namespace xegpu
} // namespace mlir

#endif // MLIR_DIALECT_XEGPU_UTILS_UARCH_H
//===--- uArch.h ---------------------------------------*- C++ -*-===//

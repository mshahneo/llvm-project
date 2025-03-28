#include "mlir/Dialect/XeGPU/Utils/IntelGpuPVC.h"
#include "llvm/Support/YAMLTraits.h"
#include <iostream>
#include <string>
#include <vector>

using namespace mlir::xegpu::uArch;
using namespace mlir::xegpu::uArch::PVCuArch;

namespace llvm {
namespace yaml {
template <>
struct MappingTraits<XeCoreInfo> {
  static void mapping(IO &io, XeCoreInfo &xe_core) {
    io.mapRequired("num_threads", xe_core.num_threads);
    io.mapRequired("shared_memory", xe_core.shared_memory);
    io.mapRequired("num_vector_units", xe_core.num_vector_units);
    io.mapRequired("num_matrix_units", xe_core.num_matrix_units);
  }
};

template <>
struct MappingTraits<Xe2Plus> {
  static void mapping(IO &io, Xe2Plus &xe2plus) {
    io.mapRequired("xe_core", xe2plus.xe_core);
  }
};
} // namespace yaml
} // namespace llvm

// namespace mlir {
// namespace xe_gpu {
//     namespace namespace mlir {
//     namespace xegpu {
//     namespace PVCuArchYAML { {
//         struct XeCoreInfo {
//             uint num_threads;
//             SharedMemory shared_memory;
//             uint num_vector_units;
//             uint num_matrix_units;
//         };

//         struct Xe2Plus {
//             XeCoreInfo xe_core;
//         };
//     }
// }
// }

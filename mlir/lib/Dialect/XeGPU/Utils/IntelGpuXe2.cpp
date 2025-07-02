#include "mlir/Dialect/XeGPU/Utils/IntelGpuPVC.h"
#include "llvm/Support/YAMLTraits.h"
#include <iostream>
#include <string>
#include <vector>

using namespace mlir::xegpu::uArch;
using namespace mlir::xegpu::uArch::Xe2Plus;

namespace mlir {
namespace xegpu {
namespace uArch {
namespace Xe2Plus {
bool DPASInstruction::checkSupportedMMATypes(mlir::Type AType, mlir::Type BType,
                                             mlir::Type CType,
                                             mlir::Type DType) {
  if (AType.isF16() || BType.isF16()) {
    if (AType != BType || (CType && (!CType.isF32() && !CType.isF16())) ||
        (!DType.isF32() && !DType.isF16())) {
      llvm::errs()
          << "Unsupported dpas combinations of Dst, Acc, A and B matrices, "
          << "Supported types are:\n"
          << "  Dst    |   Acc   |   A   |  B  \n"
          << " f, hf   |  f, hf  |   hf  |  hf \n"
          << "AType: " << AType << " BType: " << BType << " CType: " << CType
          << " DType: " << DType;
      return false;
    }
  } else if (AType.isBF16() || BType.isBF16()) {
    if (AType != BType || (CType && (!CType.isF32() && !CType.isBF16())) ||
        (!DType.isF32() && !DType.isBF16())) {
      llvm::errs()
          << "Unsupported dpas combinations of Dst, Acc, A and B matrices, "
          << "Supported types are:\n"
          << "  Dst    |   Acc   |   A   |  B  \n"
          << " f, bf   |  f, bf  |   bf  |  bf \n"
          << "AType: " << AType << " BType: " << BType << " CType: " << CType
          << " DType: " << DType;
      return false;
    }
  } else if (AType.isTF32() || BType.isTF32()) {
    if (AType != BType || (CType && (!CType.isF32() && !DType.isF32())) ||
        (!DType.isF32())) {
      llvm::errs()
          << "Unsupported dpas combinations of Dst, Acc, A and B matrices, "
          << "Supported types are:\n"
          << "  Dst    |   Acc   |   A    |   B  \n"
          << "   f     |    f    |  tf32  |  tf32 \n"
          << "AType: " << AType << " BType: " << BType << " CType: " << CType
          << " DType: " << DType;
      return false;
    }
  } else if (!(AType.isInteger(2) || AType.isInteger(4) ||
               AType.isInteger(8)) &&
             !(BType.isInteger(2) || BType.isInteger(4) ||
               BType.isInteger(8))) {
    llvm::errs()
        << "Unsupported dpas combinations of Dst, Acc, A and B matrices, "
        << "Supported types are:\n"
        << "  Dst     |   Acc    |         A           |         B          "
           " \n"
        << " ud, d    |  ud,d    |  ub,b,u4,s4,u2,s2   |  ub,b,u4,s4,u2,s2  "
        << "AType: " << AType << " BType: " << BType << " CType: " << CType
        << " DType: " << DType;
    return false;
  }

  return true;
}

std::vector<uint> DPASInstruction::getSupportedM(mlir::Type type) {
  return {1, 2, 3, 4, 5, 6, 7, 8};
}

std::vector<uint> DPASInstruction::getSupportedK(mlir::Type type) {
  // assert if data type is not int or float type
  assert(type.isIntOrFloat() && "Matrix type must be int or float");
  auto bitWidth = type.getIntOrFloatBitWidth();
  uint kSize = -1;
  switch (bitWidth) {
  case 2:
    kSize = 64;
    break;
  case 4:
    kSize = 64;
    break;
  case 8:
    kSize = 32;
    break;
  case 16:
    kSize = 16;
    break;
  case 32:
    kSize = 8;
    break;
  default:
    llvm_unreachable("Invalid int or float");
  }
}

std::vector<uint> DPASInstruction::getSupportedN(mlir::Type type) {
  return {16};
}

std::vector<std::pair<unsigned, unsigned>>
DPASInstruction::getSupportedMatrix(mlir::Type type, MatrixType matrixType) {
  auto combineVectors = [](const std::vector<unsigned> &a,
                           const std::vector<unsigned> &b)
      -> std::vector<std::pair<unsigned, unsigned>> {
    std::vector<std::pair<unsigned, unsigned>> result;
    for (unsigned x : a) {
      for (unsigned y : b) {
        result.emplace_back(x, y);
      }
    }
    return result;
  };

  auto M = getSupportedM(type);
  auto K = getSupportedK(type);
  auto N = getSupportedN(type);
  std::vector<std::pair<unsigned, unsigned>> resultMatrix;

  switch (matrixType) {
  case MatrixType::MatrixA:
    resultMatrix = combineVectors(M, K);
    break;
  case MatrixType::MatrixB:
    resultMatrix = combineVectors(K, N);
    break;
  case MatrixType::MatrixC:
    resultMatrix = combineVectors(M, N);
    break;
  case MatrixType::MatrixD:
    resultMatrix = combineVectors(M, N);
    break;
  default:
    break;
  }
}

} // namespace Xe2Plus
} // namespace uArch
} // namespace xegpu
} // namespace mlir

// namespace llvm {
// namespace yaml {
// template <>
// struct MappingTraits<XeCoreInfo> {
//   static void mapping(IO &io, XeCoreInfo &xe_core) {
//     io.mapRequired("num_threads", xe_core.num_threads);
//     io.mapRequired("shared_memory", xe_core.shared_memory);
//     io.mapRequired("num_vector_units", xe_core.num_vector_units);
//     io.mapRequired("num_matrix_units", xe_core.num_matrix_units);
//   }
// };

// template <>
// struct MappingTraits<Xe2Plus> {
//   static void mapping(IO &io, Xe2Plus &xe2plus) {
//     io.mapRequired("xe_core", xe2plus.xe_core);
//   }
// };
// } // namespace yaml
// } // namespace llvm

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

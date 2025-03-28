add_mlir_dialect_library(MLIRIntelGpuPVC
  IntelGpuPVC.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/XeGPU/Utils

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRXeGPUDialect
)

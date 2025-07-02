add_mlir_dialect_library(MLIRXeGPUUtils
  IntelGpuXe2.cpp
  XeGPUUtils.cpp


  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/XeGPU/Utils

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDialectUtils
  MLIRSCFTransforms
  MLIRXeGPUDialect
)


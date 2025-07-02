//===-- XeGPUAttachTargetDevice.cpp ---- XeGPU Attach Target Device Pass --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Dialect/XeGPU/uArch/IntelGpuXe2.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUATTACHTARGETDEVICE
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

using namespace mlir;

namespace {
struct XeGPUAttachTargetDevicePass final
    : public xegpu::impl::XeGPUAttachTargetDeviceBase<
          XeGPUAttachTargetDevicePass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void XeGPUAttachTargetDevicePass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();
  Builder b(ctx);

  // Build #dlti.dl_entry<"name", "<deviceName>">
  // auto nameEntry = dlti::DLEntryAttr::get(ctx, b.getStringAttr("name"),
  //                                         b.getStringAttr(deviceName));

  auto nameEntry = DataLayoutEntryAttr::get(b.getStringAttr("name"),
                                            b.getStringAttr(deviceName));

  // Build #dlti.target_device_spec<...>
  TargetDeviceSpecInterface deviceSpec =
      TargetDeviceSpecAttr::get(ctx, {nameEntry});

  // Construct a dl_entry for "GPU" = deviceSpec
  auto sysSpecVal =
      DataLayoutEntryAttr::get(b.getStringAttr("GPU"), deviceSpec);

  // Cast to the expected interface
  DataLayoutEntryInterface sysSpecIface =
      llvm::dyn_cast<DataLayoutEntryInterface>(sysSpecVal);

  // Now build target system spec
  auto systemSpec = TargetSystemSpecAttr::get(
      ctx, ArrayRef<DataLayoutEntryInterface>{sysSpecIface});

  // Attach to module
  module->setAttr("dlti.target_system_spec", systemSpec);

  // Create the uArch object for the target device and add it to the uArchMap
  // We don't have to do it here, we can do it in the Dialect initialization
  // phase, this is just showing one way of doing it
  if (deviceName == "pvc") {
    auto pvcuArch =
        std::make_shared<mlir::xegpu::uArch::Xe2Plus::PVCuArch::PVCuArch>();
    mlir::xegpu::uArch::uArchMap::instance().insert(deviceName, pvcuArch);
  } else if (deviceName == "bmg") {
    auto bmguArch =
        std::make_shared<mlir::xegpu::uArch::Xe2Plus::BMGuArch::BMGuArch>();
    mlir::xegpu::uArch::uArchMap::instance().insert(deviceName, bmguArch);
  }
}

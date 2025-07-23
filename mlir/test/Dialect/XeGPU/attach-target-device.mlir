// RUN: mlir-opt --xevm-attach-target='module=xevm.* O=3 chip=pvc' %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: module @valid_dpas
module @valid_dpas attributes {gpu.container_module} {
  // CHECK: gpu.module @valid_dpas [#xevm.target<O = 3, chip = "pvc">] {
  gpu.module @valid_dpas {
    // CHECK: gpu.func @valid_dpas
    gpu.func @valid_dpas(%a: memref<24x32xf16>, %b: memref<32x24xf16>) {
      // CHECK: %[[TDESC_A:.*]] = xegpu.create_nd_tdesc %[[ARG0:.*]]{{\[}}0, 0]
      // CHECK-SAME: memref<24x32xf16> -> !xegpu.tensor_desc<24x32xf16
      %tdesc_a = xegpu.create_nd_tdesc %a[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<24x32xf16, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>

      // CHECK: %[[LOAD_A:.*]] = xegpu.load_nd %[[TDESC_A]]
      // CHECK-SAME: -> vector<24x32xf16>
      %load_a = xegpu.load_nd %tdesc_a : !xegpu.tensor_desc<24x32xf16, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>> -> vector<24x32xf16>

      // CHECK: %[[TDESC_B:.*]] = xegpu.create_nd_tdesc %[[ARG1:.*]]{{\[}}0, 0]
      // CHECK-SAME: memref<32x24xf16> -> !xegpu.tensor_desc<32x24xf16
      %tdesc_b = xegpu.create_nd_tdesc %b[0, 0] : memref<32x24xf16> -> !xegpu.tensor_desc<32x24xf16, #xegpu.layout<sg_layout = [4, 2], sg_data = [8, 12], lane_layout = [8, 2], lane_data = [1, 1]>>

      // CHECK: %[[LOAD_B:.*]] = xegpu.load_nd %[[TDESC_B]]
      // CHECK-SAME: -> vector<32x24xf16>
      %load_b = xegpu.load_nd %tdesc_b : !xegpu.tensor_desc<32x24xf16, #xegpu.layout<sg_layout = [4, 2], sg_data = [8, 12], lane_layout = [8, 2], lane_data = [1, 1]>> -> vector<32x24xf16>

      // CHECK: %[[DPAS:.*]] = xegpu.dpas %[[LOAD_A]], %[[LOAD_B]]
      // CHECK-SAME: layout_result_0 = #xegpu.layout<sg_layout = [2, 2], sg_data = [12, 12], lane_layout = [2, 2], lane_data = [1, 1]>
      // CHECK-SAME: : vector<24x32xf16>, vector<32x24xf16> -> vector<24x24xf16>
      %dpas = xegpu.dpas %load_a, %load_b {layout_result_0 =  #xegpu.layout<sg_layout = [2, 2], sg_data = [12, 12], lane_layout = [2, 2], lane_data = [1, 1]>} : vector<24x32xf16>, vector<32x24xf16> -> vector<24x24xf16>

      // CHECK: gpu.return
      gpu.return
    }
  }
}

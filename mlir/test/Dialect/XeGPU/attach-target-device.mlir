// RUN: mlir-opt --xegpu-attach-target-device="device-name=pvc" %s -split-input-file -verify-diagnostics

// module @valid_dpas attributes {gpu.container_module} {
//     gpu.module @valid_dpas attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL, Bfloat16ConversionINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute, SPV_INTEL_bfloat16_conversion]>, api=OpenCL, #spirv.resource_limits<>>} {

//     gpu.func @valid_dpas(%a: memref<24x32xf16>, %b: memref<32x24xf16>) {
//         %tdesc_a = xegpu.create_nd_tdesc %a[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<24x32xf16, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
//         %load_a =  xegpu.load_nd %tdesc_a : !xegpu.tensor_desc<24x32xf16, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
//         -> vector<24x32xf16>
//         %tdesc_b = xegpu.create_nd_tdesc %b[0, 0] : memref<32x24xf16> -> !xegpu.tensor_desc<32x24xf16, #xegpu.layout<sg_layout = [4, 2], sg_data = [8, 12], lane_layout = [8, 2], lane_data = [1, 1]>>
//         %load_b =  xegpu.load_nd %tdesc_b : !xegpu.tensor_desc<32x24xf16, #xegpu.layout<sg_layout = [4, 2], sg_data = [8, 12], lane_layout = [8, 2], lane_data = [1, 1]>> -> vector<32x24xf16>

//         %dpas = xegpu.dpas %load_a, %load_b {layout_result_0 =  #xegpu.layout<sg_layout = [2, 2], sg_data = [12, 12], lane_layout = [2, 2], lane_data = [1, 1]>} : vector<24x32xf16>, vector<32x24xf16> -> vector<24x24xf16>
//         gpu.return
//     }
//   }
// }


// RUN: mlir-opt %s -my-pass | FileCheck %s

// CHECK: module @valid_dpas
// CHECK-SAME: attributes {dlti.target_system_spec = #dlti.target_system_spec<"GPU" = #dlti.target_device_spec<"name" = "pvc">>, gpu.container_module}
module @valid_dpas attributes {gpu.container_module} {
  // CHECK: gpu.module @valid_dpas
  gpu.module @valid_dpas attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0,[],[]>,api = OpenCL,#spirv.resource_limits<>>} {
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

module @eltwise_add attributes {gpu.container_module} {
    gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL, Bfloat16ConversionINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute, SPV_INTEL_bfloat16_conversion]>, api=OpenCL, #spirv.resource_limits<>>} {

    gpu.func @dpas(%a: memref<24x32xf32>, %b: memref<32x24xf32>) {
        %tdesc_a = xegpu.create_nd_tdesc %a[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
        %load_a =  xegpu.load_nd %tdesc_a : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
        -> vector<24x32xf32>
        %tdesc_b = xegpu.create_nd_tdesc %b[0, 0] : memref<32x24xf32> -> !xegpu.tensor_desc<32x24xf32, #xegpu.layout<sg_layout = [4, 2], sg_data = [8, 12], lane_layout = [8, 2], lane_data = [1, 1]>>
        %load_b =  xegpu.load_nd %tdesc_b : !xegpu.tensor_desc<32x24xf32, #xegpu.layout<sg_layout = [4, 2], sg_data = [8, 12], lane_layout = [8, 2], lane_data = [1, 1]>> -> vector<32x24xf32>
        %dpas = xegpu.dpas %load_a, %load_b {layout_result_0 =  #xegpu.layout<sg_layout = [2, 2], sg_data = [12, 12], lane_layout = [2, 2], lane_data = [1, 1]>} : vector<24x32xf32>, vector<32x24xf32> -> vector<24x24xf32>
        gpu.return
    }
  }
}

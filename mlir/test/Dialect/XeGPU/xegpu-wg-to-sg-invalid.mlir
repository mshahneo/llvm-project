// RUN: mlir-opt --xegpu-wg-to-sg-distribute -split-input-file -verify-diagnostics %s

// `packed` reshapes the result of load_nd. Workgroup distribution cannot build
// that shape, so it must reject the op instead of dropping the attribute.
gpu.module @test_packed {
  gpu.func @load_nd_packed(%src: memref<64x64xf16>) {
    %tdesc = xegpu.create_nd_tdesc %src : memref<64x64xf16> -> !xegpu.tensor_desc<32x32xf16>
    // expected-error@+1 {{failed to legalize operation 'xegpu.load_nd'}}
    %load = xegpu.load_nd %tdesc[0, 0] <{layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 16]>, packed}>
        : !xegpu.tensor_desc<32x32xf16> -> vector<16x32x2xf16>
    gpu.return
  }
}

// -----

// A `transpose` on a non-square subgroup tile swaps the result dimensions. The
// type converter derives the result type from the unpermuted layout, so the op
// must be rejected instead of loading the tile untransposed.
gpu.module @test_transpose_non_square {
  gpu.func @load_nd_transpose_non_square(%src: memref<256x128xf16>) {
    %tdesc = xegpu.create_nd_tdesc %src : memref<256x128xf16> -> !xegpu.tensor_desc<256x128xf16>
    // expected-error@+1 {{failed to legalize operation 'xegpu.load_nd'}}
    %load = xegpu.load_nd %tdesc[0, 0] <{layout = #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 16]>, transpose = array<i64: 1, 0>}>
        : !xegpu.tensor_desc<256x128xf16> -> vector<128x256xf16>
    gpu.return
  }
}

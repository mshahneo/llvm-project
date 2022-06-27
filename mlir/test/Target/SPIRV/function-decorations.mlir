// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
    spv.func @linkage_attr_test_kernel()  "DontInline"  attributes {}  {
        %uchar_0 = spv.Constant 0 : i8
        %ushort_1 = spv.Constant 1 : i16
        %uint_0 = spv.Constant 0 : i32
        spv.FunctionCall @outside.func.with.linkage(%uchar_0):(i8) -> ()
        spv.Return
    }
    // CHECK: LinkageAttributes=["outside.func", "Import"]
    spv.func @outside.func.with.linkage(%arg0 : i8) -> () "Pure" attributes {LinkageAttributes=["outside.func", "Import"], VectorComputeFunctionINTEL}
    spv.func @inside.func() -> () "Pure" attributes {} {spv.Return}
}

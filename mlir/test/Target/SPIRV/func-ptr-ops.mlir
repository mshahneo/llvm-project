// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip -split-input-file %s | FileCheck %s
// RUN: mlir-translate -no-implicit-module --serialize-spirv -split-input-file %s | FileCheck %s
// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.func @foo(%arg: vector<2xi32>) -> (vector<2xi32>) "None" {
    spirv.ReturnValue %arg : vector<2xi32>
  }
  spirv.func @copy_memory_simple() "None" {
    %c = spirv.Constant -1 : i32
    %c_dup = spirv.Constant -1 : i32
    %0 = spirv.Variable : !spirv.ptr<f32, Function>
    %1 = spirv.Variable : !spirv.ptr<f32, Function>
    %3 = spirv.INTEL.ConstantFunctionPointer @foo : !spirv.INTEL.functionptr<(vector<2xi32>) -> vector<2xi32>, CodeSectionINTEL>
    %4 = spirv.INTEL.ConstantFunctionPointer @foo : !spirv.INTEL.functionptr<(vector<2xi32>) -> vector<2xi32>, CodeSectionINTEL>
    // CHECK: spirv.CopyMemory "Function" %{{.*}}, "Function" %{{.*}} : f32
    spirv.CopyMemory "Function" %0, "Function" %1 : f32
    %arg = spirv.Constant dense<[2, 3]> : vector<2xi32>
    %result = spirv.INTEL.FunctionPointerCall %3(%arg) : (!spirv.INTEL.functionptr<(vector<2xi32>) -> vector<2xi32>, CodeSectionINTEL>, (vector<2xi32>)) -> vector<2xi32>
    spirv.Return
  }
  spirv.func @duplicate_func_ptr() "None" {
    %5 = spirv.INTEL.ConstantFunctionPointer @foo : !spirv.INTEL.functionptr<(vector<2xi32>) -> vector<2xi32>, CodeSectionINTEL>
    spirv.Return
  }
}

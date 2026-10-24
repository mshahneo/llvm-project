; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - -filetype=obj | spirv-val %}

; FP conversions on a vector wider than 16 elements stay one instruction on the
; long vector type, like the integer casts in integer-casts.ll.

; CHECK-DAG: OpName [[FPEXT:%.*]] "fpext_v17"
; CHECK-DAG: OpName [[FPTRUNC:%.*]] "fptrunc_v17"
; CHECK-DAG: OpName [[SITOFP:%.*]] "sitofp_v17"
; CHECK-DAG: OpName [[UITOFP:%.*]] "uitofp_v17"
; CHECK-DAG: OpName [[FPTOSI:%.*]] "fptosi_v17"
; CHECK-DAG: OpName [[FPTOUI:%.*]] "fptoui_v17"

; CHECK-DAG: [[F16:%.*]] = OpTypeFloat 16
; CHECK-DAG: [[F32:%.*]] = OpTypeFloat 32
; CHECK-DAG: [[U32:%.*]] = OpTypeInt 32 0
; CHECK-DAG: [[SEVENTEEN:%.*]] = OpConstant [[U32]] 17
; CHECK-DAG: [[F16v17:%.*]] = OpTypeVectorIdEXT [[F16]] [[SEVENTEEN]]
; CHECK-DAG: [[F32v17:%.*]] = OpTypeVectorIdEXT [[F32]] [[SEVENTEEN]]
; CHECK-DAG: [[U32v17:%.*]] = OpTypeVectorIdEXT [[U32]] [[SEVENTEEN]]

; CHECK:      [[FPEXT]] = OpFunction [[F32v17]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[F16v17]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpFConvert [[F32v17]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <17 x float> @fpext_v17(<17 x half> %a) {
  %r = fpext <17 x half> %a to <17 x float>
  ret <17 x float> %r
}

; CHECK:      [[FPTRUNC]] = OpFunction [[F16v17]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[F32v17]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpFConvert [[F16v17]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <17 x half> @fptrunc_v17(<17 x float> %a) {
  %r = fptrunc <17 x float> %a to <17 x half>
  ret <17 x half> %r
}

; CHECK:      [[SITOFP]] = OpFunction [[F32v17]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U32v17]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpConvertSToF [[F32v17]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <17 x float> @sitofp_v17(<17 x i32> %a) {
  %r = sitofp <17 x i32> %a to <17 x float>
  ret <17 x float> %r
}

; CHECK:      [[UITOFP]] = OpFunction [[F32v17]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U32v17]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpConvertUToF [[F32v17]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <17 x float> @uitofp_v17(<17 x i32> %a) {
  %r = uitofp <17 x i32> %a to <17 x float>
  ret <17 x float> %r
}

; CHECK:      [[FPTOSI]] = OpFunction [[U32v17]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[F32v17]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpConvertFToS [[U32v17]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <17 x i32> @fptosi_v17(<17 x float> %a) {
  %r = fptosi <17 x float> %a to <17 x i32>
  ret <17 x i32> %r
}

; CHECK:      [[FPTOUI]] = OpFunction [[U32v17]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[F32v17]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpConvertFToU [[U32v17]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <17 x i32> @fptoui_v17(<17 x float> %a) {
  %r = fptoui <17 x float> %a to <17 x i32>
  ret <17 x i32> %r
}

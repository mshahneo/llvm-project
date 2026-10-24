; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --implicit-check-not="OpTypeVector %{{[0-9]+}} 32"
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Without SPV_EXT_long_vector, a 32-wide FP conversion is split into 16-wide ones,
; as the integer casts are. One element of each half is used, so both halves stay.

; CHECK-DAG: OpName [[FPEXT:%.*]] "fpext_v32"
; CHECK-DAG: OpName [[FPTRUNC:%.*]] "fptrunc_v32"
; CHECK-DAG: [[F16:%.*]] = OpTypeFloat 16
; CHECK-DAG: [[F32:%.*]] = OpTypeFloat 32
; CHECK-DAG: [[F16v16:%.*]] = OpTypeVector [[F16]] 16
; CHECK-DAG: [[F32v16:%.*]] = OpTypeVector [[F32]] 16

; CHECK:     [[FPEXT]] = OpFunction
; CHECK:     OpFConvert [[F32v16]]
; CHECK:     OpFConvert [[F32v16]]
; CHECK:     OpFunctionEnd
define spir_kernel void @fpext_v32(ptr addrspace(1) %out, half %a) {
  %v0 = insertelement <32 x half> poison, half %a, i32 0
  %v = shufflevector <32 x half> %v0, <32 x half> poison, <32 x i32> zeroinitializer
  %e = fpext <32 x half> %v to <32 x float>
  %lo = extractelement <32 x float> %e, i32 5
  %hi = extractelement <32 x float> %e, i32 20
  %s = fadd float %lo, %hi
  store float %s, ptr addrspace(1) %out
  ret void
}

; CHECK:     [[FPTRUNC]] = OpFunction
; CHECK:     OpFConvert [[F16v16]]
; CHECK:     OpFConvert [[F16v16]]
; CHECK:     OpFunctionEnd
define spir_kernel void @fptrunc_v32(ptr addrspace(1) %out, float %a) {
  %v0 = insertelement <32 x float> poison, float %a, i32 0
  %v = shufflevector <32 x float> %v0, <32 x float> poison, <32 x i32> zeroinitializer
  %t = fptrunc <32 x float> %v to <32 x half>
  %lo = extractelement <32 x half> %t, i32 5
  %hi = extractelement <32 x half> %t, i32 20
  %s = fadd half %lo, %hi
  store half %s, ptr addrspace(1) %out
  ret void
}

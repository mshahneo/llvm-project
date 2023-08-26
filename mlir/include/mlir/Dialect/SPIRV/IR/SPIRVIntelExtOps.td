//===- SPIRVIntelExtOps.td - Intel SPIR-V extensions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the op definition spec of Intel-specific SPIR-V extensions
// These extensions are not part of Khronos specification but publicly available
// at (https://github.com/intel/llvm)
// Supported extensions
// * SPV_INTEL_bfloat16_conversion
//===----------------------------------------------------------------------===//


#ifndef MLIR_DIALECT_SPIRV_IR_INTEL_EXT_OPS
#define MLIR_DIALECT_SPIRV_IR_INTEL_EXT_OPS

// -----

def SPIRV_INTELConvertFToBF16Op : SPIRV_IntelVendorOp<"ConvertFToBF16", []> {
  let summary = "See extension SPV_INTEL_bfloat16_conversion";

  let description = [{
    Convert value numerically from 32-bit floating point to bfloat16,
    which is represented as a 16-bit unsigned integer.

    Result Type must be a scalar or vector of integer type.
    The component width must be 16 bits. Bit pattern in the Result represents a bfloat16 value.

    Float Value must be a scalar or vector of floating-point type.
    It must have the same number of components as Result Type. The component width must be 32 bits.

    Results are computed per component.

    ```
    convert-f-to-bf16-op ::= ssa-id `=` `spirv.INTEL.ConvertFToBF16` ssa-use
                          `:` operand-type `to` result-type
    ```

    #### Example:

    ```mlir
    %1 = spirv.ConvertFToBF16 %0 : f32 to i16
    %3 = spirv.ConvertFToBF16 %2 : vector<3xf32> to vector<3xi16>
    ```

  }];


  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_INTEL_bfloat16_conversion]>,
    Capability<[SPIRV_C_Bfloat16ConversionINTEL]>
  ];

  let arguments = (ins
    SPIRV_ScalarOrVectorOf<SPIRV_Float32>:$operand
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<SPIRV_Int16>:$result
  );
  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `to` type($result)
  }];

  let hasVerifier = 1;
}

// -----

def SPIRV_INTELConvertBF16ToFOp : SPIRV_IntelVendorOp<"ConvertBF16ToF", []> {
  let summary = "See extension SPV_INTEL_bfloat16_conversion";

  let description = [{
    Interpret a 16-bit integer as bfloat16 and convert the value numerically to 32-bit floating point type.

    Result Type must be a scalar or vector of floating-point. The component width must be 32 bits.

    Bfloat16 Value must be a scalar or vector of integer type, which is interpreted as a bfloat16 type.
    The type must have the same number of components as the Result Type. The component width must be 16 bits.

    Results are computed per component.

    ```
    convert-bf16-to-f-op ::= ssa-id `=` `spirv.INTEL.ConvertBF16ToF` ssa-use
                          `:` operand-type `to` result-type
    ```

    #### Example:

    ```mlir
    %1 = spirv.ConvertBF16ToF %0 : i16 to f32
    %3 = spirv.ConvertBF16ToF %2 : vector<3xi16> to vector<3xf32>
    ```

  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_INTEL_bfloat16_conversion]>,
    Capability<[SPIRV_C_Bfloat16ConversionINTEL]>
  ];

  let arguments = (ins
    SPIRV_ScalarOrVectorOf<SPIRV_Int16>:$operand
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<SPIRV_Float32>:$result
  );

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `to` type($result)
  }];
  let hasVerifier = 1;
}


// -----

def SPIRV_INTELConstantFunctionPointerOp : SPIRV_IntelVendorOp<"ConstantFunctionPointer", [InFunctionScope]> {
  let summary = "See extension SPV_INTEL_function_pointers";

  let description = [{
    Obtains constant address of the specified function.

    Result value can be used immediately in OpFunctionPointerCallINTEL, inserted into a composite constant or stored somewhere for further usage in OpFunctionPointerCallINTEL.

    Result Type must be an OpTypePointer. Its Type operand must be the same OpTypeFunction which was used as Function Type operand of the Function operand. Its Storage Class operand must be CodeSectionINTEL.

    NOTE: In SPIR-V dialect function pointers follow pointer scoping mechanism. To use a function pointer it must be obtained in the same scope.

    ```
    constant-function-pointer-op ::= ssa-id `=` `spirv.INTEL.ConstantFunctionPointer` function-id : function-type-ptr

    ```

    #### Example:

    ```mlir

    spirv.func @foo(vector<2xi32>) -> (vector<2xi32>) "None" { ... }
    spirv.func @bar(i32) -> (i32) "None" { ... }
    %funcPtrFoo = spirv.INTEL.ConstantFunctionPointer @foo : !spirv.INTEL.functionptr<(vector<2xi32>) -> vector<2xi32>, CodeSectionINTEL>
    %funcPtrBar = spirv.INTEL.ConstantFunctionPointer @bar : !spirv.INTEL.functionptr<(i32) -> i32, CodeSectionINTEL>

    ```

  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_INTEL_function_pointers]>,
    Capability<[SPIRV_C_FunctionPointersINTEL]>
  ];

  let arguments = (ins
    FlatSymbolRefAttr:$fn
    // ,
    // TypeAttrOf<FunctionType>:$function_type
  );

  let results = (outs
    SPIRV_AnyFunctionPtrINTEL:$fn_ptr
  );

  let assemblyFormat = [{
    $fn attr-dict `:` type($fn_ptr)
  }];

  let autogenSerialization = 0;
  let hasVerifier = 1;
}


// -----
// , DeclareOpInterfaceMethods<CallOpInterface>
def SPIRV_INTELFunctionPointerCallOp : SPIRV_IntelVendorOp<"FunctionPointerCall",
    [InFunctionScope]> {
  let summary = "See extension SPV_INTEL_function_pointers";

  let description = [{
    Call a function via function pointer.

    Result Type is the type of the return value of the function.

    Function Pointer is Function Pointer.

    Argument N is the object to copy to parameter N.

    Note: Result Type must match the Return Type of the OpTypeFunction which was used as Type operand of Function Pointer argument and the calling argument types must match the formal parameter types.

    ```
    constant-function-pointer-op ::= ssa-id `=` `spirv.INTEL.ConstantFunctionPointer` function-id : function-type-ptr

    function-pointer-call-op ::= `spirv.INTEL.FunctionPointerCall` ssa-use `(` ssa-use-list `)`
                     `:` function-type

    ```

    #### Example:

    ```mlir

    spirv.func @foo(vector<2xi32>) -> (vector<2xi32>) "None" { ... }
    %funcPtrFoo = spirv.INTEL.ConstantFunctionPointer @foo : !spirv.ptr<(vector<2xi32>) -> vector<2xi32>, CodeSectionINTEL>
    %arg = spirv.Constant dense<[2, 3]> : vector<2xi32>
    %result = spirv.spirv.INTEL.FunctionPointerCall %funcPtrFoo(%arg) : (!spirv.ptr<(vector<2xi32>) -> vector<2xi32>, CodeSectionINTEL>, vector<2xi32>) -> vector<2xi32>

    ```

  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_INTEL_function_pointers]>,
    Capability<[SPIRV_C_FunctionPointersINTEL]>
  ];

  let arguments = (ins
    SPIRV_AnyFunctionPtrINTEL:$fn_ptr,
    Variadic<SPIRV_Type>:$func_arguments
  );

  let results = (outs
    Optional<SPIRV_Type>:$return_value
  );


  // There are two ways to define the assembly format,

  // spirv.FunctionCall way, this would require a custom builder

  // let assemblyFormat = [{
  //   $fn_ptr `(` $arguments `)` attr-dict `:`
  //     functional-type($func_arguments, results)
  // }];

  // let builders = [
  //   OpBuilder<(ins "Value":$fn_ptr, ""ValueRange":":$func_arguments)>
  // ];

  // spirv.CL.printf way
  // Refer to: spirv.CL.printf

  let assemblyFormat = [{
    $fn_ptr `(` $func_arguments `)` attr-dict `:`
      `(` type($fn_ptr) `,` `(` type($func_arguments) `)` `)` `->` type($return_value)
  }];

  let autogenSerialization = 0;
  let hasVerifier = 1;

}


// -----


#endif // MLIR_DIALECT_SPIRV_IR_INTEL_EXT_OPS
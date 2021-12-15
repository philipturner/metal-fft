//
//  ComplexUtilities.swift
//  MetalFFT
//
//  Created by Philip Turner on 12/11/21.
//

import simd

func addComplex(_ x: simd_float2, _ y: simd_float2) -> simd_float2 {
    let real = x[0] + y[0]
    let imag = x[1] + y[1]
    
    return simd_float2(real, imag)
}

func subtractComplex(_ x: simd_float2, _ y: simd_float2) -> simd_float2 {
    let real = x[0] - y[0]
    let imag = x[1] - y[1]
    
    return simd_float2(real, imag)
}

func multiplyComplex(_ x: simd_float2, _ y: simd_float2) -> simd_float2 {
    let real = x[0] * y[0] - x[1] * y[1]
    let imag = x[0] * y[1] + x[1] * y[0]
    
    return simd_float2(real, imag)
}

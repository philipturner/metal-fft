//
//  ConvertPipeline.swift
//  MetalFFT
//
//  Created by Philip Turner on 12/12/21.
//

import ARHeadsetUtil
import Metal

struct ConvertPipelineDescriptor: Hashable {
    var deviceRegistryID: UInt64
    var inputType: FFTDataType
    var outputType: FFTDataType
}

func makeConvertPipeline(library: MTLLibrary, descriptor convertDesc: ConvertPipelineDescriptor) -> MTLComputePipelineState {
    let device = library.device
    precondition(device.registryID == convertDesc.deviceRegistryID, "Attempted to create an FFT convert pipeline for a different device than registered in the descriptor")
    
    // Create compute function
    
    let constants = MTLFunctionConstantValues()
    
    var booleanValues = simd_bool4(convertDesc.inputType == .real,
                                   convertDesc.inputType == .splitComplex,
                                   convertDesc.outputType == .real,
                                   convertDesc.outputType == .splitComplex)
    constants.setConstantValues(&booleanValues, type: .bool, range: 0..<4)
    
    let function = try! library.makeFunction(name: "fft_convert", constantValues: constants)
    
    // Create compute pipeline
    
    let pipelineDesc = MTLComputePipelineDescriptor()
    pipelineDesc.computeFunction = function
    debugLabel {
        pipelineDesc.label = "fft_convert (inputFormat: \(convertDesc.inputType), outputFormat: \(convertDesc.outputType))"
    }
    
    if device.supportsFamily(.apple3) || device.supportsFamily(.mac2) {
        pipelineDesc.supportIndirectCommandBuffers = true
    }
    
    return device.makeComputePipelineState(descriptor: pipelineDesc)
}

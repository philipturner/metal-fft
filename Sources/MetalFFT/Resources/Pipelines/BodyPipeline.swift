//
//  BodyPipeline.swift
//  MetalFFT
//
//  Created by Philip Turner on 12/12/21.
//

import ARHeadsetUtil
import Metal

struct BodyPipelineDescriptor: Hashable {
    var deviceRegistryID: UInt64
    var gridNumDims: Int
    var fftDirection: Int
    var readingTwiddleFactors: Bool
}

func makeBodyPipeline(library: MTLLibrary, descriptor bodyDesc: BodyPipelineDescriptor) -> MTLComputePipelineState {
    let device = library.device
    precondition(device.registryID == bodyDesc.deviceRegistryID, "Attempted to create an FFT body pipeline for a different device than registered in the descriptor")
    
    // Create compute function
    
    let constants = MTLFunctionConstantValues()
    
    var ushortValues = simd_ushort2(UInt16(bodyDesc.gridNumDims), UInt16(bodyDesc.fftDirection))
    constants.setConstantValues(&ushortValues, type: .ushort, range: 0..<2)
    
    var readingTwiddleFactors_copy = bodyDesc.readingTwiddleFactors
    constants.setConstantValue(&readingTwiddleFactors_copy, type: .bool, index: 2)
    
    let function = try! library.makeFunction(name: "fft_body", constantValues: constants)
    
    // Create compute pipeline
    
    let pipelineDesc = MTLComputePipelineDescriptor()
    pipelineDesc.computeFunction = function
    debugLabel {
        pipelineDesc.label = "fft_body (gridNumDimensions: \(bodyDesc.gridNumDims), fftDirection: \(["X", "Y", "Z"][bodyDesc.fftDirection]), readingTwiddleFactors: \(bodyDesc.readingTwiddleFactors))"
    }
    
    if device.supportsFamily(.apple3) || device.supportsFamily(.mac2) {
        pipelineDesc.supportIndirectCommandBuffers = true
    }
    
    return device.makeComputePipelineState(descriptor: pipelineDesc)
}

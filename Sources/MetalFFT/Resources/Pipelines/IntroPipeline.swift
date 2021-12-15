//
//  IntroPipeline.swift
//  MetalFFT
//
//  Created by Philip Turner on 12/12/21.
//

import ARHeadsetUtil
import Metal

struct IntroPipelineDescriptor: Hashable {
    var deviceRegistryID: UInt64
    var gridNumDims: Int
    var direction: Int
    var inputType: FFTDataType
    var isInverse: Bool
}

func makeIntroPipeline(library: MTLLibrary, descriptor introDesc: IntroPipelineDescriptor) -> MTLComputePipelineState {
    let device = library.device
    precondition(device.registryID == introDesc.deviceRegistryID, "Attempted to create an FFT intro pipeline for a different device than registered in the descriptor")
    
    // Create compute function
    
    let constants = MTLFunctionConstantValues()
    
    var ushortValues = simd_ushort2(UInt16(introDesc.gridNumDims), UInt16(introDesc.direction))
    constants.setConstantValues(&ushortValues, type: .ushort, range: 0..<2)
    
    var booleanValues = simd_bool3(introDesc.inputType == .real, introDesc.inputType == .splitComplex, introDesc.isInverse)
    constants.setConstantValues(&booleanValues, type: .bool, range: 2..<5)
    
    let function = try! library.makeFunction(name: "fft_intro", constantValues: constants)
    
    // Create compute pipeline
    
    let pipelineDesc = MTLComputePipelineDescriptor()
    pipelineDesc.computeFunction = function
    debugLabel {
        pipelineDesc.label = "fft_intro (gridNumDimensions: \(introDesc.gridNumDims), direction: \(["X", "Y", "Z"][introDesc.direction]), inputType: \(introDesc.inputType), isInverse: \(introDesc.isInverse))"
    }
    
    if device.supportsFamily(.apple3) || device.supportsFamily(.mac2) {
        pipelineDesc.supportIndirectCommandBuffers = true
    }
    
    return device.makeComputePipelineState(descriptor: pipelineDesc)
}

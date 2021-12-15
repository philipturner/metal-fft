//
//  ICBUtilities.swift
//  MetalFFTTests
//
//  Created by Philip Turner on 12/13/21.
//

import XCTest
@testable import MetalFFT
import ARHeadsetUtil

func canUseICBs(device: MTLDevice) -> Bool {
    guard device.supportsFamily(.apple3) || device.supportsFamily(.mac2) else {
        print("WARNING: This device does not support indirect command buffers. Skipping over tests that use indirect command buffers.")
        return false
    }
    
    return true
}

extension FastFourierTransform {
    
    func encodeAsICB(commandBuffer: MTLCommandBuffer, input: AnyFFTBuffer, output: FFTComplexBuffer) {
        let resources = self.resources
        
        let desc = MTLIndirectCommandBufferDescriptor()
        desc.commandTypes = .concurrentDispatch
        desc.inheritBuffers = false
        desc.maxKernelBufferBindCount = resources.maxKernelBufferBindCount
        
        let icb = commandBuffer.device.makeIndirectCommandBuffer(descriptor: desc, maxCommandCount: resources.maxCommandCount, options: [])!
        encode(indirectCommandBuffer: icb,
               range: 0..<resources.maxCommandCount,
               input: input,
               output: output)
        
        let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
        encodeUseResources(computeEncoder: computeEncoder, input: input, output: output)
        
        computeEncoder.executeCommandsInBuffer(icb, range: 0..<resources.maxCommandCount)
        computeEncoder.endEncoding()
    }
    
}

//
//  TwiddleFactorBuffer.swift
//  MetalFFT
//
//  Created by Philip Turner on 12/11/21.
//

import ARHeadsetUtil
import Metal

class TwiddleFactorBuffer {
    let buffer: MTLBuffer
    
    init(device: MTLDevice, descriptor desc: TwiddleFactorBufferDescriptor, commandQueue: MTLCommandQueue?) {
        precondition(desc.fftSize.nonzeroBitCount == 1, "FFT size must be a power of 2")
        precondition(desc.fftSize <= 2 << 31, "FFT size cannot exceed 2 billion")
        
        let options = device.fftResourceOptions
        precondition(device.registryID == desc.deviceRegistryID, "Attempted to create an FFT twiddle factor buffer for a different device than registered in the descriptor")
        
        let bufferSize = desc.allocationSize
        buffer = device.makeBuffer(length: bufferSize, options: [options, .cpuCacheModeWriteCombined])!
        buffer.optLabel = "FFT Twiddle Factor Buffer"
        
        desc.fillAllocation(buffer.contents())
        
        #if os(macOS)
        if device.fftResourceOptions == .storageModeManaged {
            commandQueue!.synchronize(buffer)
        }
        #endif
    }
    
    func bind(to computeEncoder: MTLComputeCommandEncoder, index: Int) {
        computeEncoder.setBuffer(buffer, offset: 0, index: index)
    }
    
    func bind(to command: MTLIndirectComputeCommand, at index: Int) {
        command.setKernelBuffer(buffer, offset: 0, at: index)
    }
    
    var allocationSize: Int {
        buffer.length
    }
}

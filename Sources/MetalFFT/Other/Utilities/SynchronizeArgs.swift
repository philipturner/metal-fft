//
//  SynchronizeArgs.swift
//  MetalFFT
//
//  Created by Philip Turner on 12/11/21.
//

import ARHeadsetUtil

extension MTLDevice {
    @inline(__always)
    internal var fftResourceOptions: MTLResourceOptions {
        #if os(macOS)
        if hasUnifiedMemory {
            return .storageModeShared
        } else {
            return .storageModeManaged
        }
        #else
        return .storageModeShared
        #endif
    }
}

#if os(macOS)
extension MTLCommandBuffer {
    func synchronize(_ buffer: MTLBuffer) {
        assert(device.fftResourceOptions == .storageModeManaged, "This function should never be called on a unified memory architecture")
        
        let blitEncoder = makeBlitCommandEncoder()!
        blitEncoder.optLabel = "Synchronize FFT Buffer"
        
        blitEncoder.synchronize(resource: buffer)
        blitEncoder.endEncoding()
    }
    
    func synchronize(_ buffers: [MTLBuffer]) {
        assert(device.fftResourceOptions == .storageModeManaged, "This function should never be called on a unified memory architecture")
        
        let blitEncoder = makeBlitCommandEncoder()!
        blitEncoder.optLabel = "Synchronize FFT Buffers"
        
        buffers.forEach(blitEncoder.synchronize(resource:))
        blitEncoder.endEncoding()
    }
}

extension MTLCommandQueue {
    func synchronize(_ buffer: MTLBuffer) {
        let commandBuffer = makeDebugCommandBuffer()
        commandBuffer.synchronize(buffer)
        commandBuffer.commit()
    }
    
    func synchronize(_ buffers: [MTLBuffer]) {
        let commandBuffer = makeDebugCommandBuffer()
        commandBuffer.synchronize(buffers)
        commandBuffer.commit()
    }
}
#endif

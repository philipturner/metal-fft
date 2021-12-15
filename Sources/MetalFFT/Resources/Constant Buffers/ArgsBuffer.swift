//
//  ArgsBuffer.swift
//  MetalFFT
//
//  Created by Philip Turner on 12/12/21.
//

import MTLLayeredBufferModule

struct ArgsBuffer {
    let layeredBuffer: MTLLayeredBuffer<ArgsLayer>
    
    init(device: MTLDevice, descriptor: ArgsBufferDescriptor, commandQueue: MTLCommandQueue?) {
        layeredBuffer = descriptor.makeArgsBuffer(device: device)
        
        #if os(macOS)
        if device.fftResourceOptions == .storageModeManaged {
            commandQueue!.synchronize(layeredBuffer.buffer)
        }
        #endif
    }
    
    func bindIntro(to computeEncoder: MTLComputeCommandEncoder, offset: Int, index: Int) {
        computeEncoder.setBuffer(layeredBuffer.buffer, offset: offset, index: index)
    }
    
    func bindIntro(to command: MTLIndirectComputeCommand, offset: Int, at index: Int) {
        command.setKernelBuffer(layeredBuffer.buffer, offset: offset, at: index)
    }
    
    func rebindBody(to computeEncoder: MTLComputeCommandEncoder, offset: Int, index: Int) {
        computeEncoder.setBuffer(layeredBuffer,
                                 layer: .bodyArgs,
                                 offset: offset,
                                 index: index,
                                 bound: true)
    }
    
    func bindBody(to command: MTLIndirectComputeCommand, offset: Int, at index: Int) {
        command.setKernelBuffer(layeredBuffer, layer: .bodyArgs, offset: offset, at: index)
    }
}

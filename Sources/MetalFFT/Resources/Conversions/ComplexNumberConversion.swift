//
//  ComplexNumberConversion.swift
//  MetalFFT
//
//  Created by Philip Turner on 12/12/21.
//

import ARHeadsetUtil
import Metal

@usableFromInline
struct ConversionDescriptor: Hashable {
    var deviceRegistryID: UInt64
    @usableFromInline var inputType: FFTDataType
    @usableFromInline var outputType: FFTDataType
    
    func makePipeline(device: MTLDevice) -> MTLComputePipelineState {
        LocalCache.threadLocal.makeComplexNumberConversion(device: device, descriptor: self)
    }
}

public class ComplexNumberConversion {
    @usableFromInline let descriptor: ConversionDescriptor
    let pipeline: MTLComputePipelineState
    
    @inlinable
    public var inputType: FFTDataType {
        descriptor.inputType
    }
    
    @inlinable
    public var outputType: FFTDataType {
        descriptor.outputType
    }
    
    @inlinable @nonobjc
    public var maxKernelBufferBindCount: Int {
        descriptor.outputType == .splitComplex ? 4 : 3
    }
    
    @inlinable @nonobjc
    public var maxCommandCount: Int { 1 }
    
    @inlinable
    public var usedResources: [(resource: MTLResource, usage: MTLResourceUsage)] { [] }
    
    public init(device: MTLDevice,
                inputType: FFTDataType,
                outputType: FFTDataType) {
        descriptor = ConversionDescriptor(deviceRegistryID: device.registryID,
                                          inputType: inputType,
                                          outputType: outputType)
        pipeline = descriptor.makePipeline(device: device)
    }
    
}

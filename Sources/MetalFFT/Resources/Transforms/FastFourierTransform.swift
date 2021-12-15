//
//  FastFourierTransform.swift
//  MetalFFT
//
//  Created by Philip Turner on 12/12/21.
//

import Metal

@usableFromInline
struct TransformDescriptor: Hashable {
    var deviceRegistryID: UInt64
    var gridDimensions: MTLSize
    var dimensionality: Int
    @usableFromInline var inputType: FFTDataType
    var isInverse: Bool
    
    func makeResources(device: MTLDevice) -> TransformResources {
        LocalCache.threadLocal.makeFastFourierTransform(device: device, descriptor: self)
    }
}

@usableFromInline
struct TransformResources {
    var introPipelines: [MTLComputePipelineState?]
    var bodyPipelines: [MTLComputePipelineState?]
    var convertPipelines: [MTLComputePipelineState?]
    
    var argsBuffer: ArgsBuffer
    var twiddleFactorBuffer: TwiddleFactorBuffer
    var scratchBuffer: FFTComplexBuffer?
    
    @usableFromInline var maxKernelBufferBindCount: Int
    @usableFromInline var maxCommandCount: Int
}

public class FastFourierTransform {
    @usableFromInline var descriptor: TransformDescriptor
    @usableFromInline var resources: TransformResources
    
    init(descriptor: TransformDescriptor, resources: TransformResources) {
        self.descriptor = descriptor
        self.resources = resources
    }
}

extension FastFourierTransform {
    
    @inlinable
    public var inputType: FFTDataType {
        descriptor.inputType
    }
    
    @inlinable @nonobjc
    public var maxKernelBufferBindCount: Int {
        resources.maxKernelBufferBindCount
    }
    
    @inlinable @nonobjc
    public var maxCommandCount: Int {
        resources.maxCommandCount
    }
    
    public var usedResources: [(resource: MTLResource, usage: MTLResourceUsage)] {
        var output = [
            (resources.argsBuffer.layeredBuffer.buffer, MTLResourceUsage.read),
            (resources.twiddleFactorBuffer.buffer, MTLResourceUsage.read),
        ]
        
        if let scratchBuffer = resources.scratchBuffer {
            output.append((scratchBuffer.buffer, [.read, .write]))
        }
        
        return output
    }
    
}

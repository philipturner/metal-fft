//
//  ConversionEncodingICB.swift
//  MetalFFT
//
//  Created by Philip Turner on 12/14/21.
//

import ARHeadsetUtil
import Metal

extension ComplexNumberConversion {
    
    @inlinable
    public func encodeUseResources(computeEncoder: MTLComputeCommandEncoder, input: AnyFFTBuffer, output: AnyFFTBuffer) {
        for (buffer, usage) in [(input, MTLResourceUsage.read), (output, .write)] {
            if let realBuffer = buffer as? FFTRealBuffer {
                computeEncoder.useResource(realBuffer.buffer, usage: usage)
            } else if let complexBuffer = buffer as? FFTComplexBuffer {
                computeEncoder.useResource(complexBuffer.buffer, usage: usage)
            } else {
                let splitBuffer = buffer as! FFTSplitComplexBuffer
                computeEncoder.useResource(splitBuffer.realBuffer, usage: usage)
                computeEncoder.useResource(splitBuffer.imaginaryBuffer, usage: usage)
            }
        }
    }
    
    public func encode(indirectCommandBuffer: MTLIndirectCommandBuffer, range: Range<Int>, input: AnyFFTBuffer, output: AnyFFTBuffer) {
        precondition(range.count >= 1, "Needed to reserve at least 1 command specified by a ComplexNumberConversion's maxCommandCount, but instead reserved \(range.count)).")
        
        debugLabel {
            print("Warning: MetalFFT has not been tested with indirect command buffers yet. There may be bugs or crashes.")
        }
        
        let computeCommand = indirectCommandBuffer.indirectComputeCommandAt(range.startIndex)
        computeCommand.reset()
        computeCommand.setBarrier()
        assert(pipeline.supportIndirectCommandBuffers)
        computeCommand.setComputePipelineState(pipeline)
        
        func encode(type: FFTDataType, opaqueBuffer: AnyFFTBuffer, baseIndex: Int) {
            switch type {
            case .real:
                let realBuffer = opaqueBuffer as! FFTRealBuffer
                computeCommand.setKernelBuffer(realBuffer.buffer, offset: realBuffer.byteOffset, at: baseIndex)
                
            case .interleavedComplex:
                let complexBuffer = opaqueBuffer as! FFTComplexBuffer
                computeCommand.setKernelBuffer(complexBuffer.buffer, offset: complexBuffer.byteOffset, at: baseIndex)
                
            case .splitComplex:
                let splitBuffer = opaqueBuffer as! FFTSplitComplexBuffer
                computeCommand.setKernelBuffer(splitBuffer.realBuffer, offset: splitBuffer.realByteOffset, at: baseIndex)
                computeCommand.setKernelBuffer(splitBuffer.imaginaryBuffer, offset: splitBuffer.imaginaryByteOffset, at: baseIndex + 1)
            }
        }
        
        encode(type: descriptor.inputType, opaqueBuffer: input, baseIndex: 0)
        encode(type: descriptor.outputType, opaqueBuffer: output, baseIndex: 2)
        
        computeCommand.concurrentDispatchThreadgroups([ input.capacity ], threadsPerThreadgroup: 1)
    }
    
}

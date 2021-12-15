//
//  ConversionEncoding.swift
//  MetalFFT
//
//  Created by Philip Turner on 12/12/21.
//

import ARHeadsetUtil
import Metal

extension ComplexNumberConversion {
    
    public func encode(commandBuffer: MTLCommandBuffer, input: AnyFFTBuffer, output: AnyFFTBuffer) {
        let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
        debugLabel {
            computeEncoder.label = "FFT Complex Number Conversion (\(descriptor.inputType) to \(descriptor.outputType))"
        }
        
        computeEncoder.setComputePipelineState(pipeline)
        
        func encode(type: FFTDataType, opaqueBuffer: AnyFFTBuffer, baseIndex: Int) {
            switch type {
            case .real:
                let realBuffer = opaqueBuffer as! FFTRealBuffer
                computeEncoder.setBuffer(realBuffer.buffer, offset: realBuffer.byteOffset, index: baseIndex)
                
            case .interleavedComplex:
                let complexBuffer = opaqueBuffer as! FFTComplexBuffer
                computeEncoder.setBuffer(complexBuffer.buffer, offset: complexBuffer.byteOffset, index: baseIndex)
                
            case .splitComplex:
                let splitBuffer = opaqueBuffer as! FFTSplitComplexBuffer
                computeEncoder.setBuffer(splitBuffer.realBuffer, offset: splitBuffer.realByteOffset, index: baseIndex)
                computeEncoder.setBuffer(splitBuffer.imaginaryBuffer, offset: splitBuffer.imaginaryByteOffset, index: baseIndex + 1)
            }
        }
        
        encode(type: descriptor.inputType, opaqueBuffer: input, baseIndex: 0)
        encode(type: descriptor.outputType, opaqueBuffer: output, baseIndex: 2)
        
        computeEncoder.dispatchThreadgroups([ input.capacity ], threadsPerThreadgroup: 1)
        computeEncoder.endEncoding()
    }
    
    
    
}

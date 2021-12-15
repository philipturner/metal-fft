//
//  TransformEncoding.swift
//  MetalFFT
//
//  Created by Philip Turner on 12/12/21.
//

import MTLLayeredBufferModule

extension FastFourierTransform {
    
    public func encode(commandBuffer: MTLCommandBuffer, input: AnyFFTBuffer, output: FFTComplexBuffer) {
        assert(input.dataType == descriptor.inputType, "Fast Fourier Transform expected input buffer of type \(descriptor.inputType) but got \(input.dataType)")
        assert(input.fitsGrid(descriptor.gridDimensions), "Fast Fourier Transform had input buffer that was too small (element capacity: \(input.capacity)")
        assert(output.fitsGrid(descriptor.gridDimensions), "Fast Fourier Transform had output buffer that was too small (element capacity: \(output.capacity)")
        
        let desc = self.descriptor
        let resources = self.resources
        let gridDimensions = desc.gridDimensions
        
        // Return early in an edge case
        
        guard !isNullFFT(gridDimensions: gridDimensions, dimensionality: desc.dimensionality) else {
            debugLabel {
                print("Warning: Encoded empty Fast Fourier Transform (grid size was 1 in each dimension), which is equivalent to copying the input to the output")
            }
            
            let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
            computeEncoder.optLabel = "Copy Fallback for Empty Fast Fourier Transform"
            defer { computeEncoder.endEncoding() }
            
            computeEncoder.setComputePipelineState(resources.convertPipelines[0]!)
            
            if let realBuffer = input as? FFTRealBuffer {
                computeEncoder.setBuffer(realBuffer.buffer, offset: realBuffer.byteOffset, index: 0)
                
            } else if let complexBuffer = input as? FFTComplexBuffer {
                computeEncoder.setBuffer(complexBuffer.buffer, offset: complexBuffer.byteOffset, index: 0)
                
            } else {
                let splitBuffer = input as! FFTSplitComplexBuffer
                
                computeEncoder.setBuffer(splitBuffer.realBuffer, offset: splitBuffer.realByteOffset, index: 0)
                computeEncoder.setBuffer(splitBuffer.imaginaryBuffer, offset: splitBuffer.imaginaryByteOffset, index: 1)
            }
            
            computeEncoder.setBuffer(output.buffer, offset: output.byteOffset, index: 2)
            
            let numElements = gridDimensions.width * gridDimensions.height * gridDimensions.depth
            computeEncoder.dispatchThreadgroups([ numElements ], threadsPerThreadgroup: 1)
            
            return
        }
        
        // Find grid metadata
        
        var maxFFTSize = gridDimensions.width
        if desc.dimensionality >= 2 { maxFFTSize = max(maxFFTSize, gridDimensions.height) }
        if desc.dimensionality == 3 { maxFFTSize = max(maxFFTSize, gridDimensions.depth) }
        
        @inline(__always)
        func getDispatchDimensions(fftDirection: Int) -> MTLSize {
            var output = gridDimensions
            
            switch fftDirection {
            case 0:
                output.width /= 2
            case 1:
                output.height /= 2
            default:
                output.depth /= 2
            }
            
            return output
        }
        
        // Encode intro pass
        
        var introSourceBuffer: AnyFFTBuffer? = input
        
        func encodeIntro(fftDirection: Int) -> MTLComputeCommandEncoder? {
            guard let pipeline = resources.introPipelines[fftDirection] else {
                return nil
            }
            
            defer {
                introSourceBuffer = resources.scratchBuffer
            }
            
            if let tempSourceBufferRef = introSourceBuffer as? FFTComplexBuffer?,
               let introSourceBuffer = tempSourceBufferRef,
               introSourceBuffer.buffer === resources.scratchBuffer?.buffer {
                
                let blitEncoder = commandBuffer.makeBlitCommandEncoder()!
                debugLabel {
                    let dimString = [nil, "2D", "3D"][fftDirection]!
                    blitEncoder.label = "FFT Copy Preceding \(dimString) Stage"
                }
                
                defer { blitEncoder.endEncoding() }
                
                let numElements = gridDimensions.width * gridDimensions.height * gridDimensions.depth
                let fftComplexDataSize = numElements * MemoryLayout<simd_float2>.stride
                
                blitEncoder.copy(from: output.buffer,
                                 sourceOffset: output.byteOffset,
                                 to: introSourceBuffer.buffer,
                                 destinationOffset: introSourceBuffer.byteOffset,
                                 size: fftComplexDataSize)
            }
            
            let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
            debugLabel {
                let stageDimString = "\(fftDirection + 1)D"
                let totalDimString = "\(desc.dimensionality)D"
                computeEncoder.label = "FFT - \(stageDimString) Stage of \(totalDimString) Transform"
            }
            
            computeEncoder.setComputePipelineState(pipeline)
            
            let introArgsOffset = fftDirection * safeArgsStride(MemoryLayout<IntroArgs>.stride)
            resources.argsBuffer.bindIntro(to: computeEncoder, offset: introArgsOffset, index: 0)
            computeEncoder.setBuffer(output.buffer, offset: output.byteOffset, index: 2)
            
            if let complexBuffer = introSourceBuffer! as? FFTComplexBuffer {
                computeEncoder.setBuffer(complexBuffer.buffer, offset: complexBuffer.byteOffset, index: 1)
                
            } else if let realBuffer = introSourceBuffer! as? FFTRealBuffer {
                computeEncoder.setBuffer(realBuffer.buffer, offset: realBuffer.byteOffset, index: 1)
                
            } else {
                let splitBuffer = introSourceBuffer as! FFTSplitComplexBuffer
                computeEncoder.setBuffer(splitBuffer.realBuffer, offset: splitBuffer.realByteOffset, index: 1)
                computeEncoder.setBuffer(splitBuffer.imaginaryBuffer, offset: splitBuffer.imaginaryByteOffset, index: 3)
            }
            
            let dispatchDimensions = getDispatchDimensions(fftDirection: fftDirection)
            computeEncoder.dispatchThreadgroups(dispatchDimensions, threadsPerThreadgroup: 1)
            
            return computeEncoder
        }
        
        // Encode body pass
        
        var bodyArgsOffset = 0
        
        func encodeBody(computeEncoder: MTLComputeCommandEncoder, fftDirection: Int, levels: ClosedRange<Int>) {
            computeEncoder.setComputePipelineState(resources.bodyPipelines[fftDirection]!)
            resources.twiddleFactorBuffer.bind(to: computeEncoder, index: 1)
            
            let dispatchDimensions = getDispatchDimensions(fftDirection: fftDirection)
            
            for _ in levels {
                resources.argsBuffer.rebindBody(to: computeEncoder, offset: bodyArgsOffset, index: 0)
                computeEncoder.dispatchThreadgroups(dispatchDimensions, threadsPerThreadgroup: 1)
                
                bodyArgsOffset += safeArgsStride(MemoryLayout<BodyArgs>.stride)
            }
        }
        
        let gridDimensionsVector = simd_long3(gridDimensions.width, gridDimensions.height, gridDimensions.depth)
        
        for fftDirection in 0..<desc.dimensionality {
            let dimension = gridDimensionsVector[fftDirection]
            
            guard let computeEncoder = encodeIntro(fftDirection: fftDirection) else {
                continue
            }
            
            defer { computeEncoder.endEncoding() }
            
            if dimension >= 4 {
                encodeBody(computeEncoder: computeEncoder, fftDirection: fftDirection, levels: 2...dimension.trailingZeroBitCount)
            }
        }
    }
    
}

//
//  TransformEncodingICB.swift
//  MetalFFT
//
//  Created by Philip Turner on 12/12/21.
//

import MTLLayeredBufferModule

extension FastFourierTransform {
    
    public func encodeUseResources(computeEncoder: MTLComputeCommandEncoder, input: AnyFFTBuffer, output: FFTComplexBuffer) {
        for (resource, usage) in usedResources {
            computeEncoder.useResource(resource, usage: usage)
        }
        
        if let realBuffer = input as? FFTRealBuffer {
            computeEncoder.useResource(realBuffer.buffer, usage: .read)
        } else if let complexBuffer = input as? FFTComplexBuffer {
            computeEncoder.useResource(complexBuffer.buffer, usage: .read)
        } else {
            let splitBuffer = input as! FFTSplitComplexBuffer
            computeEncoder.useResource(splitBuffer.realBuffer, usage: .read)
            computeEncoder.useResource(splitBuffer.imaginaryBuffer, usage: .read)
        }
        
        computeEncoder.useResource(output.buffer, usage: [.read, .write])
    }
    
    public func encode(indirectCommandBuffer: MTLIndirectCommandBuffer, range: Range<Int>, input: AnyFFTBuffer, output: FFTComplexBuffer) {
        assert(input.dataType == descriptor.inputType, "Fast Fourier Transform expected input buffer of type \(descriptor.inputType) but got \(input.dataType)")
        assert(input.fitsGrid(descriptor.gridDimensions), "Fast Fourier Transform had input buffer that was too small (element capacity: \(input.capacity)")
        assert(output.fitsGrid(descriptor.gridDimensions), "Fast Fourier Transform had output buffer that was too small (element capacity: \(output.capacity)")
        
        let desc = self.descriptor
        let resources = self.resources
        let gridDimensions = desc.gridDimensions
        
        precondition(range.count >= resources.maxCommandCount, "Needed to reserve at least \(resources.maxCommandCount) commands specified by a FastFourierTransform's maxCommandCount, but instead reserved \(range.count)).")
        
        indirectCommandBuffer.reset(range)
        var commandID = Int(range.startIndex) - 1
        
        func makeComputeCommand() -> MTLIndirectComputeCommand {
            commandID += 1
            precondition(commandID < range.endIndex, "Internal error (Fast Fourier Transform in indirect command buffer): attempted to access a compute command outside the allotted range")
            
            let computeCommand = indirectCommandBuffer.indirectComputeCommandAt(commandID)
            computeCommand.setBarrier()
            return computeCommand
        }
        
        func encodeCopy(pipeline: MTLComputePipelineState, input: AnyFFTBuffer, output: FFTComplexBuffer) {
            let computeCommand = makeComputeCommand()
            assert(pipeline.supportIndirectCommandBuffers)
            computeCommand.setComputePipelineState(pipeline)
            
            if let complexBuffer = input as? FFTComplexBuffer {
                computeCommand.setKernelBuffer(complexBuffer.buffer, offset: complexBuffer.byteOffset, at: 0)
                
            } else if let realBuffer = input as? FFTRealBuffer {
                computeCommand.setKernelBuffer(realBuffer.buffer, offset: realBuffer.byteOffset, at: 0)
                   
            } else {
                let splitBuffer = input as! FFTSplitComplexBuffer
                computeCommand.setKernelBuffer(splitBuffer.realBuffer, offset: splitBuffer.realByteOffset, at: 0)
                computeCommand.setKernelBuffer(splitBuffer.imaginaryBuffer, offset: splitBuffer.imaginaryByteOffset, at: 1)
            }
            
            computeCommand.setKernelBuffer(output.buffer, offset: output.byteOffset, at: 2)
            
            let numElements = gridDimensions.width * gridDimensions.height * gridDimensions.depth
            computeCommand.concurrentDispatchThreadgroups([ numElements ], threadsPerThreadgroup: 1)
        }
        
        // Return early in an edge case
        
        guard !isNullFFT(gridDimensions: gridDimensions, dimensionality: desc.dimensionality) else {
            debugLabel {
                print("Warning: Encoded empty Fast Fourier Transform (grid size was 1 in each dimension), which is equivalent to copying the input to the output")
            }
            
            encodeCopy(pipeline: resources.convertPipelines[0]!, input: input, output: output)
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
        
        func encodeIntro(fftDirection: Int) {
            guard let introPipeline = resources.introPipelines[fftDirection] else {
                return
            }
            
            defer {
                introSourceBuffer = resources.scratchBuffer
            }
            
            if let temp = introSourceBuffer as? FFTComplexBuffer?,
               let introSourceBuffer = temp,
               introSourceBuffer.buffer === resources.scratchBuffer?.buffer {
                let convertPipeline = resources.convertPipelines[fftDirection]!
                encodeCopy(pipeline: convertPipeline, input: output, output: introSourceBuffer)
            }
            
            let computeCommand = makeComputeCommand()
            assert(introPipeline.supportIndirectCommandBuffers)
            computeCommand.setComputePipelineState(introPipeline)
            
            let introArgsOffset = fftDirection * safeArgsStride(MemoryLayout<IntroArgs>.stride)
            resources.argsBuffer.bindIntro(to: computeCommand, offset: introArgsOffset, at: 0)
            computeCommand.setKernelBuffer(output.buffer, offset: output.byteOffset, at: 2)
            
            if let complexBuffer = introSourceBuffer! as? FFTComplexBuffer {
                computeCommand.setKernelBuffer(complexBuffer.buffer, offset: complexBuffer.byteOffset, at: 1)
                
            } else if let realBuffer = introSourceBuffer! as? FFTRealBuffer {
                computeCommand.setKernelBuffer(realBuffer.buffer, offset: realBuffer.byteOffset, at: 1)
                
            } else {
                let splitBuffer = introSourceBuffer as! FFTSplitComplexBuffer
                computeCommand.setKernelBuffer(splitBuffer.realBuffer, offset: splitBuffer.realByteOffset, at: 1)
                computeCommand.setKernelBuffer(splitBuffer.imaginaryBuffer, offset: splitBuffer.imaginaryByteOffset, at: 3)
            }
            
            let dispatchDimensions = getDispatchDimensions(fftDirection: fftDirection)
            computeCommand.concurrentDispatchThreadgroups(dispatchDimensions, threadsPerThreadgroup: 1)
        }
        
        // Encode body pass
        
        var bodyArgsOffset = 0
        
        func encodeBody(fftDirection: Int, levels: ClosedRange<Int>) {
            let bodyPipeline = resources.bodyPipelines[fftDirection]!
            let dispatchDimensions = getDispatchDimensions(fftDirection: fftDirection)
            
            for _ in levels {
                let computeCommand = makeComputeCommand()
                assert(bodyPipeline.supportIndirectCommandBuffers)
                computeCommand.setComputePipelineState(bodyPipeline)
                
                resources.argsBuffer.bindBody(to: computeCommand, offset: bodyArgsOffset, at: 0)
                resources.twiddleFactorBuffer.bind(to: computeCommand, at: 1)
                computeCommand.setKernelBuffer(output.buffer, offset: output.byteOffset, at: 2)
                computeCommand.concurrentDispatchThreadgroups(dispatchDimensions, threadsPerThreadgroup: 1)
                
                bodyArgsOffset += safeArgsStride(MemoryLayout<BodyArgs>.stride)
            }
        }
        
        let gridDimensionsVector = simd_long3(gridDimensions.width, gridDimensions.height, gridDimensions.depth)
        
        for fftDirection in 0..<desc.dimensionality {
            let dimension = gridDimensionsVector[fftDirection]
            
            encodeIntro(fftDirection: fftDirection)
            
            if dimension >= 4 {
                encodeBody(fftDirection: fftDirection, levels: 2...dimension.trailingZeroBitCount)
            }
        }
    }
    
}

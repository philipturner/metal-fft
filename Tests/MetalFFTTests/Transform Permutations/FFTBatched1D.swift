//
//  FFTBatched1D.swift
//  MetalFFTTests
//
//  Created by Philip Turner on 12/13/21.
//

import XCTest
@testable import MetalFFT
import ARHeadsetUtil

extension MetalFFTTests {
    func testRandomBatched1D(width: Int, batchSize: Int, isInverse: Bool, usingICB: Bool = false) throws {
        // Get CPU results
        
        let device = MTLCreateSystemDefaultDevice()!
        let desc = TwiddleFactorBufferDescriptor(deviceRegistryID: device.registryID,
                                                 fftSize: width,
                                                 isInverse: isInverse)
        
        if usingICB {
            guard canUseICBs(device: device) else {
                return
            }
        }
        
        let fftInputBatched1D = makeRandomFFTInput2D(width: width, height: batchSize)
        let cpuOutput = desc.fftBatched1D(fftInputBatched1D)
        
        // Get GPU results
        
        let numElements = width * batchSize
        let input = FFTComplexBuffer(device: device, capacity: numElements)
        fillFFTData(buffer: input, complexData: fftInputBatched1D)
        
        let commandQueue = device.makeCommandQueue()!
        let commandBuffer = commandQueue.makeDebugCommandBuffer()
        
        #if os(macOS) // copy data to GPU (when not on Apple silicon)
        if input.buffer.storageMode == .managed {
            commandBuffer.synchronize(input.buffer)
        }
        #endif
        
        // Create kernel
        
        let output = FFTComplexBuffer(device: device, capacity: numElements)
        var fft: FastFourierTransform
        
        if isInverse {
            fft = InverseFastFourierTransform1D(device: device,
                                                transformWidth: width,
                                                inputType: .interleavedComplex,
                                                batchSize: batchSize)
        } else {
            fft = FastFourierTransform1D(device: device,
                                         transformWidth: width,
                                         inputType: .interleavedComplex,
                                         batchSize: batchSize)
        }
        
        if usingICB {
            fft.encodeAsICB(commandBuffer: commandBuffer, input: input, output: output)
        } else {
            fft.encode(commandBuffer: commandBuffer, input: input, output: output)
        }
        
        #if os(macOS) // copy data to CPU (when not on Apple silicon)
        if output.buffer.storageMode == .managed {
            commandBuffer.synchronize(output.buffer)
        }
        #endif
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Compare both results
        
        try compareTestResults(gpuResults: output, cpuResults: cpuOutput)
    }
}

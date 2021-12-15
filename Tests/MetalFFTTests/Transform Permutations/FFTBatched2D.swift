//
//  FFTBatched2D.swift
//  MetalFFTTests
//
//  Created by Philip Turner on 12/13/21.
//

import XCTest
@testable import MetalFFT
import ARHeadsetUtil

extension MetalFFTTests {
    func testRandomBatched2D(width: Int, height: Int, batchSize: Int, isInverse: Bool, usingICB: Bool = false) throws {
        // Get CPU results
        
        let device = MTLCreateSystemDefaultDevice()!
        let desc = TwiddleFactorBufferDescriptor(deviceRegistryID: device.registryID,
                                                 fftSize: max(width, height),
                                                 isInverse: isInverse)
        
        if usingICB {
            guard canUseICBs(device: device) else {
                return
            }
        }
        
        let fftInputBatched2D = makeRandomFFTInput3D(width: width, height: height, depth: batchSize)
        let cpuOutput = desc.fftBatched2D(fftInputBatched2D)
        
        // Get GPU results
        
        let numElements = width * height * batchSize
        let input = FFTComplexBuffer(device: device, capacity: numElements)
        fillFFTData(buffer: input, complexData: fftInputBatched2D)
        
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
            fft = InverseFastFourierTransform2D(device: device,
                                                transformWidth: width,
                                                transformHeight: height,
                                                inputType: .interleavedComplex,
                                                batchSize: batchSize)
        } else {
            fft = FastFourierTransform2D(device: device,
                                         transformWidth: width,
                                         transformHeight: height,
                                         inputType: .interleavedComplex,
                                         batchSize: batchSize)
        }
        
        if usingICB {
            fft.encodeAsICB(commandBuffer: commandBuffer, input: input, output: output)
        } else {
            fft.encode(commandBuffer: commandBuffer, input: input, output: output)
        }
        
        fft.encode(commandBuffer: commandBuffer, input: input, output: output)
        
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

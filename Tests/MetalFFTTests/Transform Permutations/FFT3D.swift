//
//  FFT3D.swift
//  MetalFFTTests
//
//  Created by Philip Turner on 12/13/21.
//

import XCTest
@testable import MetalFFT
import ARHeadsetUtil

extension MetalFFTTests {
    func testRandom3D(width: Int, height: Int, depth: Int, isInverse: Bool, usingICB: Bool = false) throws {
        // Get CPU results
        
        let device = MTLCreateSystemDefaultDevice()!
        let desc = TwiddleFactorBufferDescriptor(deviceRegistryID: device.registryID,
                                                 fftSize: max(width, height, depth),
                                                 isInverse: isInverse)
        
        if usingICB {
            guard canUseICBs(device: device) else {
                return
            }
        }
        
        let fftInput3D = makeRandomFFTInput3D(width: width, height: height, depth: depth)
        let cpuOutput = desc.fft3D(fftInput3D)
        
        // Get GPU results
        
        let numElements = width * height * depth
        let input = FFTComplexBuffer(device: device, capacity: numElements)
        fillFFTData(buffer: input, complexData: fftInput3D)
        
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
            fft = InverseFastFourierTransform3D(device: device,
                                                transformWidth: width,
                                                transformHeight: height,
                                                transformDepth: depth,
                                                inputType: .interleavedComplex)
        } else {
            fft = FastFourierTransform3D(device: device,
                                         transformWidth: width,
                                         transformHeight: height,
                                         transformDepth: depth,
                                         inputType: .interleavedComplex)
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

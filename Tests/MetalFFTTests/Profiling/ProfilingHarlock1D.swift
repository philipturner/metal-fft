//
//  ProfilingHarlock1D.swift
//  MetalFFTTests
//
//  Created by Philip Turner on 12/14/21.
//

import XCTest
@testable import MetalFFT
import ARHeadsetUtil

extension MetalFFTTests {
    
    func testHarlockPerformance1D() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let commandQueue = device.makeCommandQueue()!
        
        let input = FFTSplitComplexBuffer(device: device, capacity: 128 * 28672)
        let output = FFTComplexBuffer(device: device, capacity: 128 * 28672)
        
        let fft = FastFourierTransform1D(device: device,
                                         transformWidth: 128,
                                         inputType: .splitComplex,
                                         batchSize: 28672)
        
        print("execution time in milliseconds")
        
        for _ in 0..<20 {
            let commandBuffer = commandQueue.makeCommandBuffer()!

            fft.encode(commandBuffer: commandBuffer, input: input, output: output)

            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            let elapsedTime = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
            print(elapsedTime * 1e3)
        }
    }
    
}

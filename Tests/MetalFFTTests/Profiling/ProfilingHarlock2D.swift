//
//  ProfilingHarlock2D.swift
//  MetalFFTTests
//
//  Created by Philip Turner on 12/14/21.
//

import XCTest
@testable import MetalFFT
import ARHeadsetUtil

extension MetalFFTTests {
    
    // The following numbers are for an M1 Max, which has 10 TFLOPS GPU and 64 MB system-level (L3) cache
    
    // Execution time (128 * 128 * 2048) - 47 ms
    // 1D stage (bit reversal intro) - 4.5 ms
    // 1D stage (body passes) - 11 ms
    // copy to scratch buffer - 1.6 ms
    // 2D stage (bit reversal intro) - 2.5 ms
    // 2D stage (body passes) - 11 to 29 ms ????????
    
    // Execution time (128 * 128 * 128) - 1.4 ms
    // 1D stage (bit reversal intro) - 0.30 ms
    // 1D stage (body passes) - 0.70 ms
    // copy to scratch buffer - 0.10 ms
    // 2D stage (bit reversal intro) - 0.08 ms
    // 2D stage (body passes) - 0.30 ms
    
    // 2048/128 = 16
    // 1.4 ms * 16 = 22.4 ms (what the full time would be if executing all the subdivided batches)
    // 22.4 ms < 47 ms
    // Thus, it is faster if split up into small batches, because then it fits into system-level cache
    // 128 * 128 * 128 * MemoryLayout<simd_float2>.stride = 16 MB < 64 MB
    // 128 * 128 * 2048 * MemoryLayout<simd_float2>.stride = 256 MB > 64 MB
    
    // Run this test several times, activating or deactivating certain thread dispatch/blit copy commands
    // in FastFourierTransform.encode(commandBuffer:input:output)` - located in "Sources/Resources/
    // Transforms/TransformEncoding". Also force the stage loop (at the end of the function) to only
    // execute the first stage (instead of #dimensions), then single out certain commands. With only one
    // command (e.g. intro, body, blit) active, then run both iterations of the stage loop and subtract
    // the time from only testing the 1D stage. This allows you to acquire the time spent in the 2D stage
    // on a specific command.
    
    // Note that all thread dispatches executed in the body loop are treated as one combined "command"
    // in this explanation.
    
    // The numbers you acquire may vary drastically between runs. The numbers above are the outcome of
    // the 20th trial that appeared most consistently between several runs.
    
    func testHarlockPerformance2D() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let commandQueue = device.makeCommandQueue()!
        
        let input = FFTSplitComplexBuffer(device: device, capacity: 128 * 128 * 128)
        let output = FFTComplexBuffer(device: device, capacity: 128 * 128 * 128)

        let fft = FastFourierTransform2D(device: device,
                                         transformWidth: 128,
                                         transformHeight: 128,
                                         inputType: .splitComplex,
                                         batchSize: 128)

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

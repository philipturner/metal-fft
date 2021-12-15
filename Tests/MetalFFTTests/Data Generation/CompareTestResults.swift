//
//  CompareTestResults.swift
//  MetalFFTTests
//
//  Created by Philip Turner on 12/13/21.
//

import XCTest
@testable import MetalFFT
import ARHeadsetUtil

extension MetalFFTTests {
    
    func compareTestResults(gpuResults: FFTComplexBuffer, cpuResults: [simd_float2]) throws {
        let gpuPointer = (gpuResults.buffer.contents() + gpuResults.byteOffset).assumingMemoryBound(to: simd_float2.self)
        
        for i in 0..<cpuResults.count {
            let gpuValue = gpuPointer[i]
            let cpuValue = cpuResults[i]
//            print("comparing \(gpuValue) \t \(cpuValue)")
            
            XCTAssertEqual(gpuValue.x, cpuValue.x, accuracy: 0.001)
            XCTAssertEqual(gpuValue.y, cpuValue.y, accuracy: 0.001)
        }
        
//        print("All \(cpuResults.count) elements of the Fast Fourier Transform matched the CPU")
    }
    
    func compareTestResults(gpuResults: FFTComplexBuffer, cpuResults: [[simd_float2]]) throws {
        var gpuPointer = (gpuResults.buffer.contents() + gpuResults.byteOffset).assumingMemoryBound(to: simd_float2.self)
        let jEnd = cpuResults[0].count
        
        for i in 0..<cpuResults.count {
            for j in 0..<jEnd {
                let gpuValue = gpuPointer[j]
                let cpuValue = cpuResults[i][j]
//                print("comparing \(gpuValue) \t \(cpuValue)")
                
                XCTAssertEqual(gpuValue.x, cpuValue.x, accuracy: 0.001)
                XCTAssertEqual(gpuValue.y, cpuValue.y, accuracy: 0.001)
            }
            
            gpuPointer += jEnd
        }
        
//        print("All \(cpuResults.count * jEnd) elements of the Fast Fourier Transform matched the CPU")
    }
    
    func compareTestResults(gpuResults: FFTComplexBuffer, cpuResults: [[[simd_float2]]]) throws {
        var gpuPointer = (gpuResults.buffer.contents() + gpuResults.byteOffset).assumingMemoryBound(to: simd_float2.self)
        let jEnd = cpuResults[0].count
        let kEnd = cpuResults[0][0].count
        
        for i in 0..<cpuResults.count {
            for j in 0..<jEnd {
                for k in 0..<kEnd {
                    let gpuValue = gpuPointer[k]
                    let cpuValue = cpuResults[i][j][k]
//                    print("comparing \(gpuValue) \t \(cpuValue)")
                    
                    XCTAssertEqual(gpuValue.x, cpuValue.x, accuracy: 0.001)
                    XCTAssertEqual(gpuValue.y, cpuValue.y, accuracy: 0.001)
                }
                
                gpuPointer += kEnd
            }
        }
        
//        print("All \(cpuResults.count * jEnd * kEnd) elements of the Fast Fourier Transform matched the CPU")
    }
    
}

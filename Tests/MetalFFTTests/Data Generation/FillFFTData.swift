//
//  FillFFTData.swift
//  MetalFFTTests
//
//  Created by Philip Turner on 12/13/21.
//

import XCTest
@testable import MetalFFT
import ARHeadsetUtil

extension MetalFFTTests {
    
    // Real buffer
    
    func fillFFTData(buffer gpuData: FFTRealBuffer, realData cpuData: [Float]) {
        let gpuPointer = (gpuData.buffer.contents() + gpuData.byteOffset).assumingMemoryBound(to: Float.self)
        
        for i in 0..<cpuData.count {
            gpuPointer[i] = cpuData[i]
        }
    }
    
    func fillFFTData(buffer gpuData: FFTRealBuffer, realData cpuData: [[Float]]) {
        var gpuPointer = (gpuData.buffer.contents() + gpuData.byteOffset).assumingMemoryBound(to: Float.self)
        let jEnd = cpuData[0].count
        
        for i in 0..<cpuData.count {
            for j in 0..<jEnd {
                gpuPointer[j] = cpuData[i][j]
            }
            
            gpuPointer += jEnd
        }
    }
    
    func fillFFTData(buffer gpuData: FFTRealBuffer, realData cpuData: [[[Float]]]) {
        var gpuPointer = (gpuData.buffer.contents() + gpuData.byteOffset).assumingMemoryBound(to: Float.self)
        let jEnd = cpuData[0].count
        let kEnd = cpuData[0][0].count
        
        for i in 0..<cpuData.count {
            for j in 0..<jEnd {
                for k in 0..<kEnd {
                    gpuPointer[k] = cpuData[i][j][k]
                }
                
                gpuPointer += kEnd
            }
        }
    }
    
    // Complex buffer
    
    func fillFFTData(buffer gpuData: FFTComplexBuffer, complexData cpuData: [simd_float2]) {
        let gpuPointer = (gpuData.buffer.contents() + gpuData.byteOffset).assumingMemoryBound(to: simd_float2.self)
        
        for i in 0..<cpuData.count {
            gpuPointer[i] = cpuData[i]
        }
    }
    
    func fillFFTData(buffer gpuData: FFTComplexBuffer, complexData cpuData: [[simd_float2]]) {
        var gpuPointer = (gpuData.buffer.contents() + gpuData.byteOffset).assumingMemoryBound(to: simd_float2.self)
        let jEnd = cpuData[0].count
        
        for i in 0..<cpuData.count {
            for j in 0..<jEnd {
                gpuPointer[j] = cpuData[i][j]
            }
            
            gpuPointer += jEnd
        }
    }
    
    func fillFFTData(buffer gpuData: FFTComplexBuffer, complexData cpuData: [[[simd_float2]]]) {
        var gpuPointer = (gpuData.buffer.contents() + gpuData.byteOffset).assumingMemoryBound(to: simd_float2.self)
        let jEnd = cpuData[0].count
        let kEnd = cpuData[0][0].count
        
        for i in 0..<cpuData.count {
            for j in 0..<jEnd {
                for k in 0..<kEnd {
                    gpuPointer[k] = cpuData[i][j][k]
                }
                
                gpuPointer += kEnd
            }
        }
    }
    
    // Split complex buffer
    
    func fillFFTData(buffer gpuData: FFTSplitComplexBuffer, splitRealData cpuRealData: [Float], splitImaginaryData cpuImaginaryData: [Float]) {
        let gpuRealPointer = (gpuData.realBuffer.contents() + gpuData.realByteOffset).assumingMemoryBound(to: Float.self)
        let gpuImaginaryPointer = (gpuData.imaginaryBuffer.contents() + gpuData.imaginaryByteOffset).assumingMemoryBound(to: Float.self)
        
        for i in 0..<cpuRealData.count {
            gpuRealPointer[i] = cpuRealData[i]
            gpuImaginaryPointer[i] = cpuImaginaryData[i]
        }
    }
    
    func fillFFTData(buffer gpuData: FFTSplitComplexBuffer, splitRealData cpuRealData: [[Float]], splitImaginaryData cpuImaginaryData: [[Float]]) {
        var gpuRealPointer = (gpuData.realBuffer.contents() + gpuData.realByteOffset).assumingMemoryBound(to: Float.self)
        var gpuImaginaryPointer = (gpuData.imaginaryBuffer.contents() + gpuData.imaginaryByteOffset).assumingMemoryBound(to: Float.self)
        let jEnd = cpuRealData[0].count
        
        for i in 0..<cpuRealData.count {
            for j in 0..<jEnd {
                gpuRealPointer[j] = cpuRealData[i][j]
                gpuImaginaryPointer[j] = cpuImaginaryData[i][j]
            }
            
            gpuRealPointer += jEnd
            gpuImaginaryPointer += jEnd
        }
    }
    
    func fillFFTData(buffer gpuData: FFTSplitComplexBuffer, splitRealData cpuRealData: [[[Float]]], splitImaginaryData cpuImaginaryData: [[[Float]]]) {
        var gpuRealPointer = (gpuData.realBuffer.contents() + gpuData.realByteOffset).assumingMemoryBound(to: Float.self)
        var gpuImaginaryPointer = (gpuData.imaginaryBuffer.contents() + gpuData.imaginaryByteOffset).assumingMemoryBound(to: Float.self)
        let jEnd = cpuRealData[0].count
        let kEnd = cpuRealData[0][0].count
        
        for i in 0..<cpuRealData.count {
            for j in 0..<jEnd {
                for k in 0..<kEnd {
                    gpuRealPointer[j] = cpuRealData[i][j][k]
                    gpuImaginaryPointer[j] = cpuImaginaryData[i][j][k]
                }
                
                gpuRealPointer += jEnd
                gpuImaginaryPointer += jEnd
            }
        }
    }
    
}

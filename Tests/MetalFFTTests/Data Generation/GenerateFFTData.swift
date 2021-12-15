//
//  GenerateFFTData.swift
//  MetalFFTTests
//
//  Created by Philip Turner on 12/11/21.
//

import XCTest
@testable import MetalFFT

extension MetalFFTTests {
    
    func makePredefinedFFTInput1D() -> [SIMD2<Float>] {
        return [
            .init(   9 / 10,     8 / 10),
            .init(  -7 / 10,    15 / 10),
            .init(   0.9 / 10,  80 / 10),
            .init(  -4 / 10,    16 / 10),
            .init(   0.2 / 10,   5 / 10),
            .init(   3 / 10,    -5 / 10),
            .init(   6 / 10,    -5.6 / 10),
            .init( -25 / 10,     6.001 / 10),
        ]
    }
    
    func makePredefinedFFTInput2D() -> [[SIMD2<Float>]] {
        let fftInput1D = makePredefinedFFTInput1D()
        
        return [
            fftInput1D,
            fftInput1D,
            fftInput1D.map(-),
            fftInput1D.map(-)
        ]
    }
    
    func makePredefinedFFTInput3D() -> [[[SIMD2<Float>]]] {
        let fftInput2D = makePredefinedFFTInput2D()
        
        return [
            fftInput2D,
            fftInput2D,
            fftInput2D.map { $0.map(-) },
            fftInput2D.map { $0.map(-) }
        ]
    }
    
    func makeRandomFFTInput1D(width: Int) -> [SIMD2<Float>] {
        (0..<width).map { _ -> SIMD2<Float> in
            var output = SIMD2<Float>.random(in: 0.1...2)
            
            for i in 0..<2 {
                if Bool.random() {
                    output[i] = -output[i]
                }
            }
            
            return output
        }
    }
    
    func makeRandomFFTInput2D(width: Int, height: Int) -> [[SIMD2<Float>]] {
        (0..<height).map { _ in
            makeRandomFFTInput1D(width: width)
        }
    }
    
    func makeRandomFFTInput3D(width: Int, height: Int, depth: Int) -> [[[SIMD2<Float>]]] {
        (0..<depth).map { _ in
            makeRandomFFTInput2D(width: width, height: height)
        }
    }
    
}

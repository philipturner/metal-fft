//
//  CPUTransformValidation.swift
//  MetalFFT
//
//  Created by Philip Turner on 12/11/21.
//

import ARHeadsetUtil

extension TwiddleFactorBufferDescriptor {
    
    fileprivate func twiddleFactorArray() -> [simd_float2] {
        let size = allocationSize / MemoryLayout<simd_float2>.stride
        return Array(unsafeUninitializedCapacity: size) { pointer, count in
            count = size
            fillAllocation(pointer.baseAddress!)
        }
    }
    
    fileprivate func stage1D(A: inout [simd_float2], twiddleFactors: UnsafePointer<simd_float2>, multiplier: Float) {
        let n = A.count
        guard n > 1 else { return }
        
        A = A.bitReversedElements().map { $0 * multiplier }
        
        for s in 1...n.trailingZeroBitCount {
            let m = 1 << s
            let w = twiddleFactors + m/2
            
            for k_base in 0..<n >> s {
                let k = k_base << s
                
                for j in 0..<m/2 {
                    let u = A[k + j]
                    let t = multiplyComplex(w[j], A[k + j + m/2])
                    
                    A[k + j] = addComplex(u, t)
                    A[k + j + m/2] = subtractComplex(u, t)
                }
            }
        }
    }
    
    fileprivate func stage2D(A: inout [[simd_float2]], twiddleFactors: UnsafePointer<simd_float2>, multiplier: Float) {
        func transpose2D(_ array: [[simd_float2]], multiplier: Float) -> [[simd_float2]] {
            (0..<array[0].count).map { i in
                (0..<array.count).map { j in
                    array[j][i] * multiplier
                }
            }
        }
        
        var A_transpose = transpose2D(A, multiplier: multiplier)
        
        for i in 0..<A_transpose.count {
            stage1D(A: &A_transpose[i], twiddleFactors: twiddleFactors, multiplier: 1)
        }
        
        A = transpose2D(A_transpose, multiplier: 1)
    }
    
    fileprivate func stage3D(A: inout [[[simd_float2]]], twiddleFactors: UnsafePointer<simd_float2>, multiplier: Float) {
        func transpose3D(_ array: [[[simd_float2]]], multiplier: Float) -> [[[simd_float2]]] {
            (0..<array[0][0].count).map { i in
                (0..<array[0].count).map { j in
                    (0..<array.count).map { k in
                        array[k][j][i] * multiplier
                    }
                }
            }
        }
        
        var A_transpose = transpose3D(A, multiplier: multiplier)
        
        for i in 0..<A_transpose.count {
            for j in 0..<A_transpose[0].count {
                stage1D(A: &A_transpose[i][j], twiddleFactors: twiddleFactors, multiplier: 1)
            }
        }
        
        A = transpose3D(A_transpose, multiplier: 1)
    }
    
}

extension TwiddleFactorBufferDescriptor {
    
    func logFactors() {
        let twiddleFactors = twiddleFactorArray()
        print("Logging twiddle factor buffer with capacity of \(fftSize)")
        print()
        
        var index = 1
        
        for s in 1...fftSize.trailingZeroBitCount {
            print("Layer \(s):")
            let layerSize = (1 << s) / 2
            
            for _ in 0..<layerSize {
                print(twiddleFactors[index])
                index += 1
            }
            
            print()
        }
        
        print()
        print()
    }
    
    func fft1D(_ a: [simd_float2]) -> [simd_float2] {
        var twiddleFactors = twiddleFactorArray()
        var A = a
        
        var multiplier: Float = 1
        
        if isInverse {
            multiplier = Float(a.count)
            multiplier = recip(multiplier)
        }
        
        stage1D(A: &A, twiddleFactors: &twiddleFactors, multiplier: multiplier)
        return A
    }
    
    func fftBatched1D(_ a: [[simd_float2]]) -> [[simd_float2]] {
        var twiddleFactors = twiddleFactorArray()
        var A = a
        
        var multiplier: Float = 1
        
        if isInverse {
            multiplier = Float(a[0].count)
            multiplier = recip(multiplier)
        }
        
        for i in 0..<A.count {
            stage1D(A: &A[i], twiddleFactors: &twiddleFactors, multiplier: multiplier)
        }
        
        return A
    }
    
    func fft2D(_ a: [[simd_float2]]) -> [[simd_float2]] {
        var twiddleFactors = twiddleFactorArray()
        var A = a
        
        var multiplier: Float = 1
        
        if isInverse {
            multiplier = Float(a[0].count)
            multiplier = recip(multiplier)
        }
        
        for i in 0..<A.count {
            stage1D(A: &A[i], twiddleFactors: &twiddleFactors, multiplier: multiplier)
        }
        
        if isInverse {
            multiplier = Float(a.count)
            multiplier = recip(multiplier)
        }
        
        stage2D(A: &A, twiddleFactors: &twiddleFactors, multiplier: multiplier)
        return A
    }
    
    func fftBatched2D(_ a: [[[simd_float2]]]) -> [[[simd_float2]]] {
        var twiddleFactors = twiddleFactorArray()
        var A = a
        
        var multiplier: Float = 1
        
        if isInverse {
            multiplier = Float(a[0][0].count)
            multiplier = recip(multiplier)
        }
        
        for i in 0..<A.count {
            for j in 0..<A[0].count {
                stage1D(A: &A[i][j], twiddleFactors: &twiddleFactors, multiplier: multiplier)
            }
        }
        
        if isInverse {
            multiplier = Float(a[0].count)
            multiplier = recip(multiplier)
        }
        
        for i in 0..<A.count {
            stage2D(A: &A[i], twiddleFactors: &twiddleFactors, multiplier: multiplier)
        }
        
        return A
    }
    
    func fft3D(_ a: [[[simd_float2]]]) -> [[[simd_float2]]] {
        var twiddleFactors = twiddleFactorArray()
        var A = a
        
        var multiplier: Float = 1
        
        if isInverse {
            multiplier = Float(a[0][0].count)
            multiplier = recip(multiplier)
        }
        
        for i in 0..<A.count {
            for j in 0..<A[0].count {
                stage1D(A: &A[i][j], twiddleFactors: &twiddleFactors, multiplier: multiplier)
            }
        }
        
        if isInverse {
            multiplier = Float(a[0].count)
            multiplier = recip(multiplier)
        }
        
        for i in 0..<A.count {
            stage2D(A: &A[i], twiddleFactors: &twiddleFactors, multiplier: multiplier)
        }
        
        if isInverse {
            multiplier = Float(a.count)
            multiplier = recip(multiplier)
        }
        
        stage3D(A: &A, twiddleFactors: &twiddleFactors, multiplier: multiplier)
        return A
    }
    
}

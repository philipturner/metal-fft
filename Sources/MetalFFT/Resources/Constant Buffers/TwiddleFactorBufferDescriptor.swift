//
//  TwiddleFactorBufferDescriptor.swift
//  MetalFFT
//
//  Created by Philip Turner on 12/11/21.
//

import ARHeadsetUtil

struct TwiddleFactorCacheKey: Hashable {
    var deviceRegistryID: UInt64
    var isInverse: Bool
}

struct TwiddleFactorBufferDescriptor {
    var deviceRegistryID: UInt64
    var fftSize: Int
    var isInverse: Bool
    
    var cacheKey: TwiddleFactorCacheKey {
        .init(deviceRegistryID: deviceRegistryID, isInverse: isInverse)
    }
    
    var allocationSize: Int {
        max(fftSize, 8) * MemoryLayout<simd_float2>.stride
    }
    
    func fillAllocation(_ rawPointer: UnsafeMutableRawPointer) {
        var data = rawPointer.assumingMemoryBound(to: simd_float4.self)
        let sqrt_half = Float(0.5).squareRoot()
        
        data[0] = [.nan, .nan, 1, 0]
        
        if isInverse {
            data[1] = [1, 0, 0, 1]
            data[2] = [1, 0, sqrt_half, sqrt_half]
            data[3] = [0, 1, -sqrt_half, sqrt_half]
        } else {
            data[1] = [1, 0, 0, -1]
            data[2] = [1, 0, sqrt_half, -sqrt_half]
            data[3] = [0, -1, -sqrt_half, -sqrt_half]
        }
        
        guard fftSize > 8 else {
            return
        }
        
        let m_sign: Double = isInverse ? 1 : -1
        
        for s_minus_1 in 3..<fftSize.trailingZeroBitCount {
            let m_half = 1 << s_minus_1
            let m_fourth = m_half / 2
            let m_recip_doubled = copysign(recip(Double(m_half)), m_sign)
            
            var j: UInt32 = 0
            
            repeat {
                data += 4
                
                let j_vector = simd_uint8(0, 1, 2, 3, 4, 5, 6, 7) &+ j
                let phase_doubled = simd_double8(j_vector) * m_recip_doubled
                
                let sinval_d = sinpi(phase_doubled)
                let cosval_d = __tg_sqrt(__tg_fma(sinval_d, -sinval_d, .one))
                
                let sinval = simd_float8(sinval_d)
                var cosval = simd_float8(cosval_d)
                
                if m_half == 8 {
                    cosval.highHalf = -cosval.highHalf
                } else if j >= m_fourth {
                    cosval = -cosval
                }
                
                data[0] = .init(cosval[0], sinval[0], cosval[1], sinval[1])
                data[1] = .init(cosval[2], sinval[2], cosval[3], sinval[3])
                data[2] = .init(cosval[4], sinval[4], cosval[5], sinval[5])
                data[3] = .init(cosval[6], sinval[6], cosval[7], sinval[7])
                
                j += 8
            } while j < m_half
        }
    }
}

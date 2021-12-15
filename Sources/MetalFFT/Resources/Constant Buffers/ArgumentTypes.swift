//
//  ArgumentTypes.swift
//  MetalFFT
//
//  Created by Philip Turner on 12/11/21.
//

import MTLLayeredBufferModule

public enum FFTDataType {
    case real
    case interleavedComplex
    case splitComplex
}

struct IntroArgs {
    var xPower: UInt16
    var xyPower: UInt16
    var inTIndexBit: UInt32
    
    var normalization: Float
    
    var inIndexShift: UInt16
    var outIndexShift: UInt16
    var outTIndexBit: UInt32
}

struct BodyArgs {
    var xPower: UInt16
    var xyPower: UInt16
    var tIndexBit: UInt32
    
    var jMask: UInt32
    var twiddleFactorOffset: UInt32
    var mRecipDoubled: Float
}

enum ArgsLayer: UInt16, MTLBufferLayer {
    case introArgs
    case bodyArgs
    
    static let bufferLabel = "FFT Args Buffer"
    
    func getSize(capacity numBodyStages: Int) -> Int {
        switch self {
        case .introArgs: return 3 * safeArgsStride(MemoryLayout<IntroArgs>.stride)
        case .bodyArgs:  return numBodyStages * safeArgsStride(MemoryLayout<BodyArgs>.stride)
        }
    }
}

@inline(__always)
func safeArgsStride(_ offset: Int) -> Int {
    #if os(macOS)
    ~255 & (offset + 255)
    #else
    ~3 & (offset + 3)
    #endif
}

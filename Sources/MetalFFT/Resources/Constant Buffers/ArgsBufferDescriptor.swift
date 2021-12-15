//
//  ArgsBufferDescriptor.swift
//  MetalFFT
//
//  Created by Philip Turner on 12/12/21.
//

import MTLLayeredBufferModule

extension MTLSize: Hashable {
    public static func == (lhs: MTLSize, rhs: MTLSize) -> Bool {
        lhs.width == rhs.width &&
        lhs.height == rhs.height &&
        lhs.depth == rhs.depth
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(width)
        hasher.combine(height)
        hasher.combine(depth)
    }
}

struct ArgsBufferDescriptor: Hashable {
    var registryID: UInt64
    var gridDimensions: MTLSize
    var dimensionality: Int
    var isInverse: Bool
    
    func makeArgsBuffer(device: MTLDevice) -> MTLLayeredBuffer<ArgsLayer> {
        precondition(gridDimensions.width * gridDimensions.height * gridDimensions.depth <= 2 << 31, "FFT grid size cannot exceed 2 billion elements")
        
        // Create args buffer
        
        let xPowerTemp = gridDimensions.width.trailingZeroBitCount
        let yPowerTemp = gridDimensions.height.trailingZeroBitCount
        let zPowerTemp = gridDimensions.depth.trailingZeroBitCount
        
        var numIntroStages = 1
        var numActiveStages = xPowerTemp
        var numBodyStages = max(xPowerTemp - 1, 0)
        
        if dimensionality == 1 {
            precondition(gridDimensions.depth == 1, "Must dispatch multi-instance 1D FFT as a 2D grid, not a 3D grid")
        } else {
            if yPowerTemp > 0 {
                precondition(gridDimensions.height.nonzeroBitCount == 1, "2D FFT grid height must be a power of 2")
                numIntroStages += 1
                numActiveStages += yPowerTemp
                numBodyStages += max(yPowerTemp - 1, 0)
            }
            
            if dimensionality == 3 && zPowerTemp > 0 {
                precondition(gridDimensions.depth.nonzeroBitCount == 1, "3D FFT grid depth must be a power of 2")
                numIntroStages += 1
                numActiveStages += zPowerTemp
                numBodyStages += max(zPowerTemp - 1, 0)
            }
        }
        
        var argsBuffer: MTLLayeredBuffer<ArgsLayer>
        argsBuffer = device.makeLayeredBuffer(capacity: numBodyStages, options: [device.fftResourceOptions, .cpuCacheModeWriteCombined])
        
        // Fill args buffer
        
        var rawBodyPointer = argsBuffer[.bodyArgs]
        
        func encodeArgs(fftSize: Int, fftDirection: Int, stridePower: Int) {
            guard fftSize >= 2 else {
                return
            }
            
            var rawIntroPointer = argsBuffer[.introArgs]
            rawIntroPointer += fftDirection * safeArgsStride(MemoryLayout<IntroArgs>.stride)
            let introPointer = rawIntroPointer.assumingMemoryBound(to: IntroArgs.self)
            
            let fftSizeHalf = fftSize / 2
            let normalization = recip(Float(fftSize))
            
            let xPower = UInt16(gridDimensions.width.trailingZeroBitCount)
            let xyPower = xPower + UInt16(gridDimensions.height.trailingZeroBitCount)
            
            introPointer.pointee = IntroArgs(xPower: xPower,
                                             xyPower: xyPower,
                                             inTIndexBit: UInt32(fftSizeHalf << stridePower),
                                             
                                             normalization: normalization,
                                             
                                             inIndexShift: UInt16(32 - fftSizeHalf.trailingZeroBitCount - stridePower),
                                             outIndexShift: UInt16(stridePower + 1),
                                             outTIndexBit: UInt32(1 << stridePower))
            
            guard fftSize >= 4 else {
                return
            }
            
            var mRecipDoubled: Float = isInverse ? 0.5 : -0.5
            
            for s_minus_1 in 1..<fftSize.trailingZeroBitCount {
                defer {
                    rawBodyPointer += safeArgsStride(MemoryLayout<BodyArgs>.stride)
                    mRecipDoubled /= 2
                }
                
                let bodyPointer = rawBodyPointer.assumingMemoryBound(to: BodyArgs.self)
                let mHalf = UInt32(1) << s_minus_1
                
                bodyPointer.pointee = BodyArgs(xPower: xPower,
                                               xyPower: xyPower,
                                               tIndexBit: mHalf << stridePower,
                                               
                                               jMask: mHalf - 1,
                                               twiddleFactorOffset: mHalf,
                                               mRecipDoubled: mRecipDoubled)
            }
        }
        
        encodeArgs(fftSize: gridDimensions.width, fftDirection: 0, stridePower: 0)
        
        if dimensionality >= 2 {
            let xPower = gridDimensions.width.trailingZeroBitCount
            encodeArgs(fftSize: gridDimensions.height, fftDirection: 1, stridePower: xPower)
            
            if dimensionality == 3 {
                let xyPower = xPower + gridDimensions.height.trailingZeroBitCount
                encodeArgs(fftSize: gridDimensions.depth, fftDirection: 2, stridePower: xyPower)
            }
        }
        
        return argsBuffer
    }
    
    func logArgs(device: MTLDevice) {
        let argsBuffer = makeArgsBuffer(device: device)
        print("Logging args buffer with capacity of \(argsBuffer.capacity)")
        print()
        
        var rawIntroPointer = argsBuffer[.introArgs]
        for i in 0..<3 {
            let introPointer = rawIntroPointer.assumingMemoryBound(to: IntroArgs.self)
            print("Intro arguments instance \(i):")
            print(introPointer.pointee)
            print()
            
            rawIntroPointer += safeArgsStride(MemoryLayout<IntroArgs>.stride)
        }
        
        var rawBodyPointer = argsBuffer[.bodyArgs]
        for i in 0..<argsBuffer.capacity {
            let bodyPointer = rawBodyPointer.assumingMemoryBound(to: BodyArgs.self)
            print("Body arguments instance \(i):")
            print(bodyPointer.pointee)
            print()
            
            rawBodyPointer += safeArgsStride(MemoryLayout<BodyArgs>.stride)
        }
        
        print()
        print()
    }
}

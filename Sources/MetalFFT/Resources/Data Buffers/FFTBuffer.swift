//
//  FFTBuffer.swift
//  MetalFFT
//
//  Created by Philip Turner on 12/12/21.
//

import ARHeadsetUtil
import Metal

public protocol AnyFFTBuffer {
    var capacity: Int { get }
    var dataType: FFTDataType { get }
}

public class FFTRealBuffer: AnyFFTBuffer {
    public let buffer: MTLBuffer
    public let byteOffset: Int
    public let capacity: Int
    
    public var dataType: FFTDataType { .real }
    
    public init(device: MTLDevice, capacity: Int) {
        self.byteOffset = 0
        self.capacity = capacity
        
        let bufferSize = capacity * MemoryLayout<Float>.stride
        buffer = device.makeBuffer(length: bufferSize, options: device.fftResourceOptions)!
        debugLabel {
            buffer.label = "FFT Real Buffer (capacity: \(capacity))"
        }
    }
    
    public init(buffer: MTLBuffer, byteOffset: Int, capacity: Int) {
        self.buffer = buffer
        self.byteOffset = byteOffset
        self.capacity = capacity
        
        let remainingLength = buffer.length - byteOffset
        let numRequiredBytes = capacity * MemoryLayout<Float>.stride
        precondition(numRequiredBytes <= remainingLength, "Attempted to make an FFT Real Buffer with capacity \(capacity) (\(numRequiredBytes) bytes), but got a buffer range with \(remainingLength) available bytes")
    }
}

public class FFTComplexBuffer: AnyFFTBuffer {
    public let buffer: MTLBuffer
    public let byteOffset: Int
    public let capacity: Int
    
    public var dataType: FFTDataType { .interleavedComplex }
    
    public init(device: MTLDevice, capacity: Int) {
        self.byteOffset = 0
        self.capacity = capacity
        
        let bufferSize = capacity * MemoryLayout<simd_float2>.stride
        buffer = device.makeBuffer(length: bufferSize, options: device.fftResourceOptions)!
        debugLabel {
            buffer.label = "FFT Interleaved Complex Buffer (capacity: \(capacity))"
        }
    }
    
    public init(buffer: MTLBuffer, byteOffset: Int, capacity: Int) {
        self.buffer = buffer
        self.byteOffset = byteOffset
        self.capacity = capacity
        
        let remainingLength = buffer.length - byteOffset
        let numRequiredBytes = capacity * MemoryLayout<simd_float2>.stride
        precondition(numRequiredBytes <= remainingLength, "Attempted to make an FFT Complex Buffer with capacity \(capacity) (\(numRequiredBytes) bytes), but got a buffer range with \(remainingLength) available bytes")
    }
}

public class FFTSplitComplexBuffer: AnyFFTBuffer {
    public let realBuffer: MTLBuffer
    public let imaginaryBuffer: MTLBuffer
    
    public let realByteOffset: Int
    public let imaginaryByteOffset: Int
    public let capacity: Int
    
    public var dataType: FFTDataType { .splitComplex }
    
    public init(device: MTLDevice, capacity: Int) {
        self.realByteOffset = 0
        self.imaginaryByteOffset = 0
        self.capacity = capacity
        
        let bufferSize = capacity * MemoryLayout<Float>.stride
        realBuffer = device.makeBuffer(length: bufferSize, options: device.fftResourceOptions)!
        imaginaryBuffer = device.makeBuffer(length: bufferSize, options: device.fftResourceOptions)!
        debugLabel {
            realBuffer.label = "FFT Split Complex Buffer (real part, capacity: \(capacity))"
            imaginaryBuffer.label = "FFT Split Complex Buffer (imaginary part, capacity: \(capacity))"
        }
    }
    
    public init(realBuffer: MTLBuffer,
                realByteOffset: Int,
                imaginaryBuffer: MTLBuffer,
                imaginaryByteOffset: Int,
                capacity: Int) {
        self.realBuffer = realBuffer
        self.imaginaryBuffer = imaginaryBuffer
        
        self.realByteOffset = realByteOffset
        self.imaginaryByteOffset = imaginaryByteOffset
        self.capacity = capacity
        
        let remainingRealLength = realBuffer.length - realByteOffset
        let remainingImaginaryLength = imaginaryBuffer.length - imaginaryByteOffset
        let numRequiredBytes = capacity * MemoryLayout<Float>.stride
        
        precondition(numRequiredBytes <= remainingRealLength, "Attempted to make an FFT Split Complex Buffer with capacity \(capacity) (\(numRequiredBytes) bytes), but got a real buffer range with \(remainingRealLength) available bytes")
        
        precondition(numRequiredBytes <= remainingImaginaryLength, "Attempted to make an FFT Split Complex Buffer with capacity \(capacity) (\(numRequiredBytes) bytes), but got an imaginary buffer range with \(remainingImaginaryLength) available bytes")
    }
}

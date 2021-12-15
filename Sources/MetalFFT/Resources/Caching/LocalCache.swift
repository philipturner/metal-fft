//
//  LocalCache.swift
//  MetalFFT
//
//  Created by Philip Turner on 12/12/21.
//

import Metal
import ARHeadsetUtil

class LocalCache {
    static var threadLocal: LocalCache {
        let key = "com.philipturner.MetalFFT.LocalCache.threadLocal"
        var cache = Thread.current.threadDictionary[key]
        
        if cache == nil {
            cache = LocalCache()
            Thread.current.threadDictionary[key] = cache
        }
        
        return cache as! LocalCache
    }
    
    fileprivate var fastFourierTransforms: [TransformDescriptor: TransformResources] = [:]
    fileprivate var complexNumberConversions: [ConversionDescriptor: MTLComputePipelineState] = [:]
    
    fileprivate var twiddleFactorBuffers: [TwiddleFactorCacheKey: TwiddleFactorBuffer] = [:]
    fileprivate var scratchBuffers: [UInt64: FFTComplexBuffer] = [:]
    
    fileprivate var serialQueue = DispatchQueue(label: "com.philipturner.MetalFFT.LocalCache.serialQueue")
    fileprivate var unsafeMessage: Any?
    
    fileprivate func getAsyncResources<Result>(_ operation: @escaping () async -> Result) -> Result {
        let semaphore = DispatchSemaphore(value: 0)
        
        Task {
            let result = await operation()
            
            serialQueue.sync {
                unsafeMessage = result
            }
            
            semaphore.signal()
        }
        
        semaphore.wait()
        
        return serialQueue.sync {
            let message = unsafeMessage!
            unsafeMessage = nil
            return message as! Result
        }
    }
    
    // Retrieve conversion kernel
    
    func makeComplexNumberConversion(device: MTLDevice, descriptor: ConversionDescriptor) -> MTLComputePipelineState {
        if let conversion = complexNumberConversions[descriptor] {
            return conversion
        }
        
        let conversion = getAsyncResources { () -> MTLComputePipelineState in
            let oldDesc = descriptor
            let newDesc = ConvertPipelineDescriptor(deviceRegistryID: oldDesc.deviceRegistryID,
                                                    inputType: oldDesc.inputType,
                                                    outputType: oldDesc.outputType)
            
            return await GlobalCache.shared.makeConvertPipeline(device: device, descriptor: newDesc)
        }
        
        complexNumberConversions[descriptor] = conversion
        return conversion
    }
    
    // Retrieve buffers
    
    fileprivate func makeArgsBuffer(device: MTLDevice, descriptor: ArgsBufferDescriptor) async -> ArgsBuffer {
        await GlobalCache.shared.makeArgsBuffer(device: device, descriptor: descriptor)
    }
    
    fileprivate func makeTwiddleFactorBuffer(device: MTLDevice, descriptor: TwiddleFactorBufferDescriptor) async -> TwiddleFactorBuffer {
        if let buffer = twiddleFactorBuffers[descriptor.cacheKey],
           buffer.allocationSize >= descriptor.allocationSize {
            return buffer
        } else {
            return await GlobalCache.shared.makeTwiddleFactorBuffer(device: device, descriptor: descriptor)
        }
    }
    
    fileprivate func makeScratchBuffer(device: MTLDevice, size: Int) async -> FFTComplexBuffer {
        let registryID = device.registryID
        if let buffer = scratchBuffers[registryID], buffer.capacity >= size {
            return buffer
        } else {
            return await GlobalCache.shared.makeScratchBuffer(device: device, size: size)
        }
    }
}

// Retrieve transform kernel

extension LocalCache {
    
    func makeFastFourierTransform(device: MTLDevice, descriptor: TransformDescriptor) -> TransformResources {
        if let transform = fastFourierTransforms[descriptor] {
            return transform
        }
        
        let transform = getAsyncResources { () -> TransformResources in
            let desc = descriptor
            let gridDimensions = desc.gridDimensions
            
            // Make args buffer
            
            let argsBufferDesc = ArgsBufferDescriptor(registryID: desc.deviceRegistryID,
                                                      gridDimensions: gridDimensions,
                                                      dimensionality: desc.dimensionality,
                                                      isInverse: desc.isInverse)
            let argsBuffer = await self.makeArgsBuffer(device: device, descriptor: argsBufferDesc)
            
            // Make twiddle factor buffer
            
            var maxFFTSize = gridDimensions.width
            if desc.dimensionality >= 2 { maxFFTSize = max(maxFFTSize, gridDimensions.height) }
            if desc.dimensionality == 3 { maxFFTSize = max(maxFFTSize, gridDimensions.depth) }
            
            let twiddleFactorBufferDesc = TwiddleFactorBufferDescriptor(deviceRegistryID: device.registryID,
                                                                        fftSize: maxFFTSize,
                                                                        isInverse: desc.isInverse)
            let twiddleFactorBuffer = await self.makeTwiddleFactorBuffer(device: device, descriptor: twiddleFactorBufferDesc)
            
            // Return early in an edge case
            
            guard !isNullFFT(gridDimensions: desc.gridDimensions, dimensionality: desc.dimensionality) else {
                let convertDesc = ConvertPipelineDescriptor(deviceRegistryID: desc.deviceRegistryID,
                                                            inputType: desc.inputType,
                                                            outputType: .interleavedComplex)
                let convertPipeline = await GlobalCache.shared.makeConvertPipeline(device: device, descriptor: convertDesc)
                
                return TransformResources(introPipelines: [],
                                          bodyPipelines: [],
                                          convertPipelines: [convertPipeline],
                                          argsBuffer: argsBuffer,
                                          twiddleFactorBuffer: twiddleFactorBuffer,
                                          scratchBuffer: nil,
                                          maxKernelBufferBindCount: 3,
                                          maxCommandCount: 1)
                
            }
            
            // Find number of dimensions in grid
            
            var gridNumDims: Int
            
            switch desc.dimensionality {
            case 1:
                if gridDimensions.depth > 1 { preconditionFailure("Must dispatch multi-instance 1D FFT as a 2D grid, not a 3D grid") }
                else if gridDimensions.height > 1 { gridNumDims = 2 }
                else { gridNumDims = 1 }
            case 2:
                gridNumDims = gridDimensions.depth > 1 ? 3 : 2
            default:
                gridNumDims = 3
            }
            
            // Load shaders
            
            var introPipelines: [MTLComputePipelineState?] = Array(repeating: nil, count: 3)
            var bodyPipelines: [MTLComputePipelineState?] = Array(repeating: nil, count: 3)
            var convertPipelines: [MTLComputePipelineState?] = Array(repeating: nil, count: 3)
            
            var maxKernelBufferBindCount: Int = 3
            var maxCommandCount: Int = 0
            
            let gridDimensionsVector = simd_long3(gridDimensions.width, gridDimensions.height, gridDimensions.depth)
            var currentNumberFormat = desc.inputType
            var didIntroPass = false
            
            var interleavedConvertPipeline: MTLComputePipelineState?
            var scratchBuffer: FFTComplexBuffer?
            
            for fftDirection in 0..<desc.dimensionality {
                let dimension = gridDimensionsVector[fftDirection]
                precondition(dimension.nonzeroBitCount == 1, "Internal error: attempted to initialize a FastFourierTransform where a transformed dimension was not a power of 2")
                
                guard dimension >= 2 else { continue }
                
                defer {
                    currentNumberFormat = .interleavedComplex
                    didIntroPass = true
                }
                
                // Intro pass
                
                maxCommandCount += 1
                
                if currentNumberFormat == .splitComplex {
                    maxKernelBufferBindCount = 4
                }
                
                let introDesc = IntroPipelineDescriptor(deviceRegistryID: desc.deviceRegistryID,
                                                        gridNumDims: gridNumDims,
                                                        direction: fftDirection,
                                                        inputType: currentNumberFormat,
                                                        isInverse: desc.isInverse)
                
                introPipelines[fftDirection] = await GlobalCache.shared.makeIntroPipeline(device: device, descriptor: introDesc)
                
                // Convert pass
                
                if didIntroPass {
                    maxCommandCount += 1
                    
                    if let interleavedConvertPipeline = interleavedConvertPipeline {
                        convertPipelines[fftDirection] = interleavedConvertPipeline
                    } else {
                        let convertDesc = ConvertPipelineDescriptor(deviceRegistryID: desc.deviceRegistryID,
                                                                    inputType: .interleavedComplex,
                                                                    outputType: .interleavedComplex)
                        
                        interleavedConvertPipeline = await GlobalCache.shared.makeConvertPipeline(device: device, descriptor: convertDesc)
                        
                        convertPipelines[fftDirection] = interleavedConvertPipeline!
                        
                        let numElements = gridDimensions.width * gridDimensions.height * gridDimensions.depth
                        scratchBuffer = await self.makeScratchBuffer(device: device, size: numElements)
                    }
                }
                
                guard dimension >= 4 else { continue }
                
                // Body pass
                
                maxCommandCount += dimension.trailingZeroBitCount - 1
                
                let bodyDesc = BodyPipelineDescriptor(deviceRegistryID: desc.deviceRegistryID,
                                                      gridNumDims: gridNumDims,
                                                      fftDirection: fftDirection,
                                                      readingTwiddleFactors: true)
                
                bodyPipelines[fftDirection] = await GlobalCache.shared.makeBodyPipeline(device: device, descriptor: bodyDesc)
            }
            
            return TransformResources(introPipelines: introPipelines,
                                      bodyPipelines: bodyPipelines,
                                      convertPipelines: convertPipelines,
                                      argsBuffer: argsBuffer,
                                      twiddleFactorBuffer: twiddleFactorBuffer,
                                      scratchBuffer: scratchBuffer,
                                      maxKernelBufferBindCount: maxKernelBufferBindCount,
                                      maxCommandCount: maxCommandCount)
        }
        
        fastFourierTransforms[descriptor] = transform
        return transform
    }
    
}

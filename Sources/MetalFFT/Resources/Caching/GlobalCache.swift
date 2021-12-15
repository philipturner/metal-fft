//
//  GlobalCache.swift
//  MetalFFT
//
//  Created by Philip Turner on 12/12/21.
//

import Metal
import ARHeadsetUtil

actor GlobalCache {
    static let shared = GlobalCache()
    
    private var libraries: [UInt64: MTLLibrary] = [:]
    #if os(macOS)
    private var commandQueues: [UInt64: MTLCommandQueue] = [:]
    #endif
    
    private var introPipelines: [IntroPipelineDescriptor: MTLComputePipelineState] = [:]
    private var bodyPipelines: [BodyPipelineDescriptor: MTLComputePipelineState] = [:]
    private var convertPipelines: [ConvertPipelineDescriptor: MTLComputePipelineState] = [:]
    
    private var argsBuffers: [ArgsBufferDescriptor: ArgsBuffer] = [:]
    private var twiddleFactorsBuffers: [TwiddleFactorCacheKey: TwiddleFactorBuffer] = [:]
    private var scratchBuffers: [UInt64: FFTComplexBuffer] = [:]
    private var numScratchBufferReallocations: [UInt64: Int] = [:]
    
    // Manage device-specific resource generators
    
    private func getResource<T>(for device: MTLDevice, cache: inout [UInt64: T], generator: (MTLDevice) -> T) -> T {
        let registryID = device.registryID
        if let resource = cache[registryID] {
            return resource
        }
        
        let resource = generator(device)
        cache[registryID] = resource
        return resource
    }
    
    private func getLibrary(for device: MTLDevice) -> MTLLibrary {
        getResource(for: device, cache: &libraries) { device in
            try! device.makeDefaultLibrary(bundle: Bundle.safeModule)
        }
    }
    
    #if os(macOS)
    private func getCommandQueue(for device: MTLDevice) -> MTLCommandQueue {
        getResource(for: device, cache: &commandQueues) { device in
            device.makeCommandQueue()!
        }
    }
    #endif
    
    // Create pipelines
    
    private func makePipeline<T: Hashable>(device: MTLDevice, descriptor: T, cache: inout [T: MTLComputePipelineState], generator: (MTLLibrary, T) -> MTLComputePipelineState) -> MTLComputePipelineState {
        if let pipeline = cache[descriptor] {
            return pipeline
        }
        
        let library = getLibrary(for: device)
        let pipeline = generator(library, descriptor)
        
        cache[descriptor] = pipeline
        return pipeline
    }
    
    func makeIntroPipeline(device: MTLDevice, descriptor: IntroPipelineDescriptor) -> MTLComputePipelineState {
        makePipeline(device: device, descriptor: descriptor, cache: &introPipelines, generator: makeIntroPipeline(library:descriptor:))
    }
    
    func makeBodyPipeline(device: MTLDevice, descriptor: BodyPipelineDescriptor) -> MTLComputePipelineState {
        makePipeline(device: device, descriptor: descriptor, cache: &bodyPipelines, generator: makeBodyPipeline(library:descriptor:))
    }
    
    func makeConvertPipeline(device: MTLDevice, descriptor: ConvertPipelineDescriptor) -> MTLComputePipelineState {
        makePipeline(device: device, descriptor: descriptor, cache: &convertPipelines, generator: makeConvertPipeline(library:descriptor:))
    }
    
    // Create buffers
    
    func makeArgsBuffer(device: MTLDevice, descriptor: ArgsBufferDescriptor) -> ArgsBuffer {
        if let buffer = argsBuffers[descriptor] {
            return buffer
        }
        
        var commandQueue: MTLCommandQueue?
        #if os(macOS)
        if device.fftResourceOptions == .storageModeManaged {
            commandQueue = getCommandQueue(for: device)
        }
        #endif
        
        let buffer = ArgsBuffer(device: device, descriptor: descriptor, commandQueue: commandQueue)
        argsBuffers[descriptor] = buffer
        return buffer
    }
    
    func makeTwiddleFactorBuffer(device: MTLDevice, descriptor: TwiddleFactorBufferDescriptor) -> TwiddleFactorBuffer {
        if let buffer = twiddleFactorsBuffers[descriptor.cacheKey],
           buffer.allocationSize >= descriptor.allocationSize {
            return buffer
        }
        
        var commandQueue: MTLCommandQueue?
        #if os(macOS)
        if device.fftResourceOptions == .storageModeManaged {
            commandQueue = getCommandQueue(for: device)
        }
        #endif
        
        let buffer = TwiddleFactorBuffer(device: device, descriptor: descriptor, commandQueue: commandQueue)
        twiddleFactorsBuffers[descriptor.cacheKey] = buffer
        return buffer
    }
    
    func makeScratchBuffer(device: MTLDevice, size: Int) -> FFTComplexBuffer {
        let registryID = device.registryID
        let oldBuffer = scratchBuffers[registryID]
        
        if let buffer = oldBuffer, buffer.capacity >= size {
            return buffer
        }
        
        var numReallocations = numScratchBufferReallocations[registryID] ?? 0
        if oldBuffer != nil {
            numReallocations += 1
            numScratchBufferReallocations[registryID] = numReallocations
        }
        
        var capacity = numReallocations >= 8 ? roundUpToPowerOf2(size) : size
        capacity = max(capacity, 16384 / MemoryLayout<simd_float2>.stride)
        
        let newBuffer = FFTComplexBuffer(device: device, capacity: capacity)
        
        scratchBuffers[registryID] = newBuffer
        return newBuffer
    }
}

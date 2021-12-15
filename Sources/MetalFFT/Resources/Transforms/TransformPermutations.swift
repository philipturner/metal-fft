//
//  TransformPermutations.swift
//  MetalFFT
//
//  Created by Philip Turner on 12/12/21.
//

import Metal

// One-dimensional transforms

public class FastFourierTransform1D: FastFourierTransform {
    public init(device: MTLDevice,
                transformWidth: Int,
                inputType: FFTDataType,
                batchSize: Int = 1) {
        assert(transformWidth.nonzeroBitCount == 1, "FFT 1D transformWidth was not a power of 2")
        
        let descriptor = TransformDescriptor(deviceRegistryID: device.registryID,
                                             gridDimensions: [transformWidth, batchSize],
                                             dimensionality: 1,
                                             inputType: inputType,
                                             isInverse: false)
        
        let resources = descriptor.makeResources(device: device)
        super.init(descriptor: descriptor, resources: resources)
    }
}

public class InverseFastFourierTransform1D: FastFourierTransform {
    public init(device: MTLDevice,
                transformWidth: Int,
                inputType: FFTDataType,
                batchSize: Int = 1) {
        assert(transformWidth.nonzeroBitCount == 1, "Inverse FFT 1D transformWidth was not a power of 2")
        
        let descriptor = TransformDescriptor(deviceRegistryID: device.registryID,
                                             gridDimensions: [transformWidth, batchSize],
                                             dimensionality: 1,
                                             inputType: inputType,
                                             isInverse: true)
        
        let resources = descriptor.makeResources(device: device)
        super.init(descriptor: descriptor, resources: resources)
    }
}

// Two-dimensional transforms

public class FastFourierTransform2D: FastFourierTransform {
    public init(device: MTLDevice,
                transformWidth: Int,
                transformHeight: Int,
                inputType: FFTDataType,
                batchSize: Int = 1) {
        assert(transformWidth.nonzeroBitCount == 1, "FFT 2D transformWidth was not a power of 2")
        assert(transformHeight.nonzeroBitCount == 1, "FFT 2D transformHeight was not a power of 2")
        
        let descriptor = TransformDescriptor(deviceRegistryID: device.registryID,
                                             gridDimensions: [transformWidth, transformHeight, batchSize],
                                             dimensionality: 2,
                                             inputType: inputType,
                                             isInverse: false)
        
        let resources = descriptor.makeResources(device: device)
        super.init(descriptor: descriptor, resources: resources)
    }
}

public class InverseFastFourierTransform2D: FastFourierTransform {
    public init(device: MTLDevice,
                transformWidth: Int,
                transformHeight: Int,
                inputType: FFTDataType,
                batchSize: Int = 1) {
        assert(transformWidth.nonzeroBitCount == 1, "Inverse FFT 2D transformWidth was not a power of 2")
        assert(transformHeight.nonzeroBitCount == 1, "Inverse FFT 2D transformHeight was not a power of 2")
        
        let descriptor = TransformDescriptor(deviceRegistryID: device.registryID,
                                             gridDimensions: [transformWidth, transformHeight, batchSize],
                                             dimensionality: 2,
                                             inputType: inputType,
                                             isInverse: true)
        
        let resources = descriptor.makeResources(device: device)
        super.init(descriptor: descriptor, resources: resources)
    }
}

// Three-dimensional transforms

public class FastFourierTransform3D: FastFourierTransform {
    public init(device: MTLDevice,
                transformWidth: Int,
                transformHeight: Int,
                transformDepth: Int,
                inputType: FFTDataType) {
        assert(transformWidth.nonzeroBitCount == 1, "FFT 3D transformWidth was not a power of 2")
        assert(transformHeight.nonzeroBitCount == 1, "FFT 3D transformHeight was not a power of 2")
        assert(transformDepth.nonzeroBitCount == 1, "FFT 3D transformDepth was not a power of 2")
        
        let descriptor = TransformDescriptor(deviceRegistryID: device.registryID,
                                             gridDimensions: [transformWidth, transformHeight, transformDepth],
                                             dimensionality: 3,
                                             inputType: inputType,
                                             isInverse: false)
        
        let resources = descriptor.makeResources(device: device)
        super.init(descriptor: descriptor, resources: resources)
    }
}

public class InverseFastFourierTransform3D: FastFourierTransform {
    public init(device: MTLDevice,
                transformWidth: Int,
                transformHeight: Int,
                transformDepth: Int,
                inputType: FFTDataType) {
        assert(transformWidth.nonzeroBitCount == 1, "Inverse FFT 3D transformWidth was not a power of 2")
        assert(transformHeight.nonzeroBitCount == 1, "Inverse FFT 3D transformHeight was not a power of 2")
        assert(transformDepth.nonzeroBitCount == 1, "Inverse FFT 3D transformDepth was not a power of 2")
        
        let descriptor = TransformDescriptor(deviceRegistryID: device.registryID,
                                             gridDimensions: [transformWidth, transformHeight, transformDepth],
                                             dimensionality: 3,
                                             inputType: inputType,
                                             isInverse: true)
        
        let resources = descriptor.makeResources(device: device)
        super.init(descriptor: descriptor, resources: resources)
    }
}

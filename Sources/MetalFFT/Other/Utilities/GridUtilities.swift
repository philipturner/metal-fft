//
//  GridUtilities.swift
//  
//
//  Created by Philip Turner on 12/13/21.
//

import ARHeadsetUtil
import Metal

internal extension AnyFFTBuffer {
    func fitsGrid(_ gridDimensions: MTLSize) -> Bool {
        let numElements = gridDimensions.width * gridDimensions.height * gridDimensions.depth
        return numElements <= capacity
    }
}

func isNullFFT(gridDimensions: MTLSize, dimensionality: Int) -> Bool {
    // Check 1D params
    
    if gridDimensions.width != 1 {
        return false
    }
    
    if dimensionality == 1 {
        return true
    }
    
    // Check 2D params
    
    if gridDimensions.height != 1 {
        return false
    }
    
    if dimensionality == 2 {
        return true
    }
    
    // Check 3D params
    
    if gridDimensions.depth != 1 {
        return false
    }
    
    return true
}

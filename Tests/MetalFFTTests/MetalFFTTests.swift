import XCTest
@testable import MetalFFT
import ARHeadsetUtil

final class MetalFFTTests: XCTestCase {
    func testAll() throws {
        print("===== Testing on Metal device: \(MTLCreateSystemDefaultDevice()!.name) =====")
        
        try testAllRandom1D()
        try testAllRandom2D()
        try testAllRandom3D()
    }
    
    func testAllRandom1D() throws {
        func testRange(_ range: [Int]) throws {
            for width in range {
                try testRandom1D(width: width, isInverse: false)
                try testRandom1D(width: width, isInverse: true)
                
                try testRandom1D(width: width, isInverse: false, usingICB: true)
                try testRandom1D(width: width, isInverse: true, usingICB: true)
            }
        }
        
        try testRange([1, 2, 4])
        try testRange([8, 16, 32])
        try testRange([128, 1024, 4096])
    }
    
    func testAllRandom2D() throws {
        func testRange(_ range: [Int]) throws {
            for width in range {
                for height in range {
                    try testRandomBatched1D(width: width, batchSize: height, isInverse: false)
                    try testRandomBatched1D(width: width, batchSize: height, isInverse: true)
                    
                    try testRandomBatched1D(width: width, batchSize: height, isInverse: false, usingICB: true)
                    try testRandomBatched1D(width: width, batchSize: height, isInverse: true, usingICB: true)
                    
                    try testRandom2D(width: width, height: height, isInverse: false)
                    try testRandom2D(width: width, height: height, isInverse: true)
                    
                    try testRandom2D(width: width, height: height, isInverse: false, usingICB: true)
                    try testRandom2D(width: width, height: height, isInverse: true, usingICB: true)
                }
            }
        }
        
        try testRange([1, 2, 4])
        try testRange([8, 16, 32])
    }
    
    func testAllRandom3D() throws {
        func testRange(_ range: [Int]) throws {
            for width in range {
                for height in range {
                    for depth in range {
                        try testRandomBatched2D(width: width, height: height, batchSize: depth, isInverse: false)
                        try testRandomBatched2D(width: width, height: height, batchSize: depth, isInverse: true)
                        
                        try testRandomBatched2D(width: width, height: height, batchSize: depth, isInverse: false, usingICB: true)
                        try testRandomBatched2D(width: width, height: height, batchSize: depth, isInverse: true, usingICB: true)
                        
                        try testRandom3D(width: width, height: height, depth: depth, isInverse: false)
                        try testRandom3D(width: width, height: height, depth: depth, isInverse: true)
                        
                        try testRandom3D(width: width, height: height, depth: depth, isInverse: false, usingICB: true)
                        try testRandom3D(width: width, height: height, depth: depth, isInverse: true, usingICB: true)
                    }
                }
            }
        }
        
        try testRange([1, 2, 4])
        try testRange([8, 16])
    }
}

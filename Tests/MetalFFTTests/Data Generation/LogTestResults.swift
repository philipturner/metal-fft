//
//  LogTestResults.swift
//  MetalFFTTests
//
//  Created by Philip Turner on 12/11/21.
//

import XCTest
@testable import MetalFFT

extension MetalFFTTests {
    
    func logResults<T>(_ a: [T]) {
        print()
        print("If reading twiddle factors, should be:")
        for element in a {
            print(element)
        }
    }
    
    func logResults<T>(_ a: [[T]]) {
        print()
        print("If reading twiddle factors, should be:")

        for i in 0..<a.count {
            print("Row or Instance \(i):")
            
            for element in a[i] {
                print(element)
            }
        }
    }
    
    func logResults<T>(_ a: [[[T]]]) {
        print()
        print("If reading twiddle factors, should be:")

        for i in 0..<a.count {
            print()
            print("Slice or Instance \(i):")
            
            for j in 0..<a[0].count {
                print("Row \(j):")
                
                for element in a[i][j] {
                    print(element)
                }
            }
        }
    }
    
}

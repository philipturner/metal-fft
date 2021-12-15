// swift-tools-version:5.5
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MetalFFT",
    platforms: [
        .iOS(.v15),
        .macOS(.v12),
        .tvOS(.v15),
    ],
    products: [
        .library(
            name: "MetalFFT",
            targets: ["MetalFFT"]),
    ],
    dependencies: [
        .package(name: "MTLLayeredBufferModule", url: "https://github.com/philipturner/MTLLayeredBuffer", branch: "main"),
    ],
    targets: [
        .target(
            name: "MetalFFT",
            dependencies: ["MTLLayeredBufferModule"],
            resources: [
                .process("Other/Shaders/FFTBody.metal"),
                .process("Other/Shaders/FFTConvert.metal"),
                .process("Other/Shaders/FFTIntro.metal"),
            ],
            swiftSettings: [
                .unsafeFlags(["-enable-testing"]),
            ]),
        .testTarget(
            name: "MetalFFTTests",
            dependencies: ["MetalFFT"]),
    ]
)

# MetalFFT

MetalFFT is an experiment in adding GPU acceleration for 1D, 2D, and 3D variations of Fast Fourier Transforms. 

This framework's original purpose was to become a raw operator for a [Swift for TensorFlow](https://github.com/tensorflow/swift) Metal backend. Work on it paused due to unfavorable performance. For many transforms, Apple's CPU alternative from Accelerate runs faster. The A13 and later have AMX accelerators capable of ~2 TFLOPS, and the CPU implementation may harness that processing power.

MetalFFT was also created to show Apple's Metal Performance Shaders (MPS) team how to support encoding MPS kernels into indirect command buffers. Due to the lack of a direct connection to machine learning, this will not be merged into a larger repository when constructing an S4TF Metal backend. For more context on the backend, see the [Differentiation iOS Demo](https://github.com/philipturner/differentiation-ios-demo).

## How to use

MetalFFT's API is similar to [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders). It only performs FFTs on tensors whose dimensions are powers of two. One may run a batch of several 1D or 2D FFTs simultaneously, as long as the tensors are arranged in contiguous blocks of memory. The batch size does not need to be a power of two, although it has an upper bound: a singular transform's dimensions multiplied by the batch's size must not exceed 2^31 (about 2 billion).

### Executing a 1D Fast Fourier Transform:

```swift
import MetalFFT

// Allocate GPU-accessible buffers
let input = FFTRealBuffer(device: device, capacity: 128 * batchSize)
let output = FFTComplexBuffer(device: device, capacity: 128 * batchSize)

// Fill the time-domain data
let timeDomainData = input.buffer.contents()
let audioSamples: UnsafeMutablePointer<Float> = ...
memcpy(timeDomainData, audioSamples, audioSampleSizeInBytes)

let commandBuffer = commandQueue.makeCommandBuffer()!
#if os(macOS)
if input.buffer.storageMode == .managed { // `.shared` on M1
    // Synchronize `input.buffer` between CPU and GPU
}
#endif

let fft = FastFourierTransform1D(device: device,
                                 transformWidth: 128,
                                 inputType: .real,
                                 batchSize: batchSize)
fft.encode(commandBuffer: commandBuffer, input: input, output: output)

#if os(macOS)
if output.buffer.storageMode == .managed { // `.shared` on M1
    // Synchronize `output.buffer` between CPU and GPU
}
#endif
commandBuffer.commit()

// Extract frequency-domain data
commandBuffer.waitUntilCompleted()
let frequencyDomainData = output.buffer.contents()
```

All numbers processed by FFT kernels are single-precision floats. You may use either real, interleaved complex, or split complex numbers as input. The output is interleaved complex to optimize memory accesses during the kernel's execution. Use a `ComplexNumberConversion` to convert the output from complex to real or split complex.

### Converting complex numbers into split form:

```swift
let interleavedComplexData: FFTComplexBuffer = ...
let splitComplexData: FFTSplitComplexBuffer = ...

let conversion = ComplexNumberConversion(device: device,
                                         inputType: .interleavedComplex,
                                         outputType: .splitComplex)

let commandBuffer = commandQueue.makeCommandBuffer()!
conversion.encode(commandBuffer: commandBuffer,
                  input: interleavedComplexData,
                  output: splitComplexData)
commandBuffer.commit()

// Extract real and imaginary GPU buffers
let realData = splitComplexData.realBuffer
let imaginaryData = splitComplexData.imaginaryBuffer
```

## How it works

FFT kernels operate in-place on their output buffer, using bit reversal pre-processing and decimation in time. The first stage is out-of-place and eliminates one copying pass. It reads from the input using bit reversal, performs 2-wide Discrete Fourier Transforms (butterflies), then writes to the output. The other stages proceed according to the `iterative-fft` pseudocode on the Cooley-Tukey algorithm [wikipedia article](https://en.wikipedia.org/wiki/Cooley-Tukey_FFT_algorithm). Twiddle factors are cached into a constant buffer so that the shader can avoid recalculating them.

The 2D and 3D variations repeat the steps above, once for each dimension. Between repetitions, the shader pages to a scratch buffer since bit reversal indexing cannot be done in-place. MetalFFT allocates and manages the scratch buffer internally.

The `iterative-fft` approach was selected because it can be easily parallelized, unlike the recursive approach often used on CPUs. This could be improved with a mixed-radix algorithm, executing 4-wide or 8-wide butterflies inside the shader. That approach reduces the number of memory accesses while retaining the same number of floating-point operations. However, it slightly complicates command encoding on the CPU.

## Future work

Hopefully, someone in the open source community or Apple's Metal Performance Shaders (MPS) team will pick up where I left off. MetalFFT still requires several optimizations, including mixed radices and automatically splitting large batches into smaller ones ([to improve cache coherence](Tests/MetalFFTTests/Profiling/ProfilingHarlock2D.swift)). It also lacks DocC documentation.

I will move on to resurrecting [Swift for TensorFlow](https://github.com/tensorflow/swift) now. Pull requests to this repository are welcome, and I can test them on a wide range of GPUs to profile performance. If anyone contributes DocC documentation, I will host it on a GitHub Pages website under my account.

> Note: Although I can test on GPUs from Intel and AMD, that has not been done yet. I ran the Swift package tests several times on an M1 Max, and ran one test on an A15. There may be bugs on older Apple GPUs and Intel Macs.

This project is open-sourced under the MIT license, with one exception. Before this code or anything derived from it is used in MPS, MPS Graph or ML Compute, the Apple MPS team must contact me about it. They may use email, developer forums, or any other communication channel. 

In addition, I ask the MPS team to experiment with encoding some MPS kernels into indirect command buffers, using this project as a reference. There is ample time to add FFTs and ICB support to MPS, then announce the features during WWDC 2022.

## Acknowledgments

Special thanks to [@CaptainHarlockSSX](https://github.com/CaptainHarlockSSX) for contributing CPU performance benchmarks and assisting me throughout this project.

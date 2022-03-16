# MetalFFT (In Archive Mode)

MetalFFT was an experiment in next-generation GPU acceleration for 1D, 2D, and 3D variations of Fast Fourier Transforms. 

> Note: The above statement is a parody of Swift for TensorFlow's [death](https://www.tensorflow.org/swift/guide/overview), but it is serious. I recommend that you try [VkFFT](https://github.com/DTolm/VkFFT) for hardware-accelerated FFTs, which uses OpenCL on macOS.

This framework's original purpose was to become an operator for a larger GPGPU framework. Work on it paused due to unfavorable performance. For many transforms, Apple's CPU alternative from Accelerate runs faster. The A13 and later have AMX accelerators capable of ~2 TFLOPS, and the CPU implementation may harness that processing power.

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

## Acknowledgments

Special thanks to [@CaptainHarlockSSX](https://github.com/CaptainHarlockSSX) for contributing CPU performance benchmarks and assisting me throughout this project.

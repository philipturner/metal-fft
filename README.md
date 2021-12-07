# MetalFFT

1D, 2D, and 3D variations of Fast Fourier Transforms implemented for a Metal backend for Swift for TensorFlow. This is still in progress, but star or "watch" it to be alerted of its completion in the future. It will likely be kept separate from a MetalRT or [SwiftRT](https://github.com/ewconnell/swiftrt) implementation due to its domain-specific nature. For more context on the S4TF backend, see the [Differentiation iOS Demo](https://github.com/philipturner/differentiation-ios-demo).

## How to use

MetalFFT's API will be modeled after Metal Performance Shaders, but allow encoding commands into indirect command buffers. On initial release, it will only perform FFTs on tensors with dimensions that are powers of two. It will support performing multiple 1D and 2D FFTs simultaneously when tensors are arranged in contiguous blocks of memory

There will be an option to use either real or complex numbers as input. The output buffer will store both components of each complex number in an interleaved format to improve performance of memory accesses during the kernel's execution. Converting the output from complex to real will require executing a separate shader, as that is not an in-place operation.

## How it works

The FFT shaders operate in-place on the output buffer, using bit reversal as a pre-processing step and the decimation-in-time (DIT) approach. The first stage of each transform is out of place and eliminates a separate copying pass, using bit reversal to index the input and then perform 2-wide discrete Fourier transforms (butterflies). The rest proceed as shown in the `iterative-fft` pseudocode on the Cooley-Tukey algorithm [wikipedia article](https://en.wikipedia.org/wiki/Cooley–Tukey_FFT_algorithm). All twiddle factors are cached into a `MTLBuffer` when the shader first loads from the disk.

The `iterative-fft` approach was chosen because it can be easily parallelized, unlike the recursive approaches often used on the CPU.

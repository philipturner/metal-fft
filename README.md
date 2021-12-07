# MetalFFT

1D, 2D, and 3D variations of Fast Fourier Transforms implemented for a Metal backend for Swift for TensorFlow. This is still just an idea, but star or "watch" it to be alerted of its completion in the future. It will likely be kept separate from a MetalRT or [SwiftRT](https://github.com/ewconnell/swiftrt) implementation due to its domain-specific nature. For more context on the S4TF backend, see the [Differentiation iOS Demo](https://github.com/philipturner/differentiation-ios-demo).

## How to use

This framework will be modeled after Metal Performance Shaders, but allow encoding commands into indirect command buffers as well. On initial release, it will only perform FFTs on tensors with dimensions that are powers of two. It will support performing multiple 1D and 2D FFTs simultaneously when tensors are arranged in contiguous blocks of memory.

## How it works

The FFT shaders operate in-place on the output buffer, using bit reversal as a pre-processing step and the decimation in time (DIT) approach. The first stage of each transform is out of place and eliminates an extra copying pass, using bit reversal to index the input and then perform 2-wide discrete Fourier transforms (butterflies). The rest proceed as shown in the `iterative-fft` pseudocode on the Cooley-Tukey algorithm [wikipedia article](https://en.wikipedia.org/wiki/Cooley–Tukey_FFT_algorithm).





# MetalFFT

1D, 2D, and 3D variations of Fast Fourier Transforms implemented for a Metal backend for Swift for TensorFlow. This is still just an idea, but star or "watch" it to be alerted of its completion in the future. It will likely be kept separate from a MetalRT or [SwiftRT](https://github.com/ewconnell/swiftrt) implementation due to its domain-specific nature.

On initial release, this framework will only perform FFTs on tensors with dimensions that are powers of two. It will support performing multiple 1D and 2D FFTs simultaneously when tensors are arranged in contiguous blocks of memory.

For more context on the S4TF backend, see the [Differentiation iOS Demo](https://github.com/philipturner/differentiation-ios-demo).

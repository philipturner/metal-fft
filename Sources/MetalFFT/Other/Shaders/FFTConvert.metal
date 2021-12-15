//
//  FFTConvert.metal
//  MetalFFT
//
//  Created by Philip Turner on 12/11/21.
//

#include <metal_stdlib>
#include "FFTCommon.h"
using namespace metal;

constant bool reading_real          [[function_constant(0)]];
constant bool reading_split_complex [[function_constant(1)]];

constant bool writing_real          [[function_constant(2)]];
constant bool writing_split_complex [[function_constant(3)]];

// Replaces blit commands when using an ICB and converts between different data layouts

kernel void fft_convert(device void *in_data       [[buffer(0)]],
                        device void *in_data_imag  [[buffer(1), function_constant(reading_split_complex)]],
                        
                        device void *out_data      [[buffer(2)]],
                        device void *out_data_imag [[buffer(3), function_constant(writing_split_complex)]],
                        
                        uint id [[thread_position_in_grid]])
{
    complex_f data;
    
    if (reading_real) // try using threadgroup_barrier for coherent contiguous reads
    {
        float in = reinterpret_cast<device float*>(in_data)[id];
        data = complex_f(in, 0);
    }
    else if (reading_split_complex)
    {
        float in_real = reinterpret_cast<device float*>(in_data)[id];
        float in_imag = reinterpret_cast<device float*>(in_data_imag)[id];
        data = complex_f(in_real, in_imag);
    }
    else
    {
        data = reinterpret_cast<device complex_f*>(in_data)[id];
    }
    
    if (writing_real) // try using threadgroup_barrier for coherent contiguous writes
    {
        reinterpret_cast<device float*>(out_data)[id] = data.real();
    }
    else if (writing_split_complex)
    {
        reinterpret_cast<device float*>(out_data)[id] = data.real();
        reinterpret_cast<device float*>(out_data_imag)[id] = data.imag();
    }
    else
    {
        reinterpret_cast<device complex_f*>(out_data)[id] = data;
    }
}

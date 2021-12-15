//
//  FFTIntro.metal
//  MetalFFT
//
//  Created by Philip Turner on 12/11/21.
//

#include <metal_stdlib>
#include "FFTCommon.h"
using namespace metal;

constant ushort grid_num_dims         [[function_constant(0)]];
constant ushort fft_direction         [[function_constant(1)]];
constant bool   reading_real          [[function_constant(2)]];
constant bool   reading_split_complex [[function_constant(3)]];
constant bool   is_inverse            [[function_constant(4)]];

typedef struct {
    // stride_power = uint3{ 0, x_power, xy_power }[fftDirection]
    ushort x_power; // ctz(grid_size.x)
    ushort xy_power; // ctz(grid_size.y)
    uint   in_t_index_bit; // grid_size[fftDirection] << stride_power
    
    float  normalization; // 1 / float(# data points the iFFT takes as input)
    
    ushort in_index_shift; // 32 - ctz(grid_size[fftDirection]) - stride_power
    ushort out_index_shift; // stride_power + 1
    uint   out_t_index_bit; // 1 << stride_power
} fft_intro_args;

// Performs the first FFT/iFFT stage of each pass, where m = 2.

kernel void fft_intro(constant fft_intro_args &args  [[buffer(0)]],
                      device   void      *input      [[buffer(1)]],
                      device   complex_f *output     [[buffer(2)]],
                      device   float     *input_imag [[buffer(3), function_constant(reading_split_complex)]],
                      
                      uint3 id        [[thread_position_in_grid]],
                      uint3 grid_size [[threads_per_grid]])
{
    // Locate inputs in memory
    
    uint u_index = reverse_bits(id[fft_direction]) >> args.in_index_shift;
    uint base_index = 0;
    
    if (grid_num_dims == 2)
    {
        base_index = (fft_direction == 0) ? (id.y << args.x_power) : id.x;
    }
    else if (grid_num_dims == 3)
    {
        if (fft_direction == 0)
        {
            base_index = (id.y << args.x_power) + (id.z << args.xy_power);
        }
        else
        {
            base_index = id.x;
            base_index += (fft_direction == 1) ? (id.z << args.xy_power) : (id.y << args.x_power);
        }
    }
    
    u_index += base_index;
    uint t_index = u_index | args.in_t_index_bit;
    
    // Compute radix-2 butterfly
    
    complex_f u_out;
    complex_f t_out;
    
    if (reading_real || reading_split_complex)
    {
        auto input_data = reinterpret_cast<device float*>(input);
        float u_real = input_data[u_index];
        float t_real = input_data[t_index];
        
        float u_imag = 0;
        float t_imag = 0;
        
        if (reading_split_complex)
        {
            u_imag = input_imag[u_index];
            t_imag = input_imag[t_index];
        }
        
        u_out = complex_f(u_real + t_real, u_imag + t_imag);
        t_out = complex_f(u_real - t_real, u_imag - t_imag);
    }
    else
    {
        auto input_data = reinterpret_cast<device complex_f*>(input);
        complex_f u = input_data[u_index];
        complex_f t = input_data[t_index];
        
        u_out = u + t;
        t_out = u - t;
    }
    
    if (is_inverse)
    {
        u_out = u_out * args.normalization;
        t_out = t_out * args.normalization;
    }
    
    u_index = (id[fft_direction] << args.out_index_shift) + base_index;
    t_index = u_index | args.out_t_index_bit;
    
    output[u_index] = u_out;
    output[t_index] = t_out;
}

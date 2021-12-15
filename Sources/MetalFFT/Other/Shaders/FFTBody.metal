//
//  FFTBody.metal
//  MetalFFT
//
//  Created by Philip Turner on 12/11/21.
//

#include <metal_stdlib>
#include "FFTCommon.h"
using namespace metal;

constant ushort grid_num_dims           [[function_constant(0)]];
constant ushort fft_direction           [[function_constant(1)]];
constant bool   reading_twiddle_factors [[function_constant(2)]];

typedef struct {
    // stride_power = uint3{ 0, x_power, y_power }[fftDirection]
    ushort x_power; // ctz(grid_size.x)
    ushort xy_power; // ctz(grid_size.y)
    uint   t_index_bit; // m/2 << stride_power
    
    uint   j_mask; // m/2 - 1
    uint   twiddle_factor_offset; // m/2
    float  m_recip_doubled; // 1 / float(m/2)
} fft_body_args;

// Performs FFT/iFFT stages where m >= 4.

kernel void fft_body(constant fft_body_args &args        [[buffer(0)]],
                     constant complex_f *twiddle_factors [[buffer(1), function_constant(reading_twiddle_factors)]],
                     device   complex_f *in_place_data   [[buffer(2)]],
                     
                     uint3 id [[thread_position_in_grid]])
{
    uint j = id[fft_direction] & args.j_mask;
    uint k = (id[fft_direction] - j) << 1;
    
    // Locate inputs in memory
    
    uint u_index = k + j;
    
    uint base_index = 0;
    ushort stride_shift = 0;
    
    if (grid_num_dims == 2)
    {
        if (fft_direction == 0)
        {
            base_index = id.y << args.x_power;
        }
        else
        {
            base_index = id.x;
            stride_shift = args.x_power;
        }
    }
    else if (grid_num_dims == 3)
    {
        if (fft_direction == 0)
        {
            base_index = (id.y << args.x_power) + (id.z << args.xy_power);
        }
        else if (fft_direction == 1)
        {
            base_index = id.x + (id.z << args.xy_power);
            stride_shift = args.x_power;
        }
        else
        {
            base_index = id.x + (id.y << args.x_power);
            stride_shift = args.xy_power;
        }
    }
    
    u_index = (u_index << stride_shift) + base_index;
    uint t_index = u_index | args.t_index_bit;
    
    complex_f u = in_place_data[u_index];
    complex_f t = in_place_data[t_index];
    
    // Get twiddle factor
    
    complex_f twiddle_factor;
    
    if (reading_twiddle_factors)
    {
        twiddle_factor = twiddle_factors[args.twiddle_factor_offset + j];
    }
    else
    {
        float phase_doubled = float(j) * args.m_recip_doubled;
        twiddle_factor = complex_f::get_twiddle_factor(phase_doubled);
    }
    
    // Compute radix-2 butterly
    
    t = t * twiddle_factor;
    
    in_place_data[u_index] = u + t;
    in_place_data[t_index] = u - t;
}

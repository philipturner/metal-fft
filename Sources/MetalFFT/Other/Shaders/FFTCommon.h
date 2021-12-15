//
//  FFTCommon.h
//  MetalFFT
//
//  Created by Philip Turner on 12/11/21.
//

#ifndef FFTCommon_h
#define FFTCommon_h

#include <metal_stdlib>
using namespace metal;

class complex_f {
    float2 data;
    
public:
    float real() const { return data[0]; }
    float imag() const { return data[1]; }
    
    complex_f()
    {
        
    }
    
    explicit complex_f(float real, float imag)
    {
        data = { real, imag };
    }
    
    // Add/subtract
    
    friend complex_f operator+(complex_f lhs, complex_f rhs)
    {
        return complex_f(lhs.real() + rhs.real(),
                         lhs.imag() + rhs.imag());
    }
    
    friend complex_f operator-(complex_f lhs, complex_f rhs)
    {
        return complex_f(lhs.real() - rhs.real(),
                         lhs.imag() - rhs.imag());
    }
    
    // Multiply
    
    friend complex_f operator*(complex_f lhs, complex_f rhs)
    {
        float real = lhs.real() * rhs.real() - lhs.imag() * rhs.imag();
        float imag = lhs.real() * rhs.imag() + lhs.imag() * rhs.real();
        
        return complex_f(real, imag);
    }
    
    friend complex_f operator*(float lhs, complex_f rhs)
    {
        return complex_f(lhs * rhs.real(), lhs * rhs.imag());
    }
    
    friend complex_f operator*(complex_f lhs, float rhs)
    {
        return rhs * lhs;
    }
    
    // exp(2 * pi * phase) where 0 <= abs(phase) < 0.5
    // phase <= 0 for FFT, >= 0 for iFFT
    
    static complex_f get_twiddle_factor(float phase_doubled)
    {
        float sinval = precise::sinpi(phase_doubled);
        float cosval = precise::sqrt(1 - sinval * sinval);
        
        // abs(phase) > 0.25
        if (abs(phase_doubled) > 0.5) { cosval = -cosval; }
        
        return complex_f(cosval, sinval);
    }
};

#endif /* FFTCommon_h */

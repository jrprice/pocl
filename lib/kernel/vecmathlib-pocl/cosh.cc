// Note: This file has been automatically generated. Do not modify.

#include "pocl-compat.h"

// cosh: ['VF'] -> VF

#ifdef cl_khr_fp16

// cosh: VF=half
#if defined VECMATHLIB_HAVE_VEC_HALF_1 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement cosh by calling vecmathlib
half _cl_cosh(half x0)
{
  vecmathlib::realvec<half,1> y0 = bitcast<half,vecmathlib::realvec<half,1> >(x0);
  vecmathlib::realvec<half,1> r = vecmathlib::cosh(y0);
  return bitcast<vecmathlib::realvec<half,1>,half>((r));
}
#elif ! defined POCL_VECMATHLIB_BUILTIN
// Implement cosh by calling libm
half _cl_cosh(half x0)
{
  vecmathlib::realpseudovec<half,1> y0 = x0;
  vecmathlib::realpseudovec<half,1> r = cosh(y0);
  return (r)[0];
}
#else
// Implement cosh by calling builtin
half _cl_cosh(half x0)
{
  vecmathlib::realbuiltinvec<half,1> y0 = x0;
  vecmathlib::realbuiltinvec<half,1> r = cosh(y0);
  return (r)[0];
}
#endif

// cosh: VF=half2
#if defined VECMATHLIB_HAVE_VEC_HALF_2 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement cosh by calling vecmathlib
half2 _cl_cosh(half2 x0)
{
  vecmathlib::realvec<half,2> y0 = bitcast<half2,vecmathlib::realvec<half,2> >(x0);
  vecmathlib::realvec<half,2> r = vecmathlib::cosh(y0);
  return bitcast<vecmathlib::realvec<half,2>,half2>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_HALF_4 || defined VECMATHLIB_HAVE_VEC_HALF_8 || defined VECMATHLIB_HAVE_VEC_HALF_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement cosh by using a larger vector size
half4 _cl_cosh(half4);
half2 _cl_cosh(half2 x0)
{
  half4 y0 = bitcast<half2,half4>(x0);
  half4 r = _cl_cosh(y0);
  return bitcast<half4,half2>(r);
}
#else
// Implement cosh by splitting into a smaller vector size
half _cl_cosh(half);
half2 _cl_cosh(half2 x0)
{
  pair_half y0 = bitcast<half2,pair_half>(x0);
  pair_half r;
  r.lo = _cl_cosh(y0.lo);
  r.hi = _cl_cosh(y0.hi);
  pocl_static_assert(sizeof(pair_half) == sizeof(half2));
  return bitcast<pair_half,half2>(r);
}
#endif

// cosh: VF=half3
#if defined VECMATHLIB_HAVE_VEC_HALF_3 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement cosh by calling vecmathlib
half3 _cl_cosh(half3 x0)
{
  vecmathlib::realvec<half,3> y0 = bitcast<half3,vecmathlib::realvec<half,3> >(x0);
  vecmathlib::realvec<half,3> r = vecmathlib::cosh(y0);
  return bitcast<vecmathlib::realvec<half,3>,half3>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_HALF_4 || defined VECMATHLIB_HAVE_VEC_HALF_8 || defined VECMATHLIB_HAVE_VEC_HALF_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement cosh by using a larger vector size
half4 _cl_cosh(half4);
half3 _cl_cosh(half3 x0)
{
  half4 y0 = bitcast<half3,half4>(x0);
  half4 r = _cl_cosh(y0);
  return bitcast<half4,half3>(r);
}
#else
// Implement cosh by splitting into a smaller vector size
half2 _cl_cosh(half2);
half3 _cl_cosh(half3 x0)
{
  pair_half2 y0 = bitcast<half3,pair_half2>(x0);
  pair_half2 r;
  r.lo = _cl_cosh(y0.lo);
  r.hi = _cl_cosh(y0.hi);
  pocl_static_assert(sizeof(pair_half2) == sizeof(half3));
  return bitcast<pair_half2,half3>(r);
}
#endif

// cosh: VF=half4
#if defined VECMATHLIB_HAVE_VEC_HALF_4 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement cosh by calling vecmathlib
half4 _cl_cosh(half4 x0)
{
  vecmathlib::realvec<half,4> y0 = bitcast<half4,vecmathlib::realvec<half,4> >(x0);
  vecmathlib::realvec<half,4> r = vecmathlib::cosh(y0);
  return bitcast<vecmathlib::realvec<half,4>,half4>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_HALF_8 || defined VECMATHLIB_HAVE_VEC_HALF_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement cosh by using a larger vector size
half8 _cl_cosh(half8);
half4 _cl_cosh(half4 x0)
{
  half8 y0 = bitcast<half4,half8>(x0);
  half8 r = _cl_cosh(y0);
  return bitcast<half8,half4>(r);
}
#else
// Implement cosh by splitting into a smaller vector size
half2 _cl_cosh(half2);
half4 _cl_cosh(half4 x0)
{
  pair_half2 y0 = bitcast<half4,pair_half2>(x0);
  pair_half2 r;
  r.lo = _cl_cosh(y0.lo);
  r.hi = _cl_cosh(y0.hi);
  pocl_static_assert(sizeof(pair_half2) == sizeof(half4));
  return bitcast<pair_half2,half4>(r);
}
#endif

// cosh: VF=half8
#if defined VECMATHLIB_HAVE_VEC_HALF_8 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement cosh by calling vecmathlib
half8 _cl_cosh(half8 x0)
{
  vecmathlib::realvec<half,8> y0 = bitcast<half8,vecmathlib::realvec<half,8> >(x0);
  vecmathlib::realvec<half,8> r = vecmathlib::cosh(y0);
  return bitcast<vecmathlib::realvec<half,8>,half8>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_HALF_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement cosh by using a larger vector size
half16 _cl_cosh(half16);
half8 _cl_cosh(half8 x0)
{
  half16 y0 = bitcast<half8,half16>(x0);
  half16 r = _cl_cosh(y0);
  return bitcast<half16,half8>(r);
}
#else
// Implement cosh by splitting into a smaller vector size
half4 _cl_cosh(half4);
half8 _cl_cosh(half8 x0)
{
  pair_half4 y0 = bitcast<half8,pair_half4>(x0);
  pair_half4 r;
  r.lo = _cl_cosh(y0.lo);
  r.hi = _cl_cosh(y0.hi);
  pocl_static_assert(sizeof(pair_half4) == sizeof(half8));
  return bitcast<pair_half4,half8>(r);
}
#endif

// cosh: VF=half16
#if defined VECMATHLIB_HAVE_VEC_HALF_16 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement cosh by calling vecmathlib
half16 _cl_cosh(half16 x0)
{
  vecmathlib::realvec<half,16> y0 = bitcast<half16,vecmathlib::realvec<half,16> >(x0);
  vecmathlib::realvec<half,16> r = vecmathlib::cosh(y0);
  return bitcast<vecmathlib::realvec<half,16>,half16>((r));
}
#else
// Implement cosh by splitting into a smaller vector size
half8 _cl_cosh(half8);
half16 _cl_cosh(half16 x0)
{
  pair_half8 y0 = bitcast<half16,pair_half8>(x0);
  pair_half8 r;
  r.lo = _cl_cosh(y0.lo);
  r.hi = _cl_cosh(y0.hi);
  pocl_static_assert(sizeof(pair_half8) == sizeof(half16));
  return bitcast<pair_half8,half16>(r);
}
#endif

#endif // #ifdef cl_khr_fp16

// cosh: VF=float
#if defined VECMATHLIB_HAVE_VEC_FLOAT_1 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement cosh by calling vecmathlib
float _cl_cosh(float x0)
{
  vecmathlib::realvec<float,1> y0 = bitcast<float,vecmathlib::realvec<float,1> >(x0);
  vecmathlib::realvec<float,1> r = vecmathlib::cosh(y0);
  return bitcast<vecmathlib::realvec<float,1>,float>((r));
}
#elif ! defined POCL_VECMATHLIB_BUILTIN
// Implement cosh by calling libm
float _cl_cosh(float x0)
{
  vecmathlib::realpseudovec<float,1> y0 = x0;
  vecmathlib::realpseudovec<float,1> r = cosh(y0);
  return (r)[0];
}
#else
// Implement cosh by calling builtin
float _cl_cosh(float x0)
{
  vecmathlib::realbuiltinvec<float,1> y0 = x0;
  vecmathlib::realbuiltinvec<float,1> r = cosh(y0);
  return (r)[0];
}
#endif

// cosh: VF=float2
#if defined VECMATHLIB_HAVE_VEC_FLOAT_2 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement cosh by calling vecmathlib
float2 _cl_cosh(float2 x0)
{
  vecmathlib::realvec<float,2> y0 = bitcast<float2,vecmathlib::realvec<float,2> >(x0);
  vecmathlib::realvec<float,2> r = vecmathlib::cosh(y0);
  return bitcast<vecmathlib::realvec<float,2>,float2>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_FLOAT_4 || defined VECMATHLIB_HAVE_VEC_FLOAT_8 || defined VECMATHLIB_HAVE_VEC_FLOAT_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement cosh by using a larger vector size
float4 _cl_cosh(float4);
float2 _cl_cosh(float2 x0)
{
  float4 y0 = bitcast<float2,float4>(x0);
  float4 r = _cl_cosh(y0);
  return bitcast<float4,float2>(r);
}
#else
// Implement cosh by splitting into a smaller vector size
float _cl_cosh(float);
float2 _cl_cosh(float2 x0)
{
  pair_float y0 = bitcast<float2,pair_float>(x0);
  pair_float r;
  r.lo = _cl_cosh(y0.lo);
  r.hi = _cl_cosh(y0.hi);
  pocl_static_assert(sizeof(pair_float) == sizeof(float2));
  return bitcast<pair_float,float2>(r);
}
#endif

// cosh: VF=float3
#if defined VECMATHLIB_HAVE_VEC_FLOAT_3 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement cosh by calling vecmathlib
float3 _cl_cosh(float3 x0)
{
  vecmathlib::realvec<float,3> y0 = bitcast<float3,vecmathlib::realvec<float,3> >(x0);
  vecmathlib::realvec<float,3> r = vecmathlib::cosh(y0);
  return bitcast<vecmathlib::realvec<float,3>,float3>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_FLOAT_4 || defined VECMATHLIB_HAVE_VEC_FLOAT_8 || defined VECMATHLIB_HAVE_VEC_FLOAT_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement cosh by using a larger vector size
float4 _cl_cosh(float4);
float3 _cl_cosh(float3 x0)
{
  float4 y0 = bitcast<float3,float4>(x0);
  float4 r = _cl_cosh(y0);
  return bitcast<float4,float3>(r);
}
#else
// Implement cosh by splitting into a smaller vector size
float2 _cl_cosh(float2);
float3 _cl_cosh(float3 x0)
{
  pair_float2 y0 = bitcast<float3,pair_float2>(x0);
  pair_float2 r;
  r.lo = _cl_cosh(y0.lo);
  r.hi = _cl_cosh(y0.hi);
  pocl_static_assert(sizeof(pair_float2) == sizeof(float3));
  return bitcast<pair_float2,float3>(r);
}
#endif

// cosh: VF=float4
#if defined VECMATHLIB_HAVE_VEC_FLOAT_4 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement cosh by calling vecmathlib
float4 _cl_cosh(float4 x0)
{
  vecmathlib::realvec<float,4> y0 = bitcast<float4,vecmathlib::realvec<float,4> >(x0);
  vecmathlib::realvec<float,4> r = vecmathlib::cosh(y0);
  return bitcast<vecmathlib::realvec<float,4>,float4>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_FLOAT_8 || defined VECMATHLIB_HAVE_VEC_FLOAT_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement cosh by using a larger vector size
float8 _cl_cosh(float8);
float4 _cl_cosh(float4 x0)
{
  float8 y0 = bitcast<float4,float8>(x0);
  float8 r = _cl_cosh(y0);
  return bitcast<float8,float4>(r);
}
#else
// Implement cosh by splitting into a smaller vector size
float2 _cl_cosh(float2);
float4 _cl_cosh(float4 x0)
{
  pair_float2 y0 = bitcast<float4,pair_float2>(x0);
  pair_float2 r;
  r.lo = _cl_cosh(y0.lo);
  r.hi = _cl_cosh(y0.hi);
  pocl_static_assert(sizeof(pair_float2) == sizeof(float4));
  return bitcast<pair_float2,float4>(r);
}
#endif

// cosh: VF=float8
#if defined VECMATHLIB_HAVE_VEC_FLOAT_8 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement cosh by calling vecmathlib
float8 _cl_cosh(float8 x0)
{
  vecmathlib::realvec<float,8> y0 = bitcast<float8,vecmathlib::realvec<float,8> >(x0);
  vecmathlib::realvec<float,8> r = vecmathlib::cosh(y0);
  return bitcast<vecmathlib::realvec<float,8>,float8>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_FLOAT_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement cosh by using a larger vector size
float16 _cl_cosh(float16);
float8 _cl_cosh(float8 x0)
{
  float16 y0 = bitcast<float8,float16>(x0);
  float16 r = _cl_cosh(y0);
  return bitcast<float16,float8>(r);
}
#else
// Implement cosh by splitting into a smaller vector size
float4 _cl_cosh(float4);
float8 _cl_cosh(float8 x0)
{
  pair_float4 y0 = bitcast<float8,pair_float4>(x0);
  pair_float4 r;
  r.lo = _cl_cosh(y0.lo);
  r.hi = _cl_cosh(y0.hi);
  pocl_static_assert(sizeof(pair_float4) == sizeof(float8));
  return bitcast<pair_float4,float8>(r);
}
#endif

// cosh: VF=float16
#if defined VECMATHLIB_HAVE_VEC_FLOAT_16 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement cosh by calling vecmathlib
float16 _cl_cosh(float16 x0)
{
  vecmathlib::realvec<float,16> y0 = bitcast<float16,vecmathlib::realvec<float,16> >(x0);
  vecmathlib::realvec<float,16> r = vecmathlib::cosh(y0);
  return bitcast<vecmathlib::realvec<float,16>,float16>((r));
}
#else
// Implement cosh by splitting into a smaller vector size
float8 _cl_cosh(float8);
float16 _cl_cosh(float16 x0)
{
  pair_float8 y0 = bitcast<float16,pair_float8>(x0);
  pair_float8 r;
  r.lo = _cl_cosh(y0.lo);
  r.hi = _cl_cosh(y0.hi);
  pocl_static_assert(sizeof(pair_float8) == sizeof(float16));
  return bitcast<pair_float8,float16>(r);
}
#endif

#ifdef cl_khr_fp64

// cosh: VF=double
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_1 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement cosh by calling vecmathlib
double _cl_cosh(double x0)
{
  vecmathlib::realvec<double,1> y0 = bitcast<double,vecmathlib::realvec<double,1> >(x0);
  vecmathlib::realvec<double,1> r = vecmathlib::cosh(y0);
  return bitcast<vecmathlib::realvec<double,1>,double>((r));
}
#elif ! defined POCL_VECMATHLIB_BUILTIN
// Implement cosh by calling libm
double _cl_cosh(double x0)
{
  vecmathlib::realpseudovec<double,1> y0 = x0;
  vecmathlib::realpseudovec<double,1> r = cosh(y0);
  return (r)[0];
}
#else
// Implement cosh by calling builtin
double _cl_cosh(double x0)
{
  vecmathlib::realbuiltinvec<double,1> y0 = x0;
  vecmathlib::realbuiltinvec<double,1> r = cosh(y0);
  return (r)[0];
}
#endif

// cosh: VF=double2
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_2 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement cosh by calling vecmathlib
double2 _cl_cosh(double2 x0)
{
  vecmathlib::realvec<double,2> y0 = bitcast<double2,vecmathlib::realvec<double,2> >(x0);
  vecmathlib::realvec<double,2> r = vecmathlib::cosh(y0);
  return bitcast<vecmathlib::realvec<double,2>,double2>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_DOUBLE_4 || defined VECMATHLIB_HAVE_VEC_DOUBLE_8 || defined VECMATHLIB_HAVE_VEC_DOUBLE_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement cosh by using a larger vector size
double4 _cl_cosh(double4);
double2 _cl_cosh(double2 x0)
{
  double4 y0 = bitcast<double2,double4>(x0);
  double4 r = _cl_cosh(y0);
  return bitcast<double4,double2>(r);
}
#else
// Implement cosh by splitting into a smaller vector size
double _cl_cosh(double);
double2 _cl_cosh(double2 x0)
{
  pair_double y0 = bitcast<double2,pair_double>(x0);
  pair_double r;
  r.lo = _cl_cosh(y0.lo);
  r.hi = _cl_cosh(y0.hi);
  pocl_static_assert(sizeof(pair_double) == sizeof(double2));
  return bitcast<pair_double,double2>(r);
}
#endif

// cosh: VF=double3
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_3 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement cosh by calling vecmathlib
double3 _cl_cosh(double3 x0)
{
  vecmathlib::realvec<double,3> y0 = bitcast<double3,vecmathlib::realvec<double,3> >(x0);
  vecmathlib::realvec<double,3> r = vecmathlib::cosh(y0);
  return bitcast<vecmathlib::realvec<double,3>,double3>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_DOUBLE_4 || defined VECMATHLIB_HAVE_VEC_DOUBLE_8 || defined VECMATHLIB_HAVE_VEC_DOUBLE_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement cosh by using a larger vector size
double4 _cl_cosh(double4);
double3 _cl_cosh(double3 x0)
{
  double4 y0 = bitcast<double3,double4>(x0);
  double4 r = _cl_cosh(y0);
  return bitcast<double4,double3>(r);
}
#else
// Implement cosh by splitting into a smaller vector size
double2 _cl_cosh(double2);
double3 _cl_cosh(double3 x0)
{
  pair_double2 y0 = bitcast<double3,pair_double2>(x0);
  pair_double2 r;
  r.lo = _cl_cosh(y0.lo);
  r.hi = _cl_cosh(y0.hi);
  pocl_static_assert(sizeof(pair_double2) == sizeof(double3));
  return bitcast<pair_double2,double3>(r);
}
#endif

// cosh: VF=double4
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_4 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement cosh by calling vecmathlib
double4 _cl_cosh(double4 x0)
{
  vecmathlib::realvec<double,4> y0 = bitcast<double4,vecmathlib::realvec<double,4> >(x0);
  vecmathlib::realvec<double,4> r = vecmathlib::cosh(y0);
  return bitcast<vecmathlib::realvec<double,4>,double4>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_DOUBLE_8 || defined VECMATHLIB_HAVE_VEC_DOUBLE_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement cosh by using a larger vector size
double8 _cl_cosh(double8);
double4 _cl_cosh(double4 x0)
{
  double8 y0 = bitcast<double4,double8>(x0);
  double8 r = _cl_cosh(y0);
  return bitcast<double8,double4>(r);
}
#else
// Implement cosh by splitting into a smaller vector size
double2 _cl_cosh(double2);
double4 _cl_cosh(double4 x0)
{
  pair_double2 y0 = bitcast<double4,pair_double2>(x0);
  pair_double2 r;
  r.lo = _cl_cosh(y0.lo);
  r.hi = _cl_cosh(y0.hi);
  pocl_static_assert(sizeof(pair_double2) == sizeof(double4));
  return bitcast<pair_double2,double4>(r);
}
#endif

// cosh: VF=double8
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_8 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement cosh by calling vecmathlib
double8 _cl_cosh(double8 x0)
{
  vecmathlib::realvec<double,8> y0 = bitcast<double8,vecmathlib::realvec<double,8> >(x0);
  vecmathlib::realvec<double,8> r = vecmathlib::cosh(y0);
  return bitcast<vecmathlib::realvec<double,8>,double8>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_DOUBLE_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement cosh by using a larger vector size
double16 _cl_cosh(double16);
double8 _cl_cosh(double8 x0)
{
  double16 y0 = bitcast<double8,double16>(x0);
  double16 r = _cl_cosh(y0);
  return bitcast<double16,double8>(r);
}
#else
// Implement cosh by splitting into a smaller vector size
double4 _cl_cosh(double4);
double8 _cl_cosh(double8 x0)
{
  pair_double4 y0 = bitcast<double8,pair_double4>(x0);
  pair_double4 r;
  r.lo = _cl_cosh(y0.lo);
  r.hi = _cl_cosh(y0.hi);
  pocl_static_assert(sizeof(pair_double4) == sizeof(double8));
  return bitcast<pair_double4,double8>(r);
}
#endif

// cosh: VF=double16
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_16 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement cosh by calling vecmathlib
double16 _cl_cosh(double16 x0)
{
  vecmathlib::realvec<double,16> y0 = bitcast<double16,vecmathlib::realvec<double,16> >(x0);
  vecmathlib::realvec<double,16> r = vecmathlib::cosh(y0);
  return bitcast<vecmathlib::realvec<double,16>,double16>((r));
}
#else
// Implement cosh by splitting into a smaller vector size
double8 _cl_cosh(double8);
double16 _cl_cosh(double16 x0)
{
  pair_double8 y0 = bitcast<double16,pair_double8>(x0);
  pair_double8 r;
  r.lo = _cl_cosh(y0.lo);
  r.hi = _cl_cosh(y0.hi);
  pocl_static_assert(sizeof(pair_double8) == sizeof(double16));
  return bitcast<pair_double8,double16>(r);
}
#endif

#endif // #ifdef cl_khr_fp64

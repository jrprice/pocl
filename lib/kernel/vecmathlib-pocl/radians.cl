// Note: This file has been automatically generated. Do not modify.

// Needed for fract()
#define POCL_FRACT_MIN_H 0x1.ffcp-1h
#define POCL_FRACT_MIN   0x1.fffffffffffffp-1
#define POCL_FRACT_MIN_F 0x1.fffffep-1f

// Choose a constant with a particular precision
#ifdef cl_khr_fp16
#  define IF_HALF(TYPE, VAL, OTHER) \
          (sizeof(TYPE)==sizeof(half) ? (TYPE)(VAL) : (TYPE)(OTHER))
#else
#  define IF_HALF(TYPE, VAL, OTHER) (OTHER)
#endif

#ifdef cl_khr_fp64
#  define IF_DOUBLE(TYPE, VAL, OTHER) \
          (sizeof(TYPE)==sizeof(double) ? (TYPE)(VAL) : (TYPE)(OTHER))
#else
#  define IF_DOUBLE(TYPE, VAL, OTHER) (OTHER)
#endif

#define TYPED_CONST(TYPE, HALF_VAL, SINGLE_VAL, DOUBLE_VAL) \
        IF_HALF(TYPE, HALF_VAL, IF_DOUBLE(TYPE, DOUBLE_VAL, SINGLE_VAL))



// radians: ['VF'] -> VF

#ifdef cl_khr_fp16

// radians: VF=half
// Implement radians directly
__attribute__((__overloadable__))
half _cl_radians(half x0)
{
  typedef short iscalar_t;
  typedef int jscalar_t;
  typedef int kscalar_t;
  typedef half scalar_t;
  typedef short ivector_t;
  typedef int jvector_t;
  typedef int kvector_t;
  typedef half vector_t;
#define convert_ivector_t convert_short
#define convert_jvector_t convert_int
#define convert_kvector_t convert_int
#define convert_vector_t convert_half
  return TYPED_CONST(scalar_t, M_PI_H, M_PI_F, M_PI)/(scalar_t)180*x0;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// radians: VF=half2
// Implement radians directly
__attribute__((__overloadable__))
half2 _cl_radians(half2 x0)
{
  typedef short iscalar_t;
  typedef short jscalar_t;
  typedef int kscalar_t;
  typedef half scalar_t;
  typedef short2 ivector_t;
  typedef short2 jvector_t;
  typedef int2 kvector_t;
  typedef half2 vector_t;
#define convert_ivector_t convert_short2
#define convert_jvector_t convert_short2
#define convert_kvector_t convert_int2
#define convert_vector_t convert_half2
  return TYPED_CONST(scalar_t, M_PI_H, M_PI_F, M_PI)/(scalar_t)180*x0;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// radians: VF=half3
// Implement radians directly
__attribute__((__overloadable__))
half3 _cl_radians(half3 x0)
{
  typedef short iscalar_t;
  typedef short jscalar_t;
  typedef int kscalar_t;
  typedef half scalar_t;
  typedef short3 ivector_t;
  typedef short3 jvector_t;
  typedef int3 kvector_t;
  typedef half3 vector_t;
#define convert_ivector_t convert_short3
#define convert_jvector_t convert_short3
#define convert_kvector_t convert_int3
#define convert_vector_t convert_half3
  return TYPED_CONST(scalar_t, M_PI_H, M_PI_F, M_PI)/(scalar_t)180*x0;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// radians: VF=half4
// Implement radians directly
__attribute__((__overloadable__))
half4 _cl_radians(half4 x0)
{
  typedef short iscalar_t;
  typedef short jscalar_t;
  typedef int kscalar_t;
  typedef half scalar_t;
  typedef short4 ivector_t;
  typedef short4 jvector_t;
  typedef int4 kvector_t;
  typedef half4 vector_t;
#define convert_ivector_t convert_short4
#define convert_jvector_t convert_short4
#define convert_kvector_t convert_int4
#define convert_vector_t convert_half4
  return TYPED_CONST(scalar_t, M_PI_H, M_PI_F, M_PI)/(scalar_t)180*x0;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// radians: VF=half8
// Implement radians directly
__attribute__((__overloadable__))
half8 _cl_radians(half8 x0)
{
  typedef short iscalar_t;
  typedef short jscalar_t;
  typedef int kscalar_t;
  typedef half scalar_t;
  typedef short8 ivector_t;
  typedef short8 jvector_t;
  typedef int8 kvector_t;
  typedef half8 vector_t;
#define convert_ivector_t convert_short8
#define convert_jvector_t convert_short8
#define convert_kvector_t convert_int8
#define convert_vector_t convert_half8
  return TYPED_CONST(scalar_t, M_PI_H, M_PI_F, M_PI)/(scalar_t)180*x0;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// radians: VF=half16
// Implement radians directly
__attribute__((__overloadable__))
half16 _cl_radians(half16 x0)
{
  typedef short iscalar_t;
  typedef short jscalar_t;
  typedef int kscalar_t;
  typedef half scalar_t;
  typedef short16 ivector_t;
  typedef short16 jvector_t;
  typedef int16 kvector_t;
  typedef half16 vector_t;
#define convert_ivector_t convert_short16
#define convert_jvector_t convert_short16
#define convert_kvector_t convert_int16
#define convert_vector_t convert_half16
  return TYPED_CONST(scalar_t, M_PI_H, M_PI_F, M_PI)/(scalar_t)180*x0;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

#endif // #ifdef cl_khr_fp16

// radians: VF=float
// Implement radians directly
__attribute__((__overloadable__))
float _cl_radians(float x0)
{
  typedef int iscalar_t;
  typedef int jscalar_t;
  typedef int kscalar_t;
  typedef float scalar_t;
  typedef int ivector_t;
  typedef int jvector_t;
  typedef int kvector_t;
  typedef float vector_t;
#define convert_ivector_t convert_int
#define convert_jvector_t convert_int
#define convert_kvector_t convert_int
#define convert_vector_t convert_float
  return TYPED_CONST(scalar_t, M_PI_H, M_PI_F, M_PI)/(scalar_t)180*x0;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// radians: VF=float2
// Implement radians directly
__attribute__((__overloadable__))
float2 _cl_radians(float2 x0)
{
  typedef int iscalar_t;
  typedef int jscalar_t;
  typedef int kscalar_t;
  typedef float scalar_t;
  typedef int2 ivector_t;
  typedef int2 jvector_t;
  typedef int2 kvector_t;
  typedef float2 vector_t;
#define convert_ivector_t convert_int2
#define convert_jvector_t convert_int2
#define convert_kvector_t convert_int2
#define convert_vector_t convert_float2
  return TYPED_CONST(scalar_t, M_PI_H, M_PI_F, M_PI)/(scalar_t)180*x0;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// radians: VF=float3
// Implement radians directly
__attribute__((__overloadable__))
float3 _cl_radians(float3 x0)
{
  typedef int iscalar_t;
  typedef int jscalar_t;
  typedef int kscalar_t;
  typedef float scalar_t;
  typedef int3 ivector_t;
  typedef int3 jvector_t;
  typedef int3 kvector_t;
  typedef float3 vector_t;
#define convert_ivector_t convert_int3
#define convert_jvector_t convert_int3
#define convert_kvector_t convert_int3
#define convert_vector_t convert_float3
  return TYPED_CONST(scalar_t, M_PI_H, M_PI_F, M_PI)/(scalar_t)180*x0;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// radians: VF=float4
// Implement radians directly
__attribute__((__overloadable__))
float4 _cl_radians(float4 x0)
{
  typedef int iscalar_t;
  typedef int jscalar_t;
  typedef int kscalar_t;
  typedef float scalar_t;
  typedef int4 ivector_t;
  typedef int4 jvector_t;
  typedef int4 kvector_t;
  typedef float4 vector_t;
#define convert_ivector_t convert_int4
#define convert_jvector_t convert_int4
#define convert_kvector_t convert_int4
#define convert_vector_t convert_float4
  return TYPED_CONST(scalar_t, M_PI_H, M_PI_F, M_PI)/(scalar_t)180*x0;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// radians: VF=float8
// Implement radians directly
__attribute__((__overloadable__))
float8 _cl_radians(float8 x0)
{
  typedef int iscalar_t;
  typedef int jscalar_t;
  typedef int kscalar_t;
  typedef float scalar_t;
  typedef int8 ivector_t;
  typedef int8 jvector_t;
  typedef int8 kvector_t;
  typedef float8 vector_t;
#define convert_ivector_t convert_int8
#define convert_jvector_t convert_int8
#define convert_kvector_t convert_int8
#define convert_vector_t convert_float8
  return TYPED_CONST(scalar_t, M_PI_H, M_PI_F, M_PI)/(scalar_t)180*x0;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// radians: VF=float16
// Implement radians directly
__attribute__((__overloadable__))
float16 _cl_radians(float16 x0)
{
  typedef int iscalar_t;
  typedef int jscalar_t;
  typedef int kscalar_t;
  typedef float scalar_t;
  typedef int16 ivector_t;
  typedef int16 jvector_t;
  typedef int16 kvector_t;
  typedef float16 vector_t;
#define convert_ivector_t convert_int16
#define convert_jvector_t convert_int16
#define convert_kvector_t convert_int16
#define convert_vector_t convert_float16
  return TYPED_CONST(scalar_t, M_PI_H, M_PI_F, M_PI)/(scalar_t)180*x0;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

#ifdef cl_khr_fp64

// radians: VF=double
// Implement radians directly
__attribute__((__overloadable__))
double _cl_radians(double x0)
{
  typedef long iscalar_t;
  typedef int jscalar_t;
  typedef int kscalar_t;
  typedef double scalar_t;
  typedef long ivector_t;
  typedef int jvector_t;
  typedef int kvector_t;
  typedef double vector_t;
#define convert_ivector_t convert_long
#define convert_jvector_t convert_int
#define convert_kvector_t convert_int
#define convert_vector_t convert_double
  return TYPED_CONST(scalar_t, M_PI_H, M_PI_F, M_PI)/(scalar_t)180*x0;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// radians: VF=double2
// Implement radians directly
__attribute__((__overloadable__))
double2 _cl_radians(double2 x0)
{
  typedef long iscalar_t;
  typedef long jscalar_t;
  typedef int kscalar_t;
  typedef double scalar_t;
  typedef long2 ivector_t;
  typedef long2 jvector_t;
  typedef int2 kvector_t;
  typedef double2 vector_t;
#define convert_ivector_t convert_long2
#define convert_jvector_t convert_long2
#define convert_kvector_t convert_int2
#define convert_vector_t convert_double2
  return TYPED_CONST(scalar_t, M_PI_H, M_PI_F, M_PI)/(scalar_t)180*x0;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// radians: VF=double3
// Implement radians directly
__attribute__((__overloadable__))
double3 _cl_radians(double3 x0)
{
  typedef long iscalar_t;
  typedef long jscalar_t;
  typedef int kscalar_t;
  typedef double scalar_t;
  typedef long3 ivector_t;
  typedef long3 jvector_t;
  typedef int3 kvector_t;
  typedef double3 vector_t;
#define convert_ivector_t convert_long3
#define convert_jvector_t convert_long3
#define convert_kvector_t convert_int3
#define convert_vector_t convert_double3
  return TYPED_CONST(scalar_t, M_PI_H, M_PI_F, M_PI)/(scalar_t)180*x0;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// radians: VF=double4
// Implement radians directly
__attribute__((__overloadable__))
double4 _cl_radians(double4 x0)
{
  typedef long iscalar_t;
  typedef long jscalar_t;
  typedef int kscalar_t;
  typedef double scalar_t;
  typedef long4 ivector_t;
  typedef long4 jvector_t;
  typedef int4 kvector_t;
  typedef double4 vector_t;
#define convert_ivector_t convert_long4
#define convert_jvector_t convert_long4
#define convert_kvector_t convert_int4
#define convert_vector_t convert_double4
  return TYPED_CONST(scalar_t, M_PI_H, M_PI_F, M_PI)/(scalar_t)180*x0;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// radians: VF=double8
// Implement radians directly
__attribute__((__overloadable__))
double8 _cl_radians(double8 x0)
{
  typedef long iscalar_t;
  typedef long jscalar_t;
  typedef int kscalar_t;
  typedef double scalar_t;
  typedef long8 ivector_t;
  typedef long8 jvector_t;
  typedef int8 kvector_t;
  typedef double8 vector_t;
#define convert_ivector_t convert_long8
#define convert_jvector_t convert_long8
#define convert_kvector_t convert_int8
#define convert_vector_t convert_double8
  return TYPED_CONST(scalar_t, M_PI_H, M_PI_F, M_PI)/(scalar_t)180*x0;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// radians: VF=double16
// Implement radians directly
__attribute__((__overloadable__))
double16 _cl_radians(double16 x0)
{
  typedef long iscalar_t;
  typedef long jscalar_t;
  typedef int kscalar_t;
  typedef double scalar_t;
  typedef long16 ivector_t;
  typedef long16 jvector_t;
  typedef int16 kvector_t;
  typedef double16 vector_t;
#define convert_ivector_t convert_long16
#define convert_jvector_t convert_long16
#define convert_kvector_t convert_int16
#define convert_vector_t convert_double16
  return TYPED_CONST(scalar_t, M_PI_H, M_PI_F, M_PI)/(scalar_t)180*x0;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

#endif // #ifdef cl_khr_fp64

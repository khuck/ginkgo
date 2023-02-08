/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_HIP_BASE_TYPES_HIP_HPP_
#define GKO_HIP_BASE_TYPES_HIP_HPP_


#include <ginkgo/core/base/types.hpp>


#include <type_traits>


#include <hip/hip_complex.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#if HIP_VERSION >= 50200000
#include <hipblas/hipblas.h>
#else
#include <hipblas.h>
#endif
#include <thrust/complex.h>


#include <ginkgo/core/base/half.hpp>
#include <ginkgo/core/base/matrix_data.hpp>


// thrust calls the c function not the function from std
// Maybe override the function from thrust directlry
__device__ __forceinline__ __half hypot(__half a, __half b)
{
    return hypot(static_cast<float>(a), static_cast<float>(b));
}

__device__ __forceinline__ thrust::complex<__half> sqrt(
    thrust::complex<__half> a)
{
    return sqrt(static_cast<thrust::complex<float>>(a));
}

__device__ __forceinline__ thrust::complex<float> sqrt(
    thrust::complex<float> val)
{
    return thrust::sqrt(val);
}
__device__ __forceinline__ thrust::complex<double> sqrt(
    thrust::complex<double> val)
{
    return thrust::sqrt(val);
}

#if GINKGO_HIP_PLATFORM_NVCC && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
__device__ __forceinline__ __half sqrt(__half val)
{
    return sqrt(static_cast<float>(val));
}
#else
__device__ __forceinline__ __half sqrt(__half val) { return hsqrt(val); }
#endif


namespace thrust {


// Dircetly call float versrion from here?
template <>
GKO_ATTRIBUTES GKO_INLINE __half abs<__half>(const complex<__half>& z)
{
    return hypot(static_cast<float>(z.real()), static_cast<float>(z.imag()));
}


}  // namespace thrust

#define THRUST_HALF_FRIEND_OPERATOR(_op, _opeq)                               \
    GKO_ATTRIBUTES GKO_INLINE thrust::complex<__half> operator _op(           \
        const thrust::complex<__half> lhs, const thrust::complex<__half> rhs) \
    {                                                                         \
        return thrust::complex<float>{lhs} + thrust::complex<float>(rhs);     \
    }

THRUST_HALF_FRIEND_OPERATOR(+, +=)
THRUST_HALF_FRIEND_OPERATOR(-, -=)
THRUST_HALF_FRIEND_OPERATOR(*, *=)
THRUST_HALF_FRIEND_OPERATOR(/, /=)


namespace gko {
#if GINKGO_HIP_PLATFORM_NVCC
// from the cuda_fp16.hpp
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
__device__ __forceinline__ bool is_nan(const __half& val)
{
    return __hisnan(val);
}

#if CUDA_VERSION >= 10020
__device__ __forceinline__ __half abs(const __half& val) { return __habs(val); }
#else
__device__ __forceinline__ __half abs(const __half& val)
{
    return abs(static_cast<float>(val));
}
#endif
#else
__device__ __forceinline__ bool is_nan(const __half& val)
{
    return is_nan(static_cast<float>(val));
}

__device__ __forceinline__ __half abs(const __half& val)
{
    return abs(static_cast<float>(val));
}
#endif

#else  // Not nvidia device
__device__ __forceinline__ bool is_nan(const __half& val)
{
    return __hisnan(val);
}

// rocm40 __habs is not constexpr
__device__ __forceinline__ __half abs(const __half& val) { return __habs(val); }

#endif


namespace kernels {
namespace hip {
namespace detail {


/**
 * @internal
 *
 * replacement for thrust::complex without alignment restrictions.
 */
template <typename T>
struct alignas(std::complex<T>) fake_complex {
    T real;
    T imag;

    GKO_INLINE GKO_ATTRIBUTES constexpr fake_complex() : real{}, imag{} {}

    GKO_INLINE GKO_ATTRIBUTES constexpr fake_complex(thrust::complex<T> val)
        : real{val.real()}, imag{val.imag()}
    {}

    friend GKO_INLINE GKO_ATTRIBUTES fake_complex operator+(fake_complex a,
                                                            fake_complex b)
    {
        fake_complex result{};
        result.real = a.real + b.real;
        result.imag = a.imag + b.imag;
        return result;
    }

    friend bool GKO_INLINE GKO_ATTRIBUTES constexpr operator==(fake_complex a,
                                                               fake_complex b)
    {
        return a.real == b.real && a.imag == b.imag;
    }

    friend bool GKO_INLINE GKO_ATTRIBUTES constexpr operator!=(fake_complex a,
                                                               fake_complex b)
    {
        return !(a == b);
    }
};


template <typename ValueType>
struct fake_complex_unpack_impl {
    using type = ValueType;

    GKO_INLINE GKO_ATTRIBUTES static constexpr ValueType unpack(ValueType v)
    {
        return v;
    }
};

template <typename ValueType>
struct fake_complex_unpack_impl<fake_complex<ValueType>> {
    using type = thrust::complex<ValueType>;

    GKO_INLINE GKO_ATTRIBUTES static constexpr thrust::complex<ValueType>
    unpack(fake_complex<ValueType> v)
    {
        return {v.real, v.imag};
    }
};


template <typename T>
struct hiplibs_type_impl {
    using type = T;
};

template <typename T>
struct hiplibs_type_impl<T*> {
    using type = typename hiplibs_type_impl<T>::type*;
};

template <typename T>
struct hiplibs_type_impl<T&> {
    using type = typename hiplibs_type_impl<T>::type&;
};

template <typename T>
struct hiplibs_type_impl<const T> {
    using type = const typename hiplibs_type_impl<T>::type;
};

template <typename T>
struct hiplibs_type_impl<volatile T> {
    using type = volatile typename hiplibs_type_impl<T>::type;
};

template <>
struct hiplibs_type_impl<std::complex<float>> {
    using type = hipComplex;
};

template <>
struct hiplibs_type_impl<std::complex<double>> {
    using type = hipDoubleComplex;
};

template <>
struct hiplibs_type_impl<half> {
    using type = __half;
};

template <>
struct hiplibs_type_impl<std::complex<half>> {
    using type = __half2;
};


template <typename T>
struct hiplibs_type_impl<thrust::complex<T>> {
    using type = typename hiplibs_type_impl<std::complex<T>>::type;
};


template <typename T>
struct hipblas_type_impl {
    using type = T;
};

template <typename T>
struct hipblas_type_impl<T*> {
    using type = typename hipblas_type_impl<T>::type*;
};

template <typename T>
struct hipblas_type_impl<T&> {
    using type = typename hipblas_type_impl<T>::type&;
};

template <typename T>
struct hipblas_type_impl<const T> {
    using type = const typename hipblas_type_impl<T>::type;
};

template <typename T>
struct hipblas_type_impl<volatile T> {
    using type = volatile typename hipblas_type_impl<T>::type;
};

template <>
struct hipblas_type_impl<std::complex<float>> {
    using type = hipblasComplex;
};

template <>
struct hipblas_type_impl<std::complex<double>> {
    using type = hipblasDoubleComplex;
};

template <typename T>
struct hipblas_type_impl<thrust::complex<T>> {
    using type = typename hipblas_type_impl<std::complex<T>>::type;
};


template <typename T>
struct hip_type_impl {
    using type = T;
};

template <typename T>
struct hip_type_impl<T*> {
    using type = typename hip_type_impl<T>::type*;
};

template <typename T>
struct hip_type_impl<T&> {
    using type = typename hip_type_impl<T>::type&;
};

template <typename T>
struct hip_type_impl<const T> {
    using type = const typename hip_type_impl<T>::type;
};

template <typename T>
struct hip_type_impl<volatile T> {
    using type = volatile typename hip_type_impl<T>::type;
};

template <>
struct hip_type_impl<gko::half> {
    using type = __half;
};

template <typename T>
struct hip_type_impl<std::complex<T>> {
    using type = thrust::complex<typename hip_type_impl<T>::type>;
};

template <>
struct hip_type_impl<hipDoubleComplex> {
    using type = thrust::complex<double>;
};

template <>
struct hip_type_impl<hipComplex> {
    using type = thrust::complex<float>;
};

template <>
struct hip_type_impl<__half2> {
    using type = thrust::complex<__half>;
};

template <typename T>
struct hip_struct_member_type_impl {
    using type = T;
};

template <typename T>
struct hip_struct_member_type_impl<std::complex<T>> {
    using type = fake_complex<typename hip_struct_member_type_impl<T>::type>;
};

template <>
struct hip_struct_member_type_impl<gko::half> {
    using type = __half;
};

template <typename ValueType, typename IndexType>
struct hip_type_impl<matrix_data_entry<ValueType, IndexType>> {
    using type =
        matrix_data_entry<typename hip_struct_member_type_impl<ValueType>::type,
                          IndexType>;
};

template <typename T>
constexpr hipblasDatatype_t hip_data_type_impl()
{
    return HIPBLAS_C_16F;
}

template <>
constexpr hipblasDatatype_t hip_data_type_impl<float16>()
{
    return HIPBLAS_R_16F;
}

template <>
constexpr hipblasDatatype_t hip_data_type_impl<float>()
{
    return HIPBLAS_R_32F;
}

template <>
constexpr hipblasDatatype_t hip_data_type_impl<double>()
{
    return HIPBLAS_R_64F;
}

template <>
constexpr hipblasDatatype_t hip_data_type_impl<std::complex<float>>()
{
    return HIPBLAS_C_32F;
}

template <>
constexpr hipblasDatatype_t hip_data_type_impl<std::complex<double>>()
{
    return HIPBLAS_C_64F;
}


}  // namespace detail


/**
 * This is an alias for the `hipblasDataType_t` equivalent of `T`. By default,
 * HIPBLAS_C_8U (which is unsupported by C++) is returned.
 *
 * @tparam T  a type
 *
 * @returns the actual `hipblasDatatype_t`
 */
template <typename T>
constexpr hipblasDatatype_t hip_data_type()
{
    return detail::hip_data_type_impl<T>();
}


/**
 * This is an alias for HIP's equivalent of `T`.
 *
 * @tparam T  a type
 */
template <typename T>
using hip_type = typename detail::hip_type_impl<T>::type;

/**
 * This is an alias for CUDA/HIP's equivalent of `T` depending on the namespace.
 *
 * @tparam T  a type
 */
template <typename T>
using device_type = hip_type<T>;

/**
 * This works equivalently to device_type, except for replacing std::complex by
 * detail::fake_complex to avoid issues with thrust::complex (alignment etc.)
 *
 * @tparam T  a type
 */
template <typename T>
using device_member_type =
    typename detail::hip_struct_member_type_impl<T>::type;


/**
 * Reinterprets the passed in value as a HIP type.
 *
 * @param val  the value to reinterpret
 *
 * @return `val` reinterpreted to HIP type
 */
template <typename T>
inline std::enable_if_t<
    std::is_pointer<T>::value || std::is_reference<T>::value, hip_type<T>>
as_hip_type(T val)
{
    return reinterpret_cast<hip_type<T>>(val);
}


/**
 * @copydoc as_hip_type()
 */
template <typename T>
inline std::enable_if_t<
    !std::is_pointer<T>::value && !std::is_reference<T>::value, hip_type<T>>
as_hip_type(T val)
{
    return *reinterpret_cast<hip_type<T>*>(&val);
}


/**
 * Reinterprets the passed in value as a CUDA/HIP type depending on the
 * namespace.
 *
 * @param val  the value to reinterpret
 *
 * @return `val` reinterpreted to CUDA/HIP type
 */
template <typename T>
inline device_type<T> as_device_type(T val)
{
    return as_hip_type(val);
}


/**
 * This is an alias for equivalent of type T used in HIP libraries (HIPBLAS,
 * HIPSPARSE, etc.).
 *
 * @tparam T  a type
 */
template <typename T>
using hiplibs_type = typename detail::hiplibs_type_impl<T>::type;


/**
 * Reinterprets the passed in value as an equivalent type used by the HIP
 * libraries.
 *
 * @param val  the value to reinterpret
 *
 * @return `val` reinterpreted to type used by HIP libraries
 */
template <typename T>
inline hiplibs_type<T> as_hiplibs_type(T val)
{
    return reinterpret_cast<hiplibs_type<T>>(val);
}


/**
 * This is an alias for equivalent of type T used in the HIPBLAS library.
 *
 * @tparam T  a type
 */
template <typename T>
using hipblas_type = typename detail::hipblas_type_impl<T>::type;


/**
 * Reinterprets the passed in value as an equivalent type used by the HIPBLAS
 * library.
 *
 * @param val  the value to reinterpret
 *
 * @return `val` reinterpreted to type used by HIP libraries
 */
template <typename T>
inline hipblas_type<T> as_hipblas_type(T val)
{
    return reinterpret_cast<hipblas_type<T>>(val);
}


/**
 * Casts fake_complex<T> to thrust::complex<T> and leaves any other types
 * unchanged.
 *
 * This is necessary to work around an issue with Thrust shipped in CUDA 9.2,
 * and the fact that thrust::complex has stronger alignment restrictions than
 * std::complex, i.e. structs containing them among other smaller members have
 * different sizes on device and host.
 *
 * @param val  The input value.
 *
 * @return val cast to the correct type.
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr
    typename detail::fake_complex_unpack_impl<T>::type
    fake_complex_unpack(T v)
{
    return detail::fake_complex_unpack_impl<T>::unpack(v);
}


}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_BASE_TYPES_HIP_HPP_

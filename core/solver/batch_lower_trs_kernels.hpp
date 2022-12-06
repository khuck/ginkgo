/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#ifndef GKO_CORE_SOLVER_BATCH_LOWER_TRS_KERNELS_HPP_
#define GKO_CORE_SOLVER_BATCH_LOWER_TRS_KERNELS_HPP_


#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_diagonal.hpp>


#include "core/log/batch_logging.hpp"


namespace gko {
namespace kernels {
namespace batch_lower_trs {


// #if defined GKO_COMPILING_CUDA

// #include "cuda/matrix/batch_struct.hpp"


// namespace device = gko::kernels::cuda;

// template <typename ValueType>
// using DeviceValueType = typename gko::kernels::cuda::cuda_type<ValueType>;


// #elif defined GKO_COMPILING_HIP

// #include "hip/matrix/batch_struct.hip.hpp"


// namespace device = gko::kernels::hip;

// template <typename ValueType>
// using DeviceValueType = gko::kernels::hip::hip_type<ValueType>;


// #elif defined GKO_COMPILING_DPCPP

// #error "Batch solvers are not yet supported on DPC++!"


// namespace device = gko::kernels::dpcpp;

// template <typename ValueType>
// using DeviceValueType = ValueType;


// #else

// #include "reference/matrix/batch_struct.hpp"

// namespace device = gko::kernels::host;

// template <typename ValueType>
// using DeviceValueType = ValueType;

// #endif


// template <typename ValueType>
// void dispatch_on_matrix_type(const BatchLinOp* const sys_mat,
//                              const matrix::BatchDense<ValueType>* const b,
//                              matrix::BatchDense<ValueType>* const x)
// {
//     const auto b_b = device::get_batch_struct(b);
//     const auto x_b = device::get_batch_struct(x);

//     if (auto amat = dynamic_cast<const
//     matrix::BatchCsr<ValueType>*>(sys_mat)) {
//         auto m_b = device::get_batch_struct(amat);
//         call_apply_kernel(m_b, b_b, x_b);

//     } else if (auto amat =
//                    dynamic_cast<const matrix::BatchEll<ValueType>*>(sys_mat))
//                    {
//         auto m_b = device::get_batch_struct(amat);
//         call_apply_kernel(m_b, b_b, x_b);

//     } else if (auto amat = dynamic_cast<const
//     matrix::BatchDense<ValueType>*>(
//                    sys_mat)) {
//         auto m_b = device::get_batch_struct(amat);
//         call_apply_kernel(m_b, b_b, x_b);
//     } else {
//         GKO_NOT_SUPPORTED(sys_mat);
//     }
// }


/**
 * Calculates the amount of in-solver storage needed by batch-lower-trs.
 *
 */
template <typename ValueType>
inline int local_memory_requirement(const int num_rows, const int num_rhs)
{
    return 0;
}

#define GKO_DECLARE_BATCH_LOWER_TRS_APPLY_KERNEL(_type)     \
    void apply(std::shared_ptr<const DefaultExecutor> exec, \
               const BatchLinOp* const sys_mat,             \
               const matrix::BatchDense<_type>* const b,    \
               matrix::BatchDense<_type>* const x)


#define GKO_DECLARE_ALL_AS_TEMPLATES \
    template <typename ValueType>    \
    GKO_DECLARE_BATCH_LOWER_TRS_APPLY_KERNEL(ValueType)


}  // namespace batch_lower_trs


namespace omp {
namespace batch_lower_trs {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_lower_trs
}  // namespace omp


namespace cuda {
namespace batch_lower_trs {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_lower_trs
}  // namespace cuda


namespace reference {
namespace batch_lower_trs {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_lower_trs
}  // namespace reference


namespace hip {
namespace batch_lower_trs {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_lower_trs
}  // namespace hip


namespace dpcpp {
namespace batch_lower_trs {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_lower_trs
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_BATCH_LOWER_TRS_KERNELS_HPP_
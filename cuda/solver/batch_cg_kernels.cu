/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include "core/solver/batch_cg_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "cuda/base/config.hpp"
#include "cuda/base/exception.cuh"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace cuda {

constexpr int default_block_size = 256;
constexpr int sm_multiplier = 4;

/**
 * @brief The batch Cg solver namespace.
 *
 * @ingroup batch_cg
 */
namespace batch_cg {


#include "common/cuda_hip/components/uninitialized_array.hpp.inc"
// include all depedencies (note: do not remove this comment)
#include "common/cuda_hip/components/reduction.hpp.inc"
#include "common/cuda_hip/log/batch_logger.hpp.inc"
#include "common/cuda_hip/matrix/batch_csr_kernels.hpp.inc"
#include "common/cuda_hip/matrix/batch_ell_kernels.hpp.inc"
// TODO: remove batch dense include
#include "common/cuda_hip/matrix/batch_dense_kernels.hpp.inc"
#include "common/cuda_hip/matrix/batch_vector_kernels.hpp.inc"
#include "common/cuda_hip/preconditioner/batch_identity.hpp.inc"
#include "common/cuda_hip/preconditioner/batch_jacobi.hpp.inc"
#include "common/cuda_hip/solver/batch_cg_kernels.hpp.inc"
#include "common/cuda_hip/stop/batch_criteria.hpp.inc"


template <typename T>
using BatchCgOptions = gko::kernels::batch_cg::BatchCgOptions<T>;


template <typename CuValueType>
class KernelCaller {
public:
    using value_type = CuValueType;

    KernelCaller(std::shared_ptr<const CudaExecutor> exec,
                 const BatchCgOptions<remove_complex<value_type>> opts)
        : exec_{exec}, opts_{opts}
    {}

    template <typename BatchMatrixType, typename PrecType, typename StopType,
              typename LogType>
    void call_kernel(LogType logger, const BatchMatrixType& a,
                     const gko::batch_dense::UniformBatch<const value_type>& b,
                     const gko::batch_dense::UniformBatch<value_type>& x) const
    {
        using real_type = gko::remove_complex<value_type>;
        const size_type nbatch = a.num_batch;

        const int shared_size =
            gko::kernels::batch_cg::local_memory_requirement<value_type>(
                a.num_rows, b.num_rhs) +
            PrecType::dynamic_work_size(a.num_rows, a.num_nnz) *
                sizeof(value_type);

        apply_kernel<StopType><<<nbatch, default_block_size, shared_size>>>(
            opts_.max_its, opts_.residual_tol, logger, PrecType(), a, b.values,
            x.values);

        GKO_CUDA_LAST_IF_ERROR_THROW;
    }

private:
    std::shared_ptr<const CudaExecutor> exec_;
    const BatchCgOptions<remove_complex<value_type>> opts_;
};

#include "core/solver/batch_dispatch.hpp.inc"

template <typename ValueType>
void apply(std::shared_ptr<const CudaExecutor> exec,
           const BatchCgOptions<remove_complex<ValueType>>& opts,
           const BatchLinOp* const a,
           const matrix::BatchDense<ValueType>* const b,
           matrix::BatchDense<ValueType>* const x,
           log::BatchLogData<ValueType>& logdata)
{
    using cu_value_type = cuda_type<ValueType>;
    auto dispatcher = create_dispatcher<ValueType, cu_value_type>(
        KernelCaller<cu_value_type>(exec, opts), exec, opts);
    dispatcher.apply(a, b, x, logdata);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_CG_APPLY_KERNEL);


}  // namespace batch_cg
}  // namespace cuda
}  // namespace kernels
}  // namespace gko

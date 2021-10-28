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

#include "core/solver/batch_richardson_kernels.hpp"


#include "core/matrix/batch_struct.hpp"
#include "reference/base/config.hpp"
#include "reference/log/batch_logger.hpp"
// include device kernels for every matrix and preconditioner type
#include "reference/matrix/batch_csr_kernels.hpp"
#include "reference/matrix/batch_dense_kernels.hpp"
//#include "reference/matrix/batch_struct.hpp"
#include "reference/preconditioner/batch_identity.hpp"
#include "reference/preconditioner/batch_jacobi.hpp"
#include "reference/stop/batch_criteria.hpp"


namespace gko {
namespace kernels {
namespace reference {


/**
 * @brief The batch Richardson solver namespace.
 *
 * @ingroup batch_rich
 */
namespace batch_rich {


namespace {


using gko::kernels::reference::batch_csr::advanced_spmv_kernel;
using gko::kernels::reference::batch_csr::spmv_kernel;


#include "reference/solver/batch_richardson_kernels.hpp.inc"


}  // unnamed namespace


template <typename T>
using BatchRichardsonOptions =
    gko::kernels::batch_rich::BatchRichardsonOptions<T>;

template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
static void apply_impl(
    std::shared_ptr<const ReferenceExecutor> exec,
    const BatchRichardsonOptions<remove_complex<ValueType>>& opts,
    LogType logger, PrecType prec, const BatchMatrixType& a,
    const gko::batch_dense::UniformBatch<const ValueType>& b,
    const gko::batch_dense::UniformBatch<ValueType>& x)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const size_type nbatch = a.num_batch;
    const auto nrows = a.num_rows;
    const auto nrhs = b.num_rhs;

    GKO_ASSERT((nrhs == x.num_rhs));

    const int local_size_bytes =
        gko::kernels::batch_rich::local_memory_requirement<ValueType>(nrows,
                                                                      nrhs) +
        PrecType::dynamic_work_size(nrows, a.num_nnz) * sizeof(ValueType);
    using byte = unsigned char;
    Array<byte> local_space(exec, local_size_bytes);

    for (size_type ibatch = 0; ibatch < nbatch; ibatch++) {
        batch_entry_richardson_impl<StopType>(opts, logger, prec, a, b, x,
                                              ibatch, local_space);
    }
}

template <typename BatchType, typename LoggerType, typename ValueType>
void apply_select_prec(
    std::shared_ptr<const ReferenceExecutor> exec,
    const BatchRichardsonOptions<remove_complex<ValueType>>& opts,
    const LoggerType logger, const BatchType& a,
    const gko::batch_dense::UniformBatch<const ValueType>& b,
    const gko::batch_dense::UniformBatch<ValueType>& x)
{
    if (opts.preconditioner == gko::preconditioner::batch::type::none) {
        BatchIdentity<ValueType> prec;
        if (opts.tol_type == gko::stop::batch::ToleranceType::absolute) {
            apply_impl<stop::SimpleAbsResidual<ValueType>>(exec, opts, logger,
                                                           prec, a, b, x);
        } else {
            apply_impl<stop::SimpleRelResidual<ValueType>>(exec, opts, logger,
                                                           prec, a, b, x);
        }
    } else if (opts.preconditioner ==
               gko::preconditioner::batch::type::jacobi) {
        BatchJacobi<ValueType> prec;
        if (opts.tol_type == gko::stop::batch::ToleranceType::absolute) {
            apply_impl<stop::SimpleAbsResidual<ValueType>>(exec, opts, logger,
                                                           prec, a, b, x);
        } else {
            apply_impl<stop::SimpleRelResidual<ValueType>>(exec, opts, logger,
                                                           prec, a, b, x);
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

template <typename ValueType>
void apply(std::shared_ptr<const ReferenceExecutor> exec,
           const BatchRichardsonOptions<remove_complex<ValueType>>& opts,
           const BatchLinOp* const a,
           const matrix::BatchDense<ValueType>* const b,
           matrix::BatchDense<ValueType>* const x,
           gko::log::BatchLogData<ValueType>& logdata)
{
    batch_log::SimpleFinalLogger<remove_complex<ValueType>> logger(
        logdata.res_norms->get_values(), logdata.iter_counts.get_data());

    const gko::batch_dense::UniformBatch<ValueType> x_b =
        host::get_batch_struct(x);
    if (auto a_mat = dynamic_cast<const matrix::BatchCsr<ValueType>*>(a)) {
        const auto a_b = host::get_batch_struct(a_mat);
        const auto b_b = host::get_batch_struct(b);
        apply_select_prec(exec, opts, logger, a_b, b_b, x_b);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_RICHARDSON_APPLY_KERNEL);


}  // namespace batch_rich
}  // namespace reference
}  // namespace kernels
}  // namespace gko

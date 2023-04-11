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

#include "core/solver/batch_band_solver_kernels.hpp"


#include "core/matrix/batch_struct.hpp"
#include "reference/matrix/batch_struct.hpp"

namespace gko {
namespace kernels {
namespace reference {
namespace batch_band_solver {

namespace {

//#include "reference/solver/batch_band_solver_kernels.hpp.inc"

}  // namespace


template <typename ValueType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           const matrix::BatchBand<ValueType>* const band_mat,
           const matrix::BatchDense<ValueType>* const b,
           matrix::BatchDense<ValueType>* const x, const int workspace_size,
           ValueType* const workspace_ptr,
           const enum gko::solver::batch_band_solve_approach approach,
           const int blocked_solve_panel_size) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_band_solver): change the code imported from
// solver/batch_tridiagonal_solver if needed
//    const auto nbatch = tridiag_mat->get_num_batch_entries();
//    const auto nrows = static_cast<int>(tridiag_mat->get_size().at(0)[0]);
//    const auto nrhs = static_cast<int>(rhs->get_size().at(0)[1]);
//    assert(nrhs == 1);
//
//    namespace device = gko::kernels::host;
//    const auto x_batch = device::get_batch_struct(x);
//
//    const int local_size_bytes =
//        gko::kernels::batch_band_solver::local_memory_requirement<
//            ValueType>(nrows, nrhs);
//
//    std::vector<unsigned char> local_space(local_size_bytes);
//
//    const ValueType* const tridiag_mat_subdiags =
//        tridiag_mat->get_const_sub_diagonal();
//    const ValueType* const tridiag_mat_superdiags =
//        tridiag_mat->get_const_super_diagonal();
//
//    assert(workspace_size >=
//           tridiag_mat->get_num_stored_elements_per_diagonal() +
//               rhs->get_num_stored_elements());
//
//    ValueType* const tridiag_mat_maindiags = workspace_ptr;
//    exec->copy(tridiag_mat->get_num_stored_elements_per_diagonal(),
//               tridiag_mat->get_const_main_diagonal(), tridiag_mat_maindiags);
//
//    ValueType* rhs_copy = tridiag_mat_maindiags +
//                          tridiag_mat->get_num_stored_elements_per_diagonal();
//    exec->copy(rhs->get_num_stored_elements(), rhs->get_const_values(),
//               rhs_copy);
//    auto rhs_clone_batch = gko::batch_dense::UniformBatch<ValueType>{
//        rhs_copy, nbatch, rhs->get_stride().at(0), nrows, nrhs, nrows * nrhs};
//
//    for (size_type ibatch = 0; ibatch < nbatch; ibatch++) {
//        batch_entry_tridiagonal_thomas_solve_impl(
//            ibatch, tridiag_mat_subdiags, tridiag_mat_maindiags,
//            tridiag_mat_superdiags, rhs_clone_batch, x_batch,
//            local_space.data());
//    }
//}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BAND_SOLVER_APPLY_KERNEL);


}  // namespace batch_band_solver
}  // namespace reference
}  // namespace kernels
}  // namespace gko

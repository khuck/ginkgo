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

#include "core/matrix/batch_ell_kernels.hpp"


#include <algorithm>
#include <numeric>
#include <utility>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/base/allocator.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/components/prefix_sum.hpp"
#include "omp/components/format_conversion.hpp"
#include "reference/matrix/batch_ell_kernels.hpp"
#include "reference/matrix/batch_struct.hpp"

namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The Compressed sparse row matrix format namespace.
 *
 * @ingroup batch_ell
 */
namespace batch_ell {


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const OmpExecutor> exec,
          const matrix::BatchEll<ValueType, IndexType>* a,
          const matrix::BatchDense<ValueType>* b,
          matrix::BatchDense<ValueType>* c)
{
    const auto a_ub = host::get_batch_struct(a);
    const auto b_ub = host::get_batch_struct(b);
    const auto c_ub = host::get_batch_struct(c);
#pragma omp parallel for
    for (size_type batch = 0; batch < a->get_num_batch_entries(); ++batch) {
        const auto a_b = gko::batch::batch_entry(a_ub, batch);
        const auto b_b = gko::batch::batch_entry(b_ub, batch);
        const auto c_b = gko::batch::batch_entry(c_ub, batch);
        gko::kernels::reference::batch_ell::spmv_kernel(a_b, b_b, c_b);
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const OmpExecutor> exec,
                   const matrix::BatchDense<ValueType>* alpha,
                   const matrix::BatchEll<ValueType, IndexType>* a,
                   const matrix::BatchDense<ValueType>* b,
                   const matrix::BatchDense<ValueType>* beta,
                   matrix::BatchDense<ValueType>* c)
{
    const auto a_ub = host::get_batch_struct(a);
    const auto b_ub = host::get_batch_struct(b);
    const auto c_ub = host::get_batch_struct(c);
    const auto alpha_ub = host::get_batch_struct(alpha);
    const auto beta_ub = host::get_batch_struct(beta);
#pragma omp parallel for
    for (size_type batch = 0; batch < a->get_num_batch_entries(); ++batch) {
        const auto a_b = gko::batch::batch_entry(a_ub, batch);
        const auto b_b = gko::batch::batch_entry(b_ub, batch);
        const auto c_b = gko::batch::batch_entry(c_ub, batch);
        const auto alpha_b = gko::batch::batch_entry(alpha_ub, batch);
        const auto beta_b = gko::batch::batch_entry(beta_ub, batch);
        gko::kernels::reference::batch_ell::advanced_spmv_kernel(
            alpha_b.values[0], a_b, b_b, beta_b.values[0], c_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void batch_scale(std::shared_ptr<const OmpExecutor> exec,
                 const matrix::BatchDense<ValueType>* left_scale,
                 const matrix::BatchDense<ValueType>* right_scale,
                 matrix::BatchEll<ValueType, IndexType>* mat)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_SCALE);


template <typename ValueType, typename IndexType>
void pre_diag_scale_system(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::BatchDense<ValueType>* const left_scale,
    const matrix::BatchDense<ValueType>* const right_scale,
    matrix::BatchEll<ValueType, IndexType>* const a,
    matrix::BatchDense<ValueType>* const b) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_PRE_DIAG_SCALE_SYSTEM);


template <typename IndexType>
void convert_row_ptrs_to_idxs(std::shared_ptr<const OmpExecutor> exec,
                              const IndexType* ptrs, size_type num_rows,
                              IndexType* idxs) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const OmpExecutor> exec,
                      const matrix::BatchEll<ValueType, IndexType>* source,
                      matrix::BatchDense<ValueType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType, typename UnaryOperator>
void transpose_and_transform(std::shared_ptr<const OmpExecutor> exec,
                             matrix::BatchEll<ValueType, IndexType>* trans,
                             const matrix::BatchEll<ValueType, IndexType>* orig,
                             UnaryOperator op) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const OmpExecutor> exec,
               const matrix::BatchEll<ValueType, IndexType>* orig,
               matrix::BatchEll<ValueType, IndexType>* trans)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::BatchEll<ValueType, IndexType>* orig,
                    matrix::BatchEll<ValueType, IndexType>* trans)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_total_cols(std::shared_ptr<const OmpExecutor> exec,
                          const matrix::BatchEll<ValueType, IndexType>* source,
                          size_type* result, size_type stride_factor,
                          size_type slice_size) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_max_nnz_per_row(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::BatchEll<ValueType, IndexType>* source,
    size_type* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::BatchEll<ValueType, IndexType>* source,
    Array<size_type>* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const OmpExecutor> exec,
                          matrix::BatchEll<ValueType, IndexType>* to_sort)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::BatchEll<ValueType, IndexType>* to_check,
    bool* is_sorted) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_IS_SORTED_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void convert_to_batch_dense(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::BatchEll<ValueType, IndexType>* const src,
    matrix::BatchDense<ValueType>* const dest)
{
    const size_type nbatches = src->get_num_batch_entries();
    const int num_rows = src->get_size().at(0)[0];
    const int num_cols = src->get_size().at(0)[1];
    const int num_stored_elements_per_row =
        src->get_num_stored_elements() / src->get_num_batch_entries();
#pragma omp parallel for
    for (size_type ibatch = 0; ibatch < nbatches; ibatch++) {
        for (size_type row = 0; row < num_rows; row++) {
            for (size_type col = 0; col < num_cols; col++) {
                dest->at(ibatch, row, col) = zero<ValueType>();
            }
            for (size_type i = 0; i < num_stored_elements_per_row; i++) {
                dest->at(ibatch, row, src->col_at(row, i)) +=
                    src->val_at(ibatch, row, i);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_CONVERT_TO_BATCH_DENSE);


template <typename ValueType, typename IndexType>
void convert_from_batch_csc(std::shared_ptr<const DefaultExecutor> exec,
                            matrix::BatchEll<ValueType, IndexType>* ell,
                            const Array<ValueType>& values_arr,
                            const Array<IndexType>& row_idxs_arr,
                            const Array<IndexType>& col_ptrs_arr)
{
    const size_type nbatches = ell->get_num_batch_entries();
    const int num_rows = ell->get_size().at(0)[0];
    const int num_cols = ell->get_size().at(0)[1];
    const int num_stored_elements_per_row =
        ell->get_num_stored_elements_per_row().at(0);
    const auto values = values_arr.get_const_data();
    const auto row_idxs = row_idxs_arr.get_const_data();
    const auto col_ptrs = col_ptrs_arr.get_const_data();
    const auto nnz_per_batch =
        static_cast<size_type>(values_arr.get_num_elems() / nbatches);
    std::vector<ValueType> csr_vals(values_arr.get_num_elems());
    std::vector<IndexType> row_ptrs(num_rows + 1, 0);
    std::vector<IndexType> col_idxs(row_idxs_arr.get_num_elems());
    size_type num_nnz = col_ptrs[num_cols];
    row_ptrs[0] = 0;
    convert_unsorted_idxs_to_ptrs(row_idxs, num_nnz, row_ptrs.data() + 1,
                                  num_rows);
    for (size_type col = 0; col < num_cols; ++col) {
        for (auto i = col_ptrs[col]; i < col_ptrs[col + 1]; ++i) {
            const auto dest_idx = (row_ptrs.data() + 1)[row_idxs[i]]++;
            col_idxs[dest_idx] = col;
            size_type offset = 0;
            for (size_type b = 0; b < nbatches; ++b) {
                offset = b * num_nnz;
                csr_vals[offset + dest_idx] = values[offset + i];
            }
        }
    }
    size_type offset = 0;
#pragma omp parallel for
    for (size_type ibatch = 0; ibatch < nbatches; ++ibatch) {
        offset = ibatch * nnz_per_batch;
        for (size_type row = 0; row < num_rows; row++) {
            for (size_type i = 0; i < num_stored_elements_per_row; i++) {
                ell->val_at(ibatch, row, i) = zero<ValueType>();
                ell->col_at(row, i) = 0;
            }
            for (size_type col_idx = 0;
                 col_idx < row_ptrs[row + 1] - row_ptrs[row]; col_idx++) {
                ell->val_at(ibatch, row, col_idx) =
                    csr_vals[offset + row_ptrs[row] + col_idx];
                ell->col_at(row, col_idx) = col_idxs[row_ptrs[row] + col_idx];
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ELL_CONVERT_FROM_BATCH_CSC);


}  // namespace batch_ell
}  // namespace omp
}  // namespace kernels
}  // namespace gko
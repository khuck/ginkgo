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

#ifndef GKO_CORE_MATRIX_BATCH_ELL_KERNELS_HPP_
#define GKO_CORE_MATRIX_BATCH_ELL_KERNELS_HPP_


#include <ginkgo/core/matrix/batch_ell.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/batch_diagonal.hpp>


namespace gko {
namespace kernels {


#define GKO_DECLARE_BATCH_ELL_SPMV_KERNEL(ValueType, IndexType) \
    void spmv(std::shared_ptr<const DefaultExecutor> exec,      \
              const matrix::BatchEll<ValueType, IndexType>* a,  \
              const matrix::BatchDense<ValueType>* b,           \
              matrix::BatchDense<ValueType>* c)

#define GKO_DECLARE_BATCH_ELL_ADVANCED_SPMV_KERNEL(ValueType, IndexType) \
    void advanced_spmv(std::shared_ptr<const DefaultExecutor> exec,      \
                       const matrix::BatchDense<ValueType>* alpha,       \
                       const matrix::BatchEll<ValueType, IndexType>* a,  \
                       const matrix::BatchDense<ValueType>* b,           \
                       const matrix::BatchDense<ValueType>* beta,        \
                       matrix::BatchDense<ValueType>* c)

#define GKO_DECLARE_BATCH_ELL_CONVERT_TO_DENSE_KERNEL(ValueType, IndexType) \
    void convert_to_dense(                                                  \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const matrix::BatchEll<ValueType, IndexType>* source,               \
        matrix::BatchDense<ValueType>* result)

#define GKO_DECLARE_BATCH_ELL_CALCULATE_TOTAL_COLS_KERNEL(ValueType, \
                                                          IndexType) \
    void calculate_total_cols(                                       \
        std::shared_ptr<const DefaultExecutor> exec,                 \
        const matrix::BatchEll<ValueType, IndexType>* source,        \
        size_type* result, size_type stride_factor, size_type slice_size)

#define GKO_DECLARE_BATCH_ELL_TRANSPOSE_KERNEL(ValueType, IndexType)   \
    void transpose(std::shared_ptr<const DefaultExecutor> exec,        \
                   const matrix::BatchEll<ValueType, IndexType>* orig, \
                   matrix::BatchEll<ValueType, IndexType>* trans)

#define GKO_DECLARE_BATCH_ELL_CONJ_TRANSPOSE_KERNEL(ValueType, IndexType)   \
    void conj_transpose(std::shared_ptr<const DefaultExecutor> exec,        \
                        const matrix::BatchEll<ValueType, IndexType>* orig, \
                        matrix::BatchEll<ValueType, IndexType>* trans)

#define GKO_DECLARE_BATCH_ELL_CALCULATE_MAX_NNZ_PER_ROW_KERNEL(ValueType, \
                                                               IndexType) \
    void calculate_max_nnz_per_row(                                       \
        std::shared_ptr<const DefaultExecutor> exec,                      \
        const matrix::BatchEll<ValueType, IndexType>* source,             \
        size_type* result)

#define GKO_DECLARE_BATCH_ELL_CALCULATE_NONZEROS_PER_ROW_KERNEL(ValueType, \
                                                                IndexType) \
    void calculate_nonzeros_per_row(                                       \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        const matrix::BatchEll<ValueType, IndexType>* source,              \
        array<size_type>* result)

#define GKO_DECLARE_BATCH_ELL_SORT_BY_COLUMN_INDEX(ValueType, IndexType)   \
    void sort_by_column_index(std::shared_ptr<const DefaultExecutor> exec, \
                              matrix::BatchEll<ValueType, IndexType>* to_sort)

#define GKO_DECLARE_BATCH_ELL_IS_SORTED_BY_COLUMN_INDEX(ValueType, IndexType) \
    void is_sorted_by_column_index(                                           \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        const matrix::BatchEll<ValueType, IndexType>* to_check,               \
        bool* is_sorted)

#define GKO_DECLARE_BATCH_ELL_SCALE(ValueType, IndexType)                 \
    void batch_scale(std::shared_ptr<const DefaultExecutor> exec,         \
                     const matrix::BatchDiagonal<ValueType>* left_scale,  \
                     const matrix::BatchDiagonal<ValueType>* right_scale, \
                     matrix::BatchEll<ValueType, IndexType>* mat)

#define GKO_DECLARE_BATCH_ELL_PRE_DIAG_SCALE_SYSTEM(ValueType, IndexType) \
    void pre_diag_scale_system(                                           \
        std::shared_ptr<const DefaultExecutor> exec,                      \
        const matrix::BatchDense<ValueType>* left_scale,                  \
        const matrix::BatchDense<ValueType>* right_scale,                 \
        matrix::BatchEll<ValueType, IndexType>* a,                        \
        matrix::BatchDense<ValueType>* b)

#define GKO_DECLARE_BATCH_ELL_CONVERT_TO_BATCH_DENSE(ValueType, IndexType) \
    void convert_to_batch_dense(                                           \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        const matrix::BatchEll<ValueType, IndexType>* csr,                 \
        matrix::BatchDense<ValueType>* dense)

#define GKO_DECLARE_BATCH_ELL_CHECK_DIAGONAL_ENTRIES_EXIST(ValueType, \
                                                           IndexType) \
    void check_diagonal_entries_exist(                                \
        std::shared_ptr<const DefaultExecutor> exec,                  \
        const matrix::BatchEll<ValueType, IndexType>* mtx, bool& all_diags)

#define GKO_DECLARE_BATCH_ELL_ADD_SCALED_IDENTITY_KERNEL(ValueType, IndexType) \
    void add_scaled_identity(std::shared_ptr<const DefaultExecutor> exec,      \
                             const matrix::BatchDense<ValueType>* a,           \
                             const matrix::BatchDense<ValueType>* b,           \
                             matrix::BatchEll<ValueType, IndexType>* mtx)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                          \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_BATCH_ELL_SPMV_KERNEL(ValueType, IndexType);                  \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_BATCH_ELL_ADVANCED_SPMV_KERNEL(ValueType, IndexType);         \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_BATCH_ELL_CONVERT_TO_DENSE_KERNEL(ValueType, IndexType);      \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_BATCH_ELL_CALCULATE_TOTAL_COLS_KERNEL(ValueType, IndexType);  \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_BATCH_ELL_TRANSPOSE_KERNEL(ValueType, IndexType);             \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_BATCH_ELL_CONJ_TRANSPOSE_KERNEL(ValueType, IndexType);        \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_BATCH_ELL_CALCULATE_MAX_NNZ_PER_ROW_KERNEL(ValueType,         \
                                                           IndexType);        \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_BATCH_ELL_CALCULATE_NONZEROS_PER_ROW_KERNEL(ValueType,        \
                                                            IndexType);       \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_BATCH_ELL_SORT_BY_COLUMN_INDEX(ValueType, IndexType);         \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_BATCH_ELL_IS_SORTED_BY_COLUMN_INDEX(ValueType, IndexType);    \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_BATCH_ELL_SCALE(ValueType, IndexType);                        \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_BATCH_ELL_PRE_DIAG_SCALE_SYSTEM(ValueType, IndexType);        \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_BATCH_ELL_CONVERT_TO_BATCH_DENSE(ValueType, IndexType);       \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_BATCH_ELL_CHECK_DIAGONAL_ENTRIES_EXIST(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_BATCH_ELL_ADD_SCALED_IDENTITY_KERNEL(ValueType, IndexType)


namespace omp {
namespace batch_ell {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_ell
}  // namespace omp


namespace cuda {
namespace batch_ell {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_ell
}  // namespace cuda


namespace reference {
namespace batch_ell {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_ell
}  // namespace reference


namespace hip {
namespace batch_ell {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_ell
}  // namespace hip


namespace dpcpp {
namespace batch_ell {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_ell
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_BATCH_ELL_KERNELS_HPP_

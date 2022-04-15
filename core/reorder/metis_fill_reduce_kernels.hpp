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

#ifndef GKO_CORE_REORDER_METIS_FILL_REDUCE_KERNELS_HPP_
#define GKO_CORE_REORDER_METIS_FILL_REDUCE_KERNELS_HPP_


#include <ginkgo/core/reorder/metis_fill_reduce.hpp>


#include <memory>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_METIS_FILL_REDUCE_GET_PERMUTATION_KERNEL(IndexType)   \
    void get_permutation(                                                 \
        std::shared_ptr<const DefaultExecutor> exec,                      \
        const gko::size_type num_vertices, const IndexType* mat_row_ptrs, \
        const IndexType* mat_col_idxs, const IndexType* vertex_weights,   \
        IndexType* permutation, IndexType* inv_permutation)

#define GKO_DECLARE_METIS_FILL_REDUCE_CONSTRUCT_INVERSE_PERMUTATION_KERNEL( \
    ValueType, IndexType)                                                   \
    void construct_inverse_permutation_matrix(                              \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const IndexType* inv_permutation,                                   \
        gko::matrix::Csr<ValueType, IndexType>* inverse_permutation_matrix)

#define GKO_DECLARE_METIS_FILL_REDUCE_CONSTRUCT_PERMUTATION_KERNEL(ValueType, \
                                                                   IndexType) \
    void construct_permutation_matrix(                                        \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        const IndexType* permutation,                                         \
        gko::matrix::Csr<ValueType, IndexType>* permutation_matrix)

#define GKO_DECLARE_METIS_FILL_REDUCE_PERMUTE_KERNEL(ValueType, IndexType)   \
    void permute(std::shared_ptr<const DefaultExecutor> exec,                \
                 gko::matrix::Csr<ValueType, IndexType>* permutation_matrix, \
                 gko::LinOp* to_permute)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                       \
    template <typename IndexType>                                          \
    GKO_DECLARE_METIS_FILL_REDUCE_GET_PERMUTATION_KERNEL(IndexType);       \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_METIS_FILL_REDUCE_CONSTRUCT_PERMUTATION_KERNEL(ValueType,  \
                                                               IndexType); \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_METIS_FILL_REDUCE_CONSTRUCT_INVERSE_PERMUTATION_KERNEL(    \
        ValueType, IndexType);                                             \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_METIS_FILL_REDUCE_PERMUTE_KERNEL(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(metis_fill_reduce,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_REORDER_METIS_FILL_REDUCE_KERNELS_HPP_

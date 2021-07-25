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

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/device_matrix_data_kernels.hpp"
#include "core/base/index_set_kernels.hpp"
#include "core/base/mixed_precision_types.hpp"
#include "core/components/absolute_array_kernels.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/precision_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/components/reduce_array_kernels.hpp"
#include "core/distributed/matrix_kernels.hpp"
#include "core/distributed/partition_kernels.hpp"
#include "core/distributed/vector_kernels.hpp"
#include "core/factorization/cholesky_kernels.hpp"
#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/ic_kernels.hpp"
#include "core/factorization/ilu_kernels.hpp"
#include "core/factorization/lu_kernels.hpp"
#include "core/factorization/par_ic_kernels.hpp"
#include "core/factorization/par_ict_kernels.hpp"
#include "core/factorization/par_ilu_kernels.hpp"
#include "core/factorization/par_ilut_kernels.hpp"
#include "core/matrix/bccoo_kernels.hpp"
#include "core/matrix/coo_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/matrix/diagonal_kernels.hpp"
#include "core/matrix/ell_kernels.hpp"
#include "core/matrix/fbcsr_kernels.hpp"
#include "core/matrix/fft_kernels.hpp"
#include "core/matrix/hybrid_kernels.hpp"
#include "core/matrix/sellp_kernels.hpp"
#include "core/matrix/sparsity_csr_kernels.hpp"
#include "core/multigrid/pgm_kernels.hpp"
#include "core/preconditioner/isai_kernels.hpp"
#include "core/preconditioner/jacobi_kernels.hpp"
#include "core/reorder/rcm_kernels.hpp"
#include "core/solver/bicg_kernels.hpp"
#include "core/solver/bicgstab_kernels.hpp"
#include "core/solver/cb_gmres_kernels.hpp"
#include "core/solver/cg_kernels.hpp"
#include "core/solver/cgs_kernels.hpp"
#include "core/solver/common_gmres_kernels.hpp"
#include "core/solver/fcg_kernels.hpp"
#include "core/solver/gmres_kernels.hpp"
#include "core/solver/idr_kernels.hpp"
#include "core/solver/ir_kernels.hpp"
#include "core/solver/lower_trs_kernels.hpp"
#include "core/solver/multigrid_kernels.hpp"
#include "core/solver/upper_trs_kernels.hpp"
#include "core/stop/criterion_kernels.hpp"
#include "core/stop/residual_norm_kernels.hpp"


#ifndef GKO_HOOK_MODULE
#error "Need to define GKO_HOOK_MODULE variable before including this file"
#endif  // GKO_HOOK_MODULE


#define GKO_STUB(_macro) _macro GKO_NOT_COMPILED(GKO_HOOK_MODULE)

#define GKO_STUB_VALUE_CONVERSION(_macro)                             \
    template <typename SourceType, typename TargetType>               \
    _macro(SourceType, TargetType) GKO_NOT_COMPILED(GKO_HOOK_MODULE); \
    GKO_INSTANTIATE_FOR_EACH_VALUE_CONVERSION(_macro)

#define GKO_STUB_NON_COMPLEX_VALUE_TYPE(_macro)          \
    template <typename ValueType>                        \
    _macro(ValueType) GKO_NOT_COMPILED(GKO_HOOK_MODULE); \
    GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(_macro)

#define GKO_STUB_VALUE_TYPE(_macro)                      \
    template <typename ValueType>                        \
    _macro(ValueType) GKO_NOT_COMPILED(GKO_HOOK_MODULE); \
    GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(_macro)

#define GKO_STUB_VALUE_AND_SCALAR_TYPE(_macro)                       \
    template <typename ValueType, typename ScalarType>               \
    _macro(ValueType, ScalarType) GKO_NOT_COMPILED(GKO_HOOK_MODULE); \
    GKO_INSTANTIATE_FOR_EACH_VALUE_AND_SCALAR_TYPE(_macro)

#define GKO_STUB_INDEX_TYPE(_macro)                      \
    template <typename IndexType>                        \
    _macro(IndexType) GKO_NOT_COMPILED(GKO_HOOK_MODULE); \
    GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(_macro)

#define GKO_STUB_LOCAL_GLOBAL_TYPE(_macro)                                     \
    template <typename LocalIndexType, typename GlobalIndexType>               \
    _macro(LocalIndexType, GlobalIndexType) GKO_NOT_COMPILED(GKO_HOOK_MODULE); \
    GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(_macro)

#define GKO_STUB_NON_COMPLEX_VALUE_AND_INDEX_TYPE(_macro)           \
    template <typename ValueType, typename IndexType>               \
    _macro(ValueType, IndexType) GKO_NOT_COMPILED(GKO_HOOK_MODULE); \
    GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(_macro)

#define GKO_STUB_VALUE_AND_INDEX_TYPE(_macro)                       \
    template <typename ValueType, typename IndexType>               \
    _macro(ValueType, IndexType) GKO_NOT_COMPILED(GKO_HOOK_MODULE); \
    GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(_macro)

#define GKO_STUB_MIXED_VALUE_AND_INDEX_TYPE(_macro)                     \
    template <typename InputValueType, typename MatrixValueType,        \
              typename OutputValueType, typename IndexType>             \
    _macro(InputValueType, MatrixValueType, OutputValueType, IndexType) \
        GKO_NOT_COMPILED(GKO_HOOK_MODULE);                              \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(_macro)

#define GKO_STUB_MIXED_VALUE_AND_INDEX_TYPE_2(_macro)            \
    template <typename InputValueType, typename OutputValueType, \
              typename IndexType>                                \
    _macro(InputValueType, OutputValueType, IndexType)           \
        GKO_NOT_COMPILED(GKO_HOOK_MODULE);                       \
    GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_2(_macro)

#define GKO_STUB_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(_macro) \
    template <typename ValueType, typename LocalIndexType, \
              typename GlobalIndexType>                    \
    _macro(ValueType, LocalIndexType, GlobalIndexType)     \
        GKO_NOT_COMPILED(GKO_HOOK_MODULE);                 \
    GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(_macro)

#define GKO_STUB_TEMPLATE_TYPE(_macro)                   \
    template <typename IndexType>                        \
    _macro(IndexType) GKO_NOT_COMPILED(GKO_HOOK_MODULE); \
    GKO_INSTANTIATE_FOR_EACH_TEMPLATE_TYPE(_macro)

#define GKO_STUB_VALUE_CONVERSION(_macro)                             \
    template <typename SourceType, typename TargetType>               \
    _macro(SourceType, TargetType) GKO_NOT_COMPILED(GKO_HOOK_MODULE); \
    GKO_INSTANTIATE_FOR_EACH_VALUE_CONVERSION(_macro)

#define GKO_STUB_VALUE_CONVERSION_OR_COPY(_macro)                     \
    template <typename SourceType, typename TargetType>               \
    _macro(SourceType, TargetType) GKO_NOT_COMPILED(GKO_HOOK_MODULE); \
    GKO_INSTANTIATE_FOR_EACH_VALUE_CONVERSION_OR_COPY(_macro)

#define GKO_STUB_CB_GMRES(_macro)                                              \
    template <typename ValueType, typename ValueTypeKrylovBases>               \
    _macro(ValueType, ValueTypeKrylovBases) GKO_NOT_COMPILED(GKO_HOOK_MODULE); \
    GKO_INSTANTIATE_FOR_EACH_CB_GMRES_TYPE(_macro)

#define GKO_STUB_CB_GMRES_CONST(_macro)                                        \
    template <typename ValueType, typename ValueTypeKrylovBases>               \
    _macro(ValueType, ValueTypeKrylovBases) GKO_NOT_COMPILED(GKO_HOOK_MODULE); \
    GKO_INSTANTIATE_FOR_EACH_CB_GMRES_CONST_TYPE(_macro)

namespace gko {
namespace kernels {
namespace GKO_HOOK_MODULE {
namespace components {


GKO_STUB_VALUE_CONVERSION(GKO_DECLARE_CONVERT_PRECISION_KERNEL);
GKO_STUB_INDEX_TYPE(GKO_DECLARE_PREFIX_SUM_KERNEL);
// explicitly instantiate for size_type, as this is
// used in the SellP format
template GKO_DECLARE_PREFIX_SUM_KERNEL(size_type);

GKO_STUB_TEMPLATE_TYPE(GKO_DECLARE_FILL_ARRAY_KERNEL);
template GKO_DECLARE_FILL_ARRAY_KERNEL(bool);

GKO_STUB_TEMPLATE_TYPE(GKO_DECLARE_FILL_SEQ_ARRAY_KERNEL);
GKO_STUB_TEMPLATE_TYPE(GKO_DECLARE_REDUCE_ADD_ARRAY_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_INPLACE_ABSOLUTE_ARRAY_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_OUTPLACE_ABSOLUTE_ARRAY_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_REMOVE_ZEROS_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_SUM_DUPLICATES_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_SORT_ROW_MAJOR_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DEVICE_MATRIX_DATA_AOS_TO_SOA_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DEVICE_MATRIX_DATA_SOA_TO_AOS_KERNEL);

template <typename IndexType, typename RowPtrType>
GKO_DECLARE_CONVERT_PTRS_TO_IDXS(IndexType, RowPtrType)
GKO_NOT_COMPILED(GKO_HOOK_MODULE);
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CONVERT_PTRS_TO_IDXS32);
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CONVERT_PTRS_TO_IDXS64);
template <typename IndexType, typename RowPtrType>
GKO_DECLARE_CONVERT_IDXS_TO_PTRS(IndexType, RowPtrType)
GKO_NOT_COMPILED(GKO_HOOK_MODULE);
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CONVERT_IDXS_TO_PTRS32);
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CONVERT_IDXS_TO_PTRS64);
template <typename RowPtrType>
GKO_DECLARE_CONVERT_PTRS_TO_SIZES(RowPtrType)
GKO_NOT_COMPILED(GKO_HOOK_MODULE);
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CONVERT_PTRS_TO_SIZES);


}  // namespace components


namespace idx_set {


GKO_STUB_INDEX_TYPE(GKO_DECLARE_INDEX_SET_COMPUTE_VALIDITY_KERNEL);
GKO_STUB_INDEX_TYPE(GKO_DECLARE_INDEX_SET_TO_GLOBAL_INDICES_KERNEL);
GKO_STUB_INDEX_TYPE(GKO_DECLARE_INDEX_SET_POPULATE_KERNEL);
GKO_STUB_INDEX_TYPE(GKO_DECLARE_INDEX_SET_GLOBAL_TO_LOCAL_KERNEL);
GKO_STUB_INDEX_TYPE(GKO_DECLARE_INDEX_SET_LOCAL_TO_GLOBAL_KERNEL);


}  // namespace idx_set


namespace partition {


GKO_STUB(GKO_PARTITION_COUNT_RANGES);
GKO_STUB_INDEX_TYPE(GKO_PARTITION_BUILD_FROM_CONTIGUOUS);
GKO_STUB_INDEX_TYPE(GKO_PARTITION_BUILD_FROM_MAPPING);
GKO_STUB_INDEX_TYPE(GKO_PARTITION_BUILD_FROM_GLOBAL_SIZE);
GKO_STUB_LOCAL_GLOBAL_TYPE(GKO_DECLARE_PARTITION_BUILD_STARTING_INDICES);
GKO_STUB_LOCAL_GLOBAL_TYPE(GKO_DECLARE_PARTITION_IS_ORDERED);


}  // namespace partition


namespace distributed_vector {


GKO_STUB_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_DISTRIBUTED_VECTOR_BUILD_LOCAL);


}

namespace distributed_matrix {


GKO_STUB_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(GKO_DECLARE_BUILD_LOCAL_NONLOCAL);


}  // namespace distributed_matrix


namespace dense {


GKO_STUB_VALUE_TYPE(GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_DENSE_APPLY_KERNEL);
GKO_STUB_VALUE_CONVERSION_OR_COPY(GKO_DECLARE_DENSE_COPY_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_DENSE_FILL_KERNEL);
GKO_STUB_VALUE_AND_SCALAR_TYPE(GKO_DECLARE_DENSE_SCALE_KERNEL);
GKO_STUB_VALUE_AND_SCALAR_TYPE(GKO_DECLARE_DENSE_INV_SCALE_KERNEL);
GKO_STUB_VALUE_AND_SCALAR_TYPE(GKO_DECLARE_DENSE_ADD_SCALED_KERNEL);
GKO_STUB_VALUE_AND_SCALAR_TYPE(GKO_DECLARE_DENSE_SUB_SCALED_KERNEL);
GKO_STUB_VALUE_AND_SCALAR_TYPE(GKO_DECLARE_DENSE_ADD_SCALED_IDENTITY_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_DENSE_ADD_SCALED_DIAG_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_DENSE_SUB_SCALED_DIAG_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_DOT_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_DOT_DISPATCH_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_CONJ_DOT_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_CONJ_DOT_DISPATCH_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_NORM2_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_NORM2_DISPATCH_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_NORM1_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_SQUARED_NORM2_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_SQRT_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DENSE_FILL_IN_MATRIX_DATA_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DENSE_CONVERT_TO_COO_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DENSE_CONVERT_TO_CSR_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DENSE_CONVERT_TO_ELL_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DENSE_CONVERT_TO_FBCSR_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DENSE_CONVERT_TO_HYBRID_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DENSE_CONVERT_TO_SELLP_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DENSE_CONVERT_TO_SPARSITY_CSR_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_MAX_NNZ_PER_ROW_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_SLICE_SETS_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DENSE_COUNT_NONZEROS_PER_ROW_KERNEL);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_COUNT_NONZEROS_PER_ROW_KERNEL_SIZE_T);
GKO_STUB_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_COUNT_NONZERO_BLOCKS_PER_ROW_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_DENSE_TRANSPOSE_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_DENSE_CONJ_TRANSPOSE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DENSE_SYMM_PERMUTE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DENSE_INV_SYMM_PERMUTE_KERNEL);
GKO_STUB_MIXED_VALUE_AND_INDEX_TYPE_2(GKO_DECLARE_DENSE_ROW_GATHER_KERNEL);
GKO_STUB_MIXED_VALUE_AND_INDEX_TYPE_2(
    GKO_DECLARE_DENSE_ADVANCED_ROW_GATHER_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DENSE_COLUMN_PERMUTE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DENSE_INV_ROW_PERMUTE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DENSE_INV_COLUMN_PERMUTE_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_DENSE_EXTRACT_DIAGONAL_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_INPLACE_ABSOLUTE_DENSE_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_OUTPLACE_ABSOLUTE_DENSE_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_MAKE_COMPLEX_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_GET_REAL_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_GET_IMAG_KERNEL);


}  // namespace dense


namespace diagonal {


GKO_STUB_VALUE_TYPE(GKO_DECLARE_DIAGONAL_APPLY_TO_DENSE_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_DIAGONAL_RIGHT_APPLY_TO_DENSE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DIAGONAL_APPLY_TO_CSR_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DIAGONAL_RIGHT_APPLY_TO_CSR_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DIAGONAL_CONVERT_TO_CSR_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_DIAGONAL_CONJ_TRANSPOSE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DIAGONAL_FILL_IN_MATRIX_DATA_KERNEL);


}  // namespace diagonal


namespace cg {


GKO_STUB_VALUE_TYPE(GKO_DECLARE_CG_INITIALIZE_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_CG_STEP_1_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_CG_STEP_2_KERNEL);


}  // namespace cg


namespace bicg {


GKO_STUB_VALUE_TYPE(GKO_DECLARE_BICG_INITIALIZE_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_BICG_STEP_1_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_BICG_STEP_2_KERNEL);


}  // namespace bicg


namespace lower_trs {


GKO_STUB(GKO_DECLARE_LOWER_TRS_SHOULD_PERFORM_TRANSPOSE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LOWER_TRS_GENERATE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LOWER_TRS_SOLVE_KERNEL);


}  // namespace lower_trs


namespace upper_trs {


GKO_STUB(GKO_DECLARE_UPPER_TRS_SHOULD_PERFORM_TRANSPOSE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_UPPER_TRS_GENERATE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_UPPER_TRS_SOLVE_KERNEL);


}  // namespace upper_trs


namespace fcg {


GKO_STUB_VALUE_TYPE(GKO_DECLARE_FCG_INITIALIZE_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_FCG_STEP_1_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_FCG_STEP_2_KERNEL);


}  // namespace fcg


namespace bicgstab {


GKO_STUB_VALUE_TYPE(GKO_DECLARE_BICGSTAB_INITIALIZE_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_1_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_2_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_3_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_BICGSTAB_FINALIZE_KERNEL);


}  // namespace bicgstab


namespace idr {


GKO_STUB_VALUE_TYPE(GKO_DECLARE_IDR_INITIALIZE_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_IDR_STEP_1_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_IDR_STEP_2_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_IDR_STEP_3_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_IDR_COMPUTE_OMEGA_KERNEL);


}  // namespace idr


namespace cgs {


GKO_STUB_VALUE_TYPE(GKO_DECLARE_CGS_INITIALIZE_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_CGS_STEP_1_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_CGS_STEP_2_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_CGS_STEP_3_KERNEL);


}  // namespace cgs


namespace common_gmres {


GKO_STUB_VALUE_TYPE(GKO_DECLARE_COMMON_GMRES_INITIALIZE_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_COMMON_GMRES_HESSENBERG_QR_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_COMMON_GMRES_SOLVE_KRYLOV_KERNEL);


}  // namespace common_gmres


namespace gmres {


GKO_STUB_VALUE_TYPE(GKO_DECLARE_GMRES_RESTART_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_GMRES_MULTI_AXPY_KERNEL);


}  // namespace gmres


namespace cb_gmres {


GKO_STUB_VALUE_TYPE(GKO_DECLARE_CB_GMRES_INITIALIZE_KERNEL);
GKO_STUB_CB_GMRES(GKO_DECLARE_CB_GMRES_RESTART_KERNEL);
GKO_STUB_CB_GMRES(GKO_DECLARE_CB_GMRES_ARNOLDI_KERNEL);
GKO_STUB_CB_GMRES_CONST(GKO_DECLARE_CB_GMRES_SOLVE_KRYLOV_KERNEL);


}  // namespace cb_gmres


namespace ir {


GKO_STUB(GKO_DECLARE_IR_INITIALIZE_KERNEL);


}  // namespace ir


namespace multigrid {


GKO_STUB_VALUE_TYPE(GKO_DECLARE_MULTIGRID_KCYCLE_STEP_1_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_MULTIGRID_KCYCLE_STEP_2_KERNEL);
GKO_STUB_NON_COMPLEX_VALUE_TYPE(GKO_DECLARE_MULTIGRID_KCYCLE_CHECK_STOP_KERNEL);


}  // namespace multigrid


namespace sparsity_csr {


GKO_STUB_MIXED_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SPARSITY_CSR_SPMV_KERNEL);
GKO_STUB_MIXED_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_ADVANCED_SPMV_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SPARSITY_CSR_FILL_IN_DENSE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_COUNT_NUM_DIAGONAL_ELEMENTS_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_REMOVE_DIAGONAL_ELEMENTS_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SPARSITY_CSR_TRANSPOSE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SPARSITY_CSR_SORT_BY_COLUMN_INDEX);
GKO_STUB_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_IS_SORTED_BY_COLUMN_INDEX);


}  // namespace sparsity_csr


namespace csr {


GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPMV_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_ADVANCED_SPMV_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPGEMM_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_ADVANCED_SPGEMM_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPGEAM_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_FILL_IN_DENSE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_CONVERT_TO_ELL_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_CONVERT_TO_FBCSR_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_CONVERT_TO_HYBRID_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_CONVERT_TO_SELLP_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_TRANSPOSE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_CONJ_TRANSPOSE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_INV_SYMM_PERMUTE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_ROW_PERMUTE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_INVERSE_COLUMN_PERMUTE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_INVERSE_ROW_PERMUTE_KERNEL);
GKO_STUB_INDEX_TYPE(GKO_DECLARE_INVERT_PERMUTATION_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SORT_BY_COLUMN_INDEX);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_IS_SORTED_BY_COLUMN_INDEX);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_EXTRACT_DIAGONAL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_CALC_NNZ_PER_ROW_IN_SPAN_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALC_NNZ_PER_ROW_IN_INDEX_SET_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_COMPUTE_SUB_MATRIX_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_CHECK_DIAGONAL_ENTRIES_EXIST);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_ADD_SCALED_IDENTITY_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_COMPUTE_SUB_MATRIX_FROM_INDEX_SET_KERNEL);
GKO_STUB_INDEX_TYPE(GKO_DECLARE_CSR_BUILD_LOOKUP_OFFSETS_KERNEL);
GKO_STUB_INDEX_TYPE(GKO_DECLARE_CSR_BUILD_LOOKUP_KERNEL);
GKO_STUB_INDEX_TYPE(GKO_DECLARE_CSR_BENCHMARK_LOOKUP_KERNEL);

template <typename ValueType, typename IndexType>
GKO_DECLARE_CSR_SCALE_KERNEL(ValueType, IndexType)
GKO_NOT_COMPILED(GKO_HOOK_MODULE);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SCALE_KERNEL);

template <typename ValueType, typename IndexType>
GKO_DECLARE_CSR_INV_SCALE_KERNEL(ValueType, IndexType)
GKO_NOT_COMPILED(GKO_HOOK_MODULE);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_INV_SCALE_KERNEL);


}  // namespace csr


namespace fbcsr {


GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FBCSR_SPMV_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FBCSR_ADVANCED_SPMV_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FBCSR_FILL_IN_MATRIX_DATA_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FBCSR_FILL_IN_DENSE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FBCSR_CONVERT_TO_CSR_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FBCSR_TRANSPOSE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FBCSR_CONJ_TRANSPOSE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FBCSR_IS_SORTED_BY_COLUMN_INDEX);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FBCSR_SORT_BY_COLUMN_INDEX);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FBCSR_EXTRACT_DIAGONAL);


}  // namespace fbcsr


namespace coo {


GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_ADVANCED_SPMV_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV2_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_ADVANCED_SPMV2_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_FILL_IN_DENSE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_EXTRACT_DIAGONAL_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_CONVERT_TO_BCCOO_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_MEM_SIZE_BCCOO_KERNEL);
/*
template <typename IndexType>
GKO_DECLARE_MEM_SIZE_BCCOO_KERNEL(IndexType)
GKO_NOT_COMPILED(GKO_HOOK_MODULE);
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_MEM_SIZE_BCCOO_KERNEL);
*/


}  // namespace coo


// TODO (script:bccoo): adapt this block as needed
namespace bccoo {


GKO_DECLARE_GET_DEFAULT_BLOCK_SIZE_KERNEL()
GKO_NOT_COMPILED(GKO_HOOK_MODULE);

GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCCOO_SPMV_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCCOO_ADVANCED_SPMV_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCCOO_SPMV2_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCCOO_ADVANCED_SPMV2_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCCOO_CONVERT_TO_CSR_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCCOO_CONVERT_TO_DENSE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCCOO_EXTRACT_DIAGONAL_KERNEL);


}  // namespace bccoo


namespace ell {


GKO_STUB_MIXED_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ELL_SPMV_KERNEL);
GKO_STUB_MIXED_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ELL_ADVANCED_SPMV_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ELL_FILL_IN_MATRIX_DATA_KERNEL);
GKO_STUB_INDEX_TYPE(GKO_DECLARE_ELL_COMPUTE_MAX_ROW_NNZ_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ELL_FILL_IN_DENSE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ELL_COPY_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ELL_CONVERT_TO_CSR_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ELL_COUNT_NONZEROS_PER_ROW_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ELL_EXTRACT_DIAGONAL_KERNEL);


}  // namespace ell


namespace fft {


template <typename ValueType>
GKO_DECLARE_FFT_KERNEL(ValueType)
GKO_NOT_COMPILED(GKO_HOOK_MODULE);
GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(GKO_DECLARE_FFT_KERNEL);

template <typename ValueType>
GKO_DECLARE_FFT2_KERNEL(ValueType)
GKO_NOT_COMPILED(GKO_HOOK_MODULE);
GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(GKO_DECLARE_FFT2_KERNEL);

template <typename ValueType>
GKO_DECLARE_FFT3_KERNEL(ValueType)
GKO_NOT_COMPILED(GKO_HOOK_MODULE);
GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(GKO_DECLARE_FFT3_KERNEL);


}  // namespace fft


namespace hybrid {


GKO_STUB(GKO_DECLARE_HYBRID_COMPUTE_COO_ROW_PTRS_KERNEL);
GKO_STUB(GKO_DECLARE_HYBRID_COMPUTE_ROW_NNZ);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_HYBRID_FILL_IN_MATRIX_DATA_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_HYBRID_CONVERT_TO_CSR_KERNEL);


}  // namespace hybrid


namespace sellp {


GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SELLP_SPMV_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SELLP_FILL_IN_MATRIX_DATA_KERNEL);
GKO_STUB_INDEX_TYPE(GKO_DECLARE_SELLP_COMPUTE_SLICE_SETS_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SELLP_ADVANCED_SPMV_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SELLP_FILL_IN_DENSE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SELLP_CONVERT_TO_CSR_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SELLP_COUNT_NONZEROS_PER_ROW_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SELLP_EXTRACT_DIAGONAL_KERNEL);


}  // namespace sellp


namespace jacobi {


GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_JACOBI_FIND_BLOCKS_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_JACOBI_GENERATE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_JACOBI_APPLY_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_JACOBI_SIMPLE_APPLY_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_JACOBI_SCALAR_APPLY_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_JACOBI_SIMPLE_SCALAR_APPLY_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_JACOBI_TRANSPOSE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_JACOBI_CONJ_TRANSPOSE_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_JACOBI_SCALAR_CONVERT_TO_DENSE_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_JACOBI_SCALAR_CONJ_KERNEL);
GKO_STUB_VALUE_TYPE(GKO_DECLARE_JACOBI_INVERT_DIAGONAL_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_JACOBI_CONVERT_TO_DENSE_KERNEL);
GKO_STUB(GKO_DECLARE_JACOBI_INITIALIZE_PRECISIONS_KERNEL);


}  // namespace jacobi


namespace isai {


GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ISAI_GENERATE_TRI_INVERSE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ISAI_GENERATE_GENERAL_INVERSE_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ISAI_GENERATE_EXCESS_SYSTEM_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ISAI_SCALE_EXCESS_SOLUTION_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ISAI_SCATTER_EXCESS_SOLUTION_KERNEL);


}  // namespace isai


namespace cholesky {


GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CHOLESKY_SYMBOLIC_COUNT);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CHOLESKY_SYMBOLIC_FACTORIZE);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CHOLESKY_FOREST_FROM_FACTOR);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CHOLESKY_INITIALIZE);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CHOLESKY_FACTORIZE);


}  // namespace cholesky


namespace factorization {


GKO_STUB_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_ADD_DIAGONAL_ELEMENTS_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_L_U_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FACTORIZATION_INITIALIZE_L_U_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_L_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FACTORIZATION_INITIALIZE_L_KERNEL);


}  // namespace factorization


namespace ic_factorization {


GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_IC_COMPUTE_KERNEL);


}  // namespace ic_factorization


namespace ilu_factorization {


GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ILU_COMPUTE_LU_KERNEL);


}  // namespace ilu_factorization


namespace lu_factorization {


GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LU_INITIALIZE);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LU_FACTORIZE);


}  // namespace lu_factorization


namespace par_ic_factorization {


GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PAR_IC_INIT_FACTOR_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PAR_IC_COMPUTE_FACTOR_KERNEL);


}  // namespace par_ic_factorization


namespace par_ict_factorization {


GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PAR_ICT_ADD_CANDIDATES_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PAR_ICT_COMPUTE_FACTOR_KERNEL);


}  // namespace par_ict_factorization


namespace par_ilu_factorization {


GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PAR_ILU_COMPUTE_L_U_FACTORS_KERNEL);


}  // namespace par_ilu_factorization


namespace par_ilut_factorization {


GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PAR_ILUT_ADD_CANDIDATES_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PAR_ILUT_COMPUTE_LU_FACTORS_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PAR_ILUT_THRESHOLD_SELECT_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PAR_ILUT_THRESHOLD_FILTER_KERNEL);
GKO_STUB_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_THRESHOLD_FILTER_APPROX_KERNEL);


}  // namespace par_ilut_factorization
namespace rcm {


GKO_STUB_INDEX_TYPE(GKO_DECLARE_RCM_GET_PERMUTATION_KERNEL);
GKO_STUB_INDEX_TYPE(GKO_DECLARE_RCM_GET_DEGREE_OF_NODES_KERNEL);


}  // namespace rcm


namespace pgm {


GKO_STUB_INDEX_TYPE(GKO_DECLARE_PGM_MATCH_EDGE_KERNEL);
GKO_STUB_INDEX_TYPE(GKO_DECLARE_PGM_COUNT_UNAGG_KERNEL);
GKO_STUB_INDEX_TYPE(GKO_DECLARE_PGM_RENUMBER_KERNEL);
GKO_STUB_INDEX_TYPE(GKO_DECLARE_PGM_SORT_AGG_KERNEL);
GKO_STUB_INDEX_TYPE(GKO_DECLARE_PGM_MAP_ROW_KERNEL);
GKO_STUB_INDEX_TYPE(GKO_DECLARE_PGM_MAP_COL_KERNEL);
GKO_STUB_INDEX_TYPE(GKO_DECLARE_PGM_COUNT_UNREPEATED_NNZ_KERNEL);
GKO_STUB_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PGM_FIND_STRONGEST_NEIGHBOR);
GKO_STUB_NON_COMPLEX_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PGM_ASSIGN_TO_EXIST_AGG);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PGM_SORT_ROW_MAJOR);
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PGM_COMPUTE_COARSE_COO);


}  // namespace pgm


namespace set_all_statuses {


GKO_STUB(GKO_DECLARE_SET_ALL_STATUSES_KERNEL);


}  // namespace set_all_statuses


namespace residual_norm {


GKO_STUB_NON_COMPLEX_VALUE_TYPE(GKO_DECLARE_RESIDUAL_NORM_KERNEL);


}  // namespace residual_norm


namespace implicit_residual_norm {


GKO_STUB_VALUE_TYPE(GKO_DECLARE_IMPLICIT_RESIDUAL_NORM_KERNEL);


}  // namespace implicit_residual_norm
}  // namespace GKO_HOOK_MODULE
}  // namespace kernels
}  // namespace gko

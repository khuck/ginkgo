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

#include "core/matrix/bccoo_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/bccoo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/unaligned_access.hpp"
#include "core/matrix/bccoo_helper.hpp"
#include "core/matrix/bccoo_memsize_convert.hpp"
#include "core/matrix/dense_kernels.hpp"


namespace gko {
namespace kernels {
/**
 * @brief The Reference namespace.
 *
 * @ingroup reference
 */
namespace reference {
/**
 * @brief The Bccoordinate matrix format namespace.
 *
 * @ingroup bccoo
 */
namespace bccoo {


using namespace matrix::bccoo;


template <typename IndexType>
void get_default_block_size(std::shared_ptr<const ReferenceExecutor> exec,
                            IndexType* block_size)
{
    *block_size = 32;
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_GET_DEFAULT_BLOCK_SIZE_KERNEL);


void get_default_compression(std::shared_ptr<const ReferenceExecutor> exec,
                             compression* compression)
{
    *compression = matrix::bccoo::compression::individual;
}


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const ReferenceExecutor> exec,
          const matrix::Bccoo<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    dense::fill(exec, c, zero<ValueType>());
    spmv2(exec, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCCOO_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const ReferenceExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Bccoo<ValueType, IndexType>* a,
                   const matrix::Dense<ValueType>* b,
                   const matrix::Dense<ValueType>* beta,
                   matrix::Dense<ValueType>* c)
{
    dense::scale(exec, beta, c);
    advanced_spmv2(exec, alpha, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void spmv2(std::shared_ptr<const ReferenceExecutor> exec,
           const matrix::Bccoo<ValueType, IndexType>* a,
           const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    auto* start_rows = a->get_const_start_rows();
    auto* block_offsets = a->get_const_block_offsets();
    auto* compressed_data = a->get_const_compressed_data();

    auto block_size = a->get_block_size();
    auto num_stored_elements = a->get_num_stored_elements();
    auto num_cols = b->get_size()[1];

    compr_idxs<IndexType> idxs;
    ValueType val;

    if (a->use_individual_compression()) {
        // For individual compression objects
        for (IndexType i = 0; i < num_stored_elements; i++) {
            // Reading (row,col,val) from matrix
            get_detect_newblock<IndexType>(start_rows, block_offsets, idxs);
            uint8 key = get_position_newrow(compressed_data, idxs);
            get_next_position_value(compressed_data, key, idxs, val);
            get_detect_endblock(block_size, idxs);
            // Writing (row,col,val) to result
            for (IndexType j = 0; j < num_cols; j++) {
                c->at(idxs.row, j) += val * b->at(idxs.col, j);
            }
        }
    } else {
        // For group compression objects
        auto* start_cols = a->get_const_start_cols();
        auto* compression_types = a->get_const_compression_types();

        for (IndexType i = 0; i < num_stored_elements; i += block_size) {
            IndexType block_size_local =
                std::min(block_size, num_stored_elements - i);
            compr_grp_idxs<IndexType> grp_idxs(start_rows, start_cols,
                                               block_size_local, idxs,
                                               compression_types[idxs.blk]);
            for (IndexType j = 0; j < block_size_local; j++) {
                // Reading (row,col,val) from matrix
                get_group_position_value<IndexType, ValueType>(
                    compressed_data, grp_idxs, idxs, val);
                // Processing (row,col,val)
                for (IndexType k = 0; k < num_cols; k++) {
                    c->at(idxs.row, k) += val * b->at(idxs.col, k);
                }
            }
            idxs.blk++;
            idxs.shf = grp_idxs.shf_val;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCCOO_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv2(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Dense<ValueType>* alpha,
                    const matrix::Bccoo<ValueType, IndexType>* a,
                    const matrix::Dense<ValueType>* b,
                    matrix::Dense<ValueType>* c)
{
    auto* start_rows = a->get_const_start_rows();
    auto* block_offsets = a->get_const_block_offsets();
    auto* compressed_data = a->get_const_compressed_data();

    auto num_stored_elements = a->get_num_stored_elements();
    auto block_size = a->get_block_size();
    auto alpha_val = alpha->at(0, 0);
    auto num_cols = b->get_size()[1];

    compr_idxs<IndexType> idxs;
    ValueType val;

    if (a->use_individual_compression()) {
        // For individual compression objects
        for (IndexType i = 0; i < num_stored_elements; i++) {
            // Reading (row,col,val) from matrix
            get_detect_newblock<IndexType>(start_rows, block_offsets, idxs);
            uint8 key = get_position_newrow(compressed_data, idxs);
            get_next_position_value(compressed_data, key, idxs, val);
            get_detect_endblock(block_size, idxs);
            // Writing (row,col,val) to result
            for (IndexType j = 0; j < num_cols; j++) {
                c->at(idxs.row, j) += alpha_val * val * b->at(idxs.col, j);
            }
        }
    } else {
        // For group compression objects
        auto* start_cols = a->get_const_start_cols();
        auto* compression_types = a->get_const_compression_types();

        for (IndexType i = 0; i < num_stored_elements; i += block_size) {
            IndexType block_size_local =
                std::min(block_size, num_stored_elements - i);
            compr_grp_idxs<IndexType> grp_idxs(start_rows, start_cols,
                                               block_size_local, idxs,
                                               compression_types[idxs.blk]);
            for (IndexType j = 0; j < block_size_local; j++) {
                // Reading (row,col,val) from matrix
                get_group_position_value<IndexType, ValueType>(
                    compressed_data, grp_idxs, idxs, val);
                // Processing (row,col,val)
                for (IndexType k = 0; k < num_cols; k++) {
                    c->at(idxs.row, k) += alpha_val * val * b->at(idxs.col, k);
                }
            }
            idxs.blk++;
            idxs.shf = grp_idxs.shf_val;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_ADVANCED_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void mem_size_bccoo(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Bccoo<ValueType, IndexType>* source,
                    compression compress_res, const IndexType block_size_res,
                    size_type* mem_size)
{
    // If source and result have the same block_size and compression
    // size of the compressed data will also be the same
    if ((source->get_block_size() == block_size_res) &&
        (source->get_compression() == compress_res)) {
        *mem_size = source->get_num_bytes();
    } else if ((source->use_individual_compression()) &&
               (compress_res == source->get_compression())) {
        mem_size_bccoo_ind_ind(exec, source, block_size_res, mem_size);
    } else if (source->use_individual_compression()) {
        mem_size_bccoo_ind_grp(exec, source, block_size_res, mem_size);
    } else if (compress_res == compression::individual) {
        mem_size_bccoo_grp_ind(exec, source, block_size_res, mem_size);
    } else {
        mem_size_bccoo_grp_grp(exec, source, block_size_res, mem_size);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_MEM_SIZE_BCCOO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_bccoo(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* source,
                      matrix::Bccoo<ValueType, IndexType>* result)
{
    auto block_size_res = result->get_block_size();
    auto compress_res = result->get_compression();
    // If the compression and block_size values are the same in source and
    // result objects, a raw copy is applied
    if ((source->get_block_size() == result->get_block_size()) &&
        (source->get_compression() == result->get_compression())) {
        convert_to_bccoo_copy(exec, source, result);
    } else if ((source->use_individual_compression()) &&
               (result->use_individual_compression())) {
        convert_to_bccoo_ind_ind(exec, source, result,
                                 [](ValueType val) { return val; });
    } else if (source->use_individual_compression()) {
        convert_to_bccoo_ind_grp(exec, source, result);
    } else if (compress_res == compression::individual) {
        convert_to_bccoo_grp_ind(exec, source, result);
    } else {
        convert_to_bccoo_grp_grp(exec, source, result,
                                 [](ValueType val) { return val; });
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_BCCOO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_next_precision(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    matrix::Bccoo<next_precision<ValueType>, IndexType>* result)
{
    auto compress_res = result->get_compression();
    // In this case, a raw copy is never applied because the sizes of the
    // values in source and result are different
    if ((source->use_individual_compression()) &&
        (result->use_individual_compression())) {
        convert_to_bccoo_ind_ind(exec, source, result,
                                 [](ValueType val) { return val; });
    } else if (source->use_individual_compression()) {
        convert_to_bccoo_ind_grp(exec, source, result);
    } else if (compress_res == compression::individual) {
        convert_to_bccoo_grp_ind(exec, source, result);
    } else {
        convert_to_bccoo_grp_grp(exec, source, result,
                                 [](ValueType val) { return val; });
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_NEXT_PRECISION_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Bccoo<ValueType, IndexType>* source,
                    matrix::Coo<ValueType, IndexType>* result)
{
    auto* start_rows = source->get_const_start_rows();
    auto* block_offsets = source->get_const_block_offsets();
    auto* compressed_data = source->get_const_compressed_data();

    auto block_size = source->get_block_size();
    auto num_stored_elements = source->get_num_stored_elements();

    compr_idxs<IndexType> idxs;
    ValueType val;

    auto row_idxs = result->get_row_idxs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

    if (source->use_individual_compression()) {
        // For individual compression objects
        for (IndexType i = 0; i < num_stored_elements; i++) {
            // Reading (row,col,val) from source
            get_detect_newblock<IndexType>(start_rows, block_offsets, idxs);
            uint8 key = get_position_newrow(compressed_data, idxs);
            get_next_position_value(compressed_data, key, idxs, val);
            get_detect_endblock(block_size, idxs);
            // Writing (row,col,val) to result
            row_idxs[i] = idxs.row;
            col_idxs[i] = idxs.col;
            values[i] = val;
        }
    } else {
        // For group compression objects
        auto* start_cols = source->get_const_start_cols();
        auto* compression_types = source->get_const_compression_types();

        for (IndexType i = 0; i < num_stored_elements; i += block_size) {
            IndexType block_size_local =
                std::min(block_size, num_stored_elements - i);
            compr_grp_idxs<IndexType> grp_idxs(start_rows, start_cols,
                                               block_size_local, idxs,
                                               compression_types[idxs.blk]);
            for (IndexType j = 0; j < block_size_local; j++) {
                // Reading (row,col,val) from source
                get_group_position_value<IndexType, ValueType>(
                    compressed_data, grp_idxs, idxs, val);
                // Writing (row,col,val) to result
                row_idxs[i + j] = idxs.row;
                col_idxs[i + j] = idxs.col;
                values[i + j] = val;
            }
            idxs.blk++;
            idxs.shf = grp_idxs.shf_val;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_COO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Bccoo<ValueType, IndexType>* source,
                    matrix::Csr<ValueType, IndexType>* result)
{
    auto* start_rows = source->get_const_start_rows();
    auto* block_offsets = source->get_const_block_offsets();
    auto* compressed_data = source->get_const_compressed_data();

    auto block_size = source->get_block_size();
    auto num_stored_elements = source->get_num_stored_elements();

    compr_idxs<IndexType> idxs;
    ValueType val;

    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

    row_ptrs[0] = 0;
    if (source->use_individual_compression()) {
        // For individual compression objects
        for (IndexType i = 0; i < num_stored_elements; i++) {
            // Reading (row,col,val) from source
            get_detect_newblock_csr<IndexType>(start_rows, block_offsets,
                                               row_ptrs, i, idxs);
            uint8 key =
                get_position_newrow_csr(compressed_data, row_ptrs, i, idxs);
            get_next_position_value(compressed_data, key, idxs, val);
            get_detect_endblock(block_size, idxs);
            // Writing (row,col,val) to result
            col_idxs[i] = idxs.col;
            values[i] = val;
        }
    } else {
        // For group compression objects
        auto* start_cols = source->get_const_start_cols();
        auto* compression_types = source->get_const_compression_types();

        IndexType row_prv = 0;

        for (IndexType i = 0; i < num_stored_elements; i += block_size) {
            IndexType block_size_local =
                std::min(block_size, num_stored_elements - i);
            compr_grp_idxs<IndexType> grp_idxs(start_rows, start_cols,
                                               block_size_local, idxs,
                                               compression_types[idxs.blk]);
            for (IndexType j = 0; j < block_size_local; j++) {
                // Reading (row,col,val) from source
                get_group_position_value<IndexType, ValueType>(
                    compressed_data, grp_idxs, idxs, val);
                // Writing (row,col,val) to result
                if (row_prv < idxs.row) {
                    row_ptrs[idxs.row] = i + j;
                }
                col_idxs[i + j] = idxs.col;
                values[i + j] = val;
                row_prv = idxs.row;
            }
            idxs.blk++;
            idxs.shf = grp_idxs.shf_val;
        }
    }
    if (num_stored_elements > 0) {
        row_ptrs[idxs.row + 1] = num_stored_elements;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* source,
                      matrix::Dense<ValueType>* result)
{
    auto* start_rows = source->get_const_start_rows();
    auto* block_offsets = source->get_const_block_offsets();
    auto* compressed_data = source->get_const_compressed_data();

    auto block_size = source->get_block_size();
    auto num_stored_elements = source->get_num_stored_elements();

    compr_idxs<IndexType> idxs;
    ValueType val;

    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];

    // First, result is initialized to zero
    for (IndexType row = 0; row < num_rows; row++) {
        for (IndexType col = 0; col < num_cols; col++) {
            result->at(row, col) = zero<ValueType>();
        }
    }

    if (source->use_individual_compression()) {
        // For individual compression objects
        for (IndexType i = 0; i < num_stored_elements; i++) {
            // Reading (row,col,val) from source
            get_detect_newblock<IndexType>(start_rows, block_offsets, idxs);
            uint8 key = get_position_newrow(compressed_data, idxs);
            get_next_position_value(compressed_data, key, idxs, val);
            get_detect_endblock(block_size, idxs);
            // Writing (row,col,val) to result
            result->at(idxs.row, idxs.col) += val;
        }
    } else {
        // For group compression objects
        auto* start_cols = source->get_const_start_cols();
        auto* compression_types = source->get_const_compression_types();

        for (IndexType i = 0; i < num_stored_elements; i += block_size) {
            IndexType block_size_local =
                std::min(block_size, num_stored_elements - i);
            compr_grp_idxs<IndexType> grp_idxs(start_rows, start_cols,
                                               block_size_local, idxs,
                                               compression_types[idxs.blk]);
            for (IndexType j = 0; j < block_size_local; j++) {
                // Reading (row,col,val) from source
                get_group_position_value<IndexType, ValueType>(
                    compressed_data, grp_idxs, idxs, val);
                // Writing (row,col,val) to result
                result->at(idxs.row, idxs.col) += val;
            }
            idxs.blk++;
            idxs.shf = grp_idxs.shf_val;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag)
{
    auto* start_rows = orig->get_const_start_rows();
    auto* block_offsets = orig->get_const_block_offsets();
    auto* compressed_data = orig->get_const_compressed_data();

    auto block_size = orig->get_block_size();
    auto num_stored_elements = orig->get_num_stored_elements();

    compr_idxs<IndexType> idxs;
    ValueType val;

    auto diag_values = diag->get_values();
    auto num_rows = diag->get_size()[0];
    // First, diag is initialized to zero
    for (IndexType row = 0; row < num_rows; row++) {
        diag_values[row] = zero<ValueType>();
    }

    if (orig->use_individual_compression()) {
        // For individual compression objects
        for (IndexType i = 0; i < num_stored_elements; i++) {
            // Reading (row,col,val) from orig
            get_detect_newblock<IndexType>(start_rows, block_offsets, idxs);
            uint8 key = get_position_newrow(compressed_data, idxs);
            get_next_position_value(compressed_data, key, idxs, val);
            get_detect_endblock(block_size, idxs);
            // Writing (row,col,val) to diag
            if (idxs.row == idxs.col) {
                diag_values[idxs.row] = val;
            }
        }
    } else {
        // For group compression objects
        auto* start_cols = orig->get_const_start_cols();
        auto* compression_types = orig->get_const_compression_types();

        for (IndexType i = 0; i < num_stored_elements; i += block_size) {
            IndexType block_size_local =
                std::min(block_size, num_stored_elements - i);
            compr_grp_idxs<IndexType> grp_idxs(start_rows, start_cols,
                                               block_size_local, idxs,
                                               compression_types[idxs.blk]);
            for (IndexType j = 0; j < block_size_local; j++) {
                // Reading (row,col,val) from orig
                get_group_position_value<IndexType, ValueType>(
                    compressed_data, grp_idxs, idxs, val);
                // Writing (row,col,val) to diag
                if (idxs.row == idxs.col) {
                    diag_values[idxs.row] = val;
                }
            }
            idxs.blk++;
            idxs.shf = grp_idxs.shf_val;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_EXTRACT_DIAGONAL_KERNEL);


template <typename ValueType, typename IndexType>
void compute_absolute_inplace(std::shared_ptr<const ReferenceExecutor> exec,
                              matrix::Bccoo<ValueType, IndexType>* matrix)
{
    auto* start_rows = matrix->get_const_start_rows();
    auto* block_offsets = matrix->get_const_block_offsets();
    auto* compressed_data = matrix->get_compressed_data();

    auto block_size = matrix->get_block_size();
    auto num_stored_elements = matrix->get_num_stored_elements();

    compr_idxs<IndexType> idxs;
    ValueType val;

    if (matrix->use_individual_compression()) {
        // For individual compression objects
        for (IndexType i = 0; i < num_stored_elements; i++) {
            // Reading/Writing (row,col,val) from/to matrix
            get_detect_newblock<IndexType>(start_rows, block_offsets, idxs);
            uint8 key = get_position_newrow(compressed_data, idxs);
            get_next_position_value_put(compressed_data, key, idxs, val,
                                        [](ValueType val) { return abs(val); });
            get_detect_endblock(block_size, idxs);
        }
    } else {
        // For group compression objects
        auto* start_cols = matrix->get_const_start_cols();
        auto* compression_types = matrix->get_const_compression_types();

        for (IndexType i = 0; i < num_stored_elements; i += block_size) {
            IndexType block_size_local =
                std::min(block_size, num_stored_elements - i);
            compr_grp_idxs<IndexType> grp_idxs(start_rows, start_cols,
                                               block_size_local, idxs,
                                               compression_types[idxs.blk]);
            for (IndexType j = 0; j < block_size_local; j++) {
                // Reading/Writing (row,col,val) from/to matrix
                get_group_position_value_put<IndexType, ValueType>(
                    compressed_data, grp_idxs, idxs, val,
                    [](ValueType val) { return abs(val); });
            }
            idxs.blk++;
            idxs.shf = grp_idxs.shf_val;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_COMPUTE_ABSOLUTE_INPLACE_KERNEL);


template <typename ValueType, typename IndexType>
void compute_absolute(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    remove_complex<matrix::Bccoo<ValueType, IndexType>>* result)
{
    // Only conversions between objects with the same compression are allowed
    if (source->use_individual_compression()) {
        convert_to_bccoo_ind_ind(exec, source, result,
                                 [](ValueType val) { return abs(val); });
    } else {
        convert_to_bccoo_grp_grp(exec, source, result,
                                 [](ValueType val) { return abs(val); });
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_COMPUTE_ABSOLUTE_KERNEL);


}  // namespace bccoo
}  // namespace reference
}  // namespace kernels
}  // namespace gko

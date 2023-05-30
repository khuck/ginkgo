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

#ifndef GKO_CORE_MATRIX_BCCOO_MEMSIZE_CONVERT_HPP_
#define GKO_CORE_MATRIX_BCCOO_MEMSIZE_CONVERT_HPP_


#include <cstring>
#include <functional>


#include <ginkgo/core/base/types.hpp>


#include "core/base/unaligned_access.hpp"
#include "core/matrix/bccoo_helper.hpp"


namespace gko {
namespace matrix {
namespace bccoo {


/**
 *  Routines for mem_size computing
 */


/**
 *  Returns the size of the chunk, which it is need
 *  to store the data included in an element compression object
 *  into an element compression object whose block_size is specified
 */
template <typename ValueType, typename IndexType>
inline void mem_size_bccoo_elm_elm(
    std::shared_ptr<const Executor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    const IndexType block_size_res, size_type* mem_size)
{
    // This routine only is useful for master executor
    GKO_ASSERT(exec == exec->get_master());
    GKO_ASSERT(source->use_element_compression());

    const IndexType* rows_data_src = source->get_const_rows();
    const size_type* offsets_data_src = source->get_const_offsets();
    const uint8* chunk_data_src = source->get_const_chunk();

    const IndexType num_stored_elements = source->get_num_stored_elements();
    const IndexType block_size_src = source->get_block_size();

    compr_idxs<IndexType> idxs_src;
    ValueType val_src;
    compr_idxs<IndexType> idxs_res;
    ValueType val_res;

    for (IndexType i = 0; i < num_stored_elements; i++) {
        // Reading (row,col,val) from source
        get_detect_newblock(rows_data_src, offsets_data_src, idxs_src);
        uint8 ind_src = get_position_newrow(chunk_data_src, idxs_src);
        get_next_position_value(chunk_data_src, ind_src, idxs_src, val_src);
        get_detect_endblock(block_size_src, idxs_src);
        // Counting bytes to write (row,col,val) on result
        cnt_detect_newblock(idxs_src.row - idxs_res.row, idxs_res);
        IndexType col_src_res =
            cnt_position_newrow_mat_data(idxs_src.row, idxs_src.col, idxs_res);
        cnt_next_position_value(col_src_res, val_src, idxs_res);
        cnt_detect_endblock(block_size_res, idxs_res);
    }
    *mem_size = idxs_res.shf;
}


/**
 *  Returns the size of the chunk, which it is need
 *  to store the data included in an element compression object
 *  into a block compression object whose block_size is specified
 */
template <typename ValueType, typename IndexType>
inline void mem_size_bccoo_elm_blk(
    std::shared_ptr<const Executor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    const IndexType block_size_res, size_type* mem_size)
{
    // This routine only is useful for master executor
    GKO_ASSERT(exec == exec->get_master());
    GKO_ASSERT(source->use_element_compression());

    const IndexType* rows_data_src = source->get_const_rows();
    const size_type* offsets_data_src = source->get_const_offsets();
    const uint8* chunk_data_src = source->get_const_chunk();

    const IndexType num_stored_elements = source->get_num_stored_elements();
    const IndexType block_size_src = source->get_block_size();

    compr_idxs<IndexType> idxs_src;
    ValueType val_src;

    compr_idxs<IndexType> idxs_res;

    for (IndexType i = 0; i < num_stored_elements; i += block_size_res) {
        IndexType block_size_local =
            std::min(block_size_res, num_stored_elements - i);
        compr_blk_idxs<IndexType> blk_idxs_res;
        blk_idxs_res.row_frst = idxs_src.row;
        blk_idxs_res.col_frst = idxs_src.col;
        for (IndexType j = 0; j < block_size_local; j++) {
            // Reading (row,col,val) from source
            get_detect_newblock(rows_data_src, offsets_data_src, idxs_src);
            uint8 ind_src = get_position_newrow(chunk_data_src, idxs_src);
            get_next_position_value(chunk_data_src, ind_src, idxs_src, val_src);
            get_detect_endblock(block_size_src, idxs_src);
            // Analyzing the impact of (row,col,val) in the block
            idxs_res.nblk = j;
            proc_block_indices<IndexType>(idxs_src.row, idxs_src.col, idxs_res,
                                          blk_idxs_res);
        }
        // Counting bytes to write block on result
        cnt_block_indices<IndexType, ValueType>(block_size_local, blk_idxs_res,
                                                idxs_res);
    }
    *mem_size = idxs_res.shf;
}


/**
 *  Returns the size of the chunk, which it is need
 *  to store the data included in a blok compression object
 *  into an element compression object whose block_size is specified
 */
template <typename ValueType, typename IndexType>
inline void mem_size_bccoo_blk_elm(
    std::shared_ptr<const Executor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    const IndexType block_size_res, size_type* mem_size)
{
    // This routine only is useful for master executor
    GKO_ASSERT(exec == exec->get_master());
    GKO_ASSERT(source->use_block_compression());

    const IndexType* rows_data_src = source->get_const_rows();
    const size_type* offsets_data_src = source->get_const_offsets();
    const uint8* chunk_data_src = source->get_const_chunk();
    const IndexType* cols_data_src = source->get_const_cols();
    const uint8* types_data_src = source->get_const_types();

    const IndexType block_size_src = source->get_block_size();
    const size_type num_bytes_src = source->get_num_bytes();
    const IndexType num_stored_elements = source->get_num_stored_elements();

    compr_idxs<IndexType> idxs_src;
    ValueType val_src;

    compr_idxs<IndexType> idxs_res;

    for (IndexType i = 0; i < num_stored_elements; i += block_size_src) {
        IndexType block_size_local =
            std::min(block_size_src, num_stored_elements - i);
        compr_blk_idxs<IndexType> blk_idxs_src(rows_data_src, cols_data_src,
                                               block_size_local, idxs_src,
                                               types_data_src[idxs_src.blk]);
        for (IndexType j = 0; j < block_size_local; j++) {
            // Reading (row,col,val) from source
            get_block_position_value<IndexType, ValueType>(
                chunk_data_src, blk_idxs_src, idxs_src, val_src);
            // Counting bytes to write (row,col,val) on result
            cnt_detect_newblock(idxs_src.row - idxs_res.row, idxs_res);
            IndexType col_src_res = cnt_position_newrow_mat_data(
                idxs_src.row, idxs_src.col, idxs_res);
            cnt_next_position_value(col_src_res, val_src, idxs_res);
            cnt_detect_endblock(block_size_res, idxs_res);
        }
        idxs_src.blk++;
        idxs_src.shf = blk_idxs_src.shf_val;
    }
    *mem_size = idxs_res.shf;
}


/**
 *  Returns the size of the chunk, which it is need
 *  to store the data included in a block compression object
 *  into a block compression object whose block_size is specified
 */
template <typename ValueType, typename IndexType>
inline void mem_size_bccoo_blk_blk(
    std::shared_ptr<const Executor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    const IndexType block_size_res, size_type* mem_size)
{
    // This routine only is useful for master executor
    GKO_ASSERT(exec == exec->get_master());
    GKO_ASSERT(source->use_block_compression());

    const IndexType* rows_data_src = source->get_const_rows();
    const size_type* offsets_data_src = source->get_const_offsets();
    const uint8* chunk_data_src = source->get_const_chunk();
    const IndexType* cols_data_src = source->get_const_cols();
    const uint8* types_data_src = source->get_const_types();

    const IndexType block_size_src = source->get_block_size();
    const size_type num_bytes_src = source->get_num_bytes();
    const IndexType num_stored_elements = source->get_num_stored_elements();

    compr_idxs<IndexType> idxs_src;
    ValueType val_src;

    const IndexType* rows_data_res = source->get_const_rows();
    const size_type* offsets_data_res = source->get_const_offsets();
    const uint8* chunk_data_res = source->get_const_chunk();
    const IndexType* cols_data_res = source->get_const_cols();
    const uint8* types_data_res = source->get_const_types();

    compr_idxs<IndexType> idxs_res;
    compr_blk_idxs<IndexType> blk_idxs_res;
    ValueType val_res;

    IndexType i_res = 0;
    IndexType block_size_local_res =
        std::min(block_size_res, num_stored_elements - i_res);

    for (IndexType i = 0; i < num_stored_elements; i += block_size_src) {
        IndexType block_size_local_src =
            std::min(block_size_src, num_stored_elements - i);
        compr_blk_idxs<IndexType> blk_idxs_src(rows_data_src, cols_data_src,
                                               block_size_local_src, idxs_src,
                                               types_data_src[idxs_src.blk]);
        for (IndexType j = 0; j < block_size_local_src; j++) {
            // Reading (row,col,val) from source
            get_block_position_value<IndexType, ValueType>(
                chunk_data_src, blk_idxs_src, idxs_src, val_src);
            proc_block_indices<IndexType>(idxs_src.row, idxs_src.col, idxs_res,
                                          blk_idxs_res);
            idxs_res.nblk++;
            if (idxs_res.nblk == block_size_local_res) {
                // Counting bytes to write block on result
                cnt_block_indices<IndexType, ValueType>(block_size_local_res,
                                                        blk_idxs_res, idxs_res);
                i_res += block_size_local_res;
                block_size_local_res =
                    std::min(block_size_res, num_stored_elements - i_res);
                idxs_res.nblk = 0;
                blk_idxs_res = {};
            }
        }
        idxs_src.blk++;
        idxs_src.shf = blk_idxs_src.shf_val;
    }
    *mem_size = idxs_res.shf;
}


/**
 *  Routines for conversion between bccoo objects
 */


/**
 *  This routine makes a raw copy between bccoo objects whose block_size
 *  and compression are the same
 */
template <typename ValueType, typename IndexType>
void convert_to_bccoo_copy(std::shared_ptr<const Executor> exec,
                           const matrix::Bccoo<ValueType, IndexType>* source,
                           matrix::Bccoo<ValueType, IndexType>* result)
{
    // This routine only is useful for master executor
    GKO_ASSERT(exec == exec->get_master());
    GKO_ASSERT(source->get_compression() == result->get_compression());

    // Try to remove static_cast
    if (source->get_num_stored_elements() > 0) {
        if (source->use_element_compression()) {
            std::memcpy(result->get_rows(), source->get_const_rows(),
                        source->get_num_blocks() * sizeof(IndexType));
            const size_type* offsets_data_src = source->get_const_offsets();
            size_type* offsets_data_res = result->get_offsets();
            std::memcpy(offsets_data_res, offsets_data_src,
                        (source->get_num_blocks() + 1) * sizeof(size_type));
            const uint8* chunk_data_src = source->get_const_chunk();
            uint8* chunk_data_res = result->get_chunk();
            std::memcpy(chunk_data_res, chunk_data_src,
                        source->get_num_bytes() * sizeof(uint8));
        } else {
            std::memcpy(result->get_rows(), source->get_const_rows(),
                        source->get_num_blocks() * sizeof(IndexType));
            std::memcpy(result->get_cols(), source->get_const_cols(),
                        source->get_num_blocks() * sizeof(IndexType));
            std::memcpy(result->get_types(), source->get_const_types(),
                        source->get_num_blocks() * sizeof(uint8));
            std::memcpy(result->get_offsets(), source->get_const_offsets(),
                        (source->get_num_blocks() + 1) * sizeof(size_type));
            std::memcpy(result->get_chunk(), source->get_const_chunk(),
                        source->get_num_bytes() * sizeof(uint8));
        }
    }
}


/**
 *  This routine makes the conversion between two element compression objects
 *  Additionally, finalize_op function is applied before to copy the values
 */
template <typename ValueType_src, typename ValueType_res, typename IndexType,
          typename Callable>
void convert_to_bccoo_elm_elm(
    std::shared_ptr<const Executor> exec,
    const matrix::Bccoo<ValueType_src, IndexType>* source,
    matrix::Bccoo<ValueType_res, IndexType>* result, Callable finalize_op)
{
    // This routine only is useful for master executor
    GKO_ASSERT(exec == exec->get_master());
    GKO_ASSERT(source->use_element_compression());
    GKO_ASSERT(result->use_element_compression());

    const IndexType* rows_data_src = source->get_const_rows();
    const size_type* offsets_data_src = source->get_const_offsets();
    const uint8* chunk_data_src = source->get_const_chunk();

    const IndexType num_stored_elements = source->get_num_stored_elements();
    const IndexType block_size_src = source->get_block_size();

    compr_idxs<IndexType> idxs_src;
    ValueType_src val_src;

    IndexType* rows_data_res = result->get_rows();
    size_type* offsets_data_res = result->get_offsets();
    uint8* chunk_data_res = result->get_chunk();

    IndexType block_size_res = result->get_block_size();

    compr_idxs<IndexType> idxs_res;
    ValueType_res val_res;

    if (num_stored_elements > 0) {
        offsets_data_res[0] = 0;
    }
    for (IndexType i = 0; i < num_stored_elements; i++) {
        // Reading (row,col,val) from source
        get_detect_newblock(rows_data_src, offsets_data_src, idxs_src);
        uint8 ind_src = get_position_newrow(chunk_data_src, idxs_src);
        get_next_position_value(chunk_data_src, ind_src, idxs_src, val_src);
        get_detect_endblock(block_size_src, idxs_src);
        // Writing (row,col,val) to result
        val_res = finalize_op(val_src);
        put_detect_newblock(rows_data_res, idxs_src.row - idxs_res.row,
                            idxs_res);
        IndexType col_src_res = put_position_newrow_mat_data(
            idxs_src.row, idxs_src.col, chunk_data_res, idxs_res);
        put_next_position_value(chunk_data_res, col_src_res, val_res, idxs_res);
        put_detect_endblock(offsets_data_res, block_size_res, idxs_res);
    }
    if (idxs_res.nblk > 0) {
        offsets_data_res[++idxs_res.blk] = idxs_res.shf;
    }
}


/**
 *  This routine makes the conversion between an element compression object
 *  and a block compression object
 */
template <typename ValueType_src, typename ValueType_res, typename IndexType>
void convert_to_bccoo_elm_blk(
    std::shared_ptr<const Executor> exec,
    const matrix::Bccoo<ValueType_src, IndexType>* source,
    matrix::Bccoo<ValueType_res, IndexType>* result)
{
    // This routine only is useful for master executor
    GKO_ASSERT(exec == exec->get_master());
    GKO_ASSERT(source->use_element_compression());
    GKO_ASSERT(result->use_block_compression());

    const IndexType* rows_data_src = source->get_const_rows();
    const size_type* offsets_data_src = source->get_const_offsets();
    const uint8* chunk_data_src = source->get_const_chunk();

    const IndexType num_stored_elements = source->get_num_stored_elements();
    const IndexType block_size_src = source->get_block_size();

    compr_idxs<IndexType> idxs_src;
    ValueType_src val_src;

    IndexType* rows_data_res = result->get_rows();
    size_type* offsets_data_res = result->get_offsets();
    uint8* chunk_data_res = result->get_chunk();
    IndexType* cols_data_res = result->get_cols();
    uint8* types_data_res = result->get_types();

    const IndexType block_size_res = result->get_block_size();

    compr_idxs<IndexType> idxs_res;
    ValueType_res val_res;

    array<IndexType> rows_blk(exec, block_size_res);
    array<IndexType> cols_blk(exec, block_size_res);
    array<ValueType_res> vals_blk(exec, block_size_res);

    if (num_stored_elements > 0) {
        offsets_data_res[0] = 0;
    }
    for (IndexType i = 0; i < num_stored_elements; i += block_size_res) {
        IndexType block_size_local =
            std::min(block_size_res, num_stored_elements - i);
        compr_blk_idxs<IndexType> blk_idxs_res;
        uint8 type_blk = {};

        blk_idxs_res.row_frst = idxs_src.row;
        blk_idxs_res.col_frst = idxs_src.col;
        for (IndexType j = 0; j < block_size_local; j++) {
            // Reading (row,col,val) from source
            get_detect_newblock(rows_data_src, offsets_data_src, idxs_src);
            uint8 ind_src = get_position_newrow(chunk_data_src, idxs_src);
            get_next_position_value(chunk_data_src, ind_src, idxs_src, val_src);
            get_detect_endblock(block_size_src, idxs_src);
            // Analyzing the impact of (row,col,val) in the block
            idxs_res.nblk = j;
            proc_block_indices<IndexType>(idxs_src.row, idxs_src.col, idxs_res,
                                          blk_idxs_res);
            rows_blk.get_data()[j] = idxs_src.row;
            cols_blk.get_data()[j] = idxs_src.col;
            vals_blk.get_data()[j] = val_src;
        }
        // Writing block on result
        idxs_res.nblk = block_size_local;
        type_blk = write_chunk_blk_type(idxs_res, blk_idxs_res, rows_blk,
                                        cols_blk, vals_blk, chunk_data_res);
        rows_data_res[idxs_res.blk] = blk_idxs_res.row_frst;
        cols_data_res[idxs_res.blk] = blk_idxs_res.col_frst;
        types_data_res[idxs_res.blk] = type_blk;
        offsets_data_res[++idxs_res.blk] = idxs_res.shf;
    }
}


/**
 *  This routine makes the conversion between a block compression object
 *  and an element compression object
 */
template <typename ValueType_src, typename ValueType_res, typename IndexType>
void convert_to_bccoo_blk_elm(
    std::shared_ptr<const Executor> exec,
    const matrix::Bccoo<ValueType_src, IndexType>* source,
    matrix::Bccoo<ValueType_res, IndexType>* result)
{
    // This routine only is useful for master executor
    GKO_ASSERT(exec == exec->get_master());
    GKO_ASSERT(source->use_block_compression());
    GKO_ASSERT(result->use_element_compression());

    const IndexType* rows_data_src = source->get_const_rows();
    const size_type* offsets_data_src = source->get_const_offsets();
    const uint8* chunk_data_src = source->get_const_chunk();
    const IndexType* cols_data_src = source->get_const_cols();
    const uint8* types_data_src = source->get_const_types();

    const IndexType block_size_src = source->get_block_size();
    const size_type num_bytes_src = source->get_num_bytes();
    const IndexType num_stored_elements = source->get_num_stored_elements();

    compr_idxs<IndexType> idxs_src;
    ValueType_src val_src;

    IndexType* rows_data_res = result->get_rows();
    size_type* offsets_data_res = result->get_offsets();
    uint8* chunk_data_res = result->get_chunk();
    IndexType block_size_res = result->get_block_size();

    compr_idxs<IndexType> idxs_res;
    ValueType_res val_res;

    if (num_stored_elements > 0) {
        offsets_data_res[0] = 0;
    }
    for (IndexType i = 0; i < num_stored_elements; i += block_size_src) {
        IndexType block_size_local =
            std::min(block_size_src, num_stored_elements - i);

        compr_blk_idxs<IndexType> blk_idxs_src(rows_data_src, cols_data_src,
                                               block_size_local, idxs_src,
                                               types_data_src[idxs_src.blk]);
        for (IndexType j = 0; j < block_size_local; j++) {
            // Reading (row,col,val) from source
            get_block_position_value<IndexType, ValueType_src>(
                chunk_data_src, blk_idxs_src, idxs_src, val_src);
            // Writing (row,col,val) to result
            val_res = val_src;
            put_detect_newblock(rows_data_res, idxs_src.row - idxs_res.row,
                                idxs_res);
            IndexType col_src_res = put_position_newrow_mat_data(
                idxs_src.row, idxs_src.col, chunk_data_res, idxs_res);
            put_next_position_value(chunk_data_res, col_src_res, val_res,
                                    idxs_res);
            put_detect_endblock(offsets_data_res, block_size_res, idxs_res);
        }
        idxs_src.blk++;
        idxs_src.shf = blk_idxs_src.shf_val;
    }
    if (idxs_res.nblk > 0) {
        offsets_data_res[++idxs_res.blk] = idxs_res.shf;
    }
}


/**
 *  This routine makes the conversion between two block compression objects
 *  Additionally, finalize_op function is applied before to copy the values
 */
template <typename ValueType_src, typename ValueType_res, typename IndexType,
          typename Callable>
void convert_to_bccoo_blk_blk(
    std::shared_ptr<const Executor> exec,
    const matrix::Bccoo<ValueType_src, IndexType>* source,
    matrix::Bccoo<ValueType_res, IndexType>* result, Callable finalize_op)
{
    // This routine only is useful for master executor
    GKO_ASSERT(exec == exec->get_master());
    GKO_ASSERT(source->use_block_compression());
    GKO_ASSERT(result->use_block_compression());

    const IndexType* rows_data_src = source->get_const_rows();
    const size_type* offsets_data_src = source->get_const_offsets();
    const uint8* chunk_data_src = source->get_const_chunk();
    const IndexType* cols_data_src = source->get_const_cols();
    const uint8* types_data_src = source->get_const_types();

    const IndexType block_size_src = source->get_block_size();
    const size_type num_bytes_src = source->get_num_bytes();
    const IndexType num_stored_elements = source->get_num_stored_elements();

    compr_idxs<IndexType> idxs_src;
    ValueType_src val_src;

    IndexType* rows_data_res = result->get_rows();
    size_type* offsets_data_res = result->get_offsets();
    uint8* chunk_data_res = result->get_chunk();
    IndexType* cols_data_res = result->get_cols();
    uint8* types_data_res = result->get_types();

    const IndexType block_size_res = result->get_block_size();

    compr_idxs<IndexType> idxs_res;
    compr_blk_idxs<IndexType> blk_idxs_res;

    array<IndexType> rows_blk_res(exec, block_size_res);
    array<IndexType> cols_blk_res(exec, block_size_res);
    array<ValueType_res> vals_blk_res(exec, block_size_res);

    uint8 type_blk = {};
    IndexType i_res = 0;
    IndexType block_size_local_res =
        std::min(block_size_res, num_stored_elements - i_res);

    blk_idxs_res.row_frst = idxs_src.row;
    blk_idxs_res.col_frst = idxs_src.col;
    if (num_stored_elements > 0) {
        offsets_data_res[0] = 0;
    }
    for (IndexType i = 0; i < num_stored_elements; i += block_size_src) {
        IndexType block_size_local_src =
            std::min(block_size_src, num_stored_elements - i);
        compr_blk_idxs<IndexType> blk_idxs_src(rows_data_src, cols_data_src,
                                               block_size_local_src, idxs_src,
                                               types_data_src[idxs_src.blk]);
        for (IndexType j = 0; j < block_size_local_src; j++) {
            // Reading (row,col,val) from source
            get_block_position_value<IndexType, ValueType_src>(
                chunk_data_src, blk_idxs_src, idxs_src, val_src);
            // Analyzing the impact of (row,col,val) in the block
            proc_block_indices<IndexType>(idxs_src.row, idxs_src.col, idxs_res,
                                          blk_idxs_res);
            rows_blk_res.get_data()[idxs_res.nblk] = idxs_src.row;
            cols_blk_res.get_data()[idxs_res.nblk] = idxs_src.col;
            vals_blk_res.get_data()[idxs_res.nblk] =
                (ValueType_res)finalize_op(val_src);
            idxs_res.nblk++;
            if (idxs_res.nblk == block_size_local_res) {
                // Writing block on result
                idxs_res.nblk = block_size_local_res;
                type_blk = write_chunk_blk_type(idxs_res, blk_idxs_res,
                                                rows_blk_res, cols_blk_res,
                                                vals_blk_res, chunk_data_res);
                rows_data_res[idxs_res.blk] = blk_idxs_res.row_frst;
                cols_data_res[idxs_res.blk] = blk_idxs_res.col_frst;
                types_data_res[idxs_res.blk] = type_blk;
                offsets_data_res[++idxs_res.blk] = idxs_res.shf;
                i_res += block_size_local_res;
                block_size_local_res =
                    std::min(block_size_res, num_stored_elements - i_res);
                idxs_res.nblk = 0;
                blk_idxs_res = {};
            }
        }
        idxs_src.blk++;
        idxs_src.shf = blk_idxs_src.shf_val;
    }
}


}  // namespace bccoo
}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_MATRIX_BCCOO_MEMSIZE_CONVERT_HPP_
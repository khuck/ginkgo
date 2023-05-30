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

#ifndef GKO_PUBLIC_CORE_MATRIX_BCCOO_HPP_
#define GKO_PUBLIC_CORE_MATRIX_BCCOO_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>


namespace gko {
/**
 * @brief The matrix namespace.
 *
 * @ingroup matrix
 */
namespace matrix {


template <typename ValueType, typename IndexType>
class Coo;


template <typename ValueType, typename IndexType>
class Csr;


template <typename ValueType>
class Dense;


namespace bccoo {


enum class compression { def_value, element, block };


}


/**
 * BCCOO is a matrix format which only stores nonzero coeffficients
 * using an element or block compression.
 *
 * In both cases, the object stores blocks of consecutive elements.
 * The use of element compression allows that a specific compression
 * criteria was applied to each element of a block, whereas
 * the block compression forces that all elements in a block use
 * the same criteria, which can lead to a lower compression ratio.
 *
 * In the element compression, the elements are sorted first by row
 * and then column indexes. Then, tuples of (lev_compress, column, value)
 * are stored in a 1D array of bytes. For each tuple, lev_compress
 * determinates how the colums indexes will be stored, directly or
 * as the difference from the previous element in the same row.
 * In the case that a new row is started in the block, lev_compress
 * has an specific value, whereas column and value are void.
 * Two additional 1-D vectors complete the block structure. One of them
 * contains the starting point of each block in the array of bytes,
 * whereas the second one indicates the row index of the first tuple
 * in the block.
 *
 * In the block compression, the elements are also sorted by row
 * and column indexes, but in this case, the 1D array of bytes contains
 * up to three vectors, whose size is equal to the block size, related,
 * respectively, to the row indices, column indices and values. The row
 * index vector is ommited if all elements of the block are in the same row.
 * Both types of indices always refer to the difference to the corresponding
 * minimum index in the block, which are stored in other 1D vectors.
 * The row indexes always occupy 1 byte, whereas different compression
 * alternatives can be used for the column indices.
 * In fact, there are four additional vectors joint with the 1D array of bytes,
 * containing, respectively, the starting point of each block in the array of
 * bytes, the minimum of the row indexes in the block, the mininum of the column
 * indexes in the block, and one byte per block including its main features,
 * such as if it is s multirow block and the compression level of the column
 * indices.
 *
 * Read the next papers for more details:
 *   https://doi.org/10.1007/978-3-030-71593-9_7
 *   https://doi.org/10.1002/cpe.6515
 *
 * The BCCOO LinOp supports different operations:
 *
 * ```cpp
 * matrix::Bccoo *A, *B, *C;    // matrices
 * matrix::Dense *b, *x;        // vectors tall-and-skinny matrices
 * matrix::Dense *alpha, *beta; // scalars of dimension 1x1
 * matrix::Identity *I;         // identity matrix
 *
 * // Applying to Dense matrices computes an SpMV/SpMM product
 * A->apply(b, x)              // x = A*b
 * A->apply(alpha, b, beta, x) // x = alpha*A*b + beta*x
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 * @ingroup bccoo
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Bccoo : public EnableLinOp<Bccoo<ValueType, IndexType>>,
              public EnableCreateMethod<Bccoo<ValueType, IndexType>>,
              public ConvertibleTo<Bccoo<next_precision<ValueType>, IndexType>>,
              public ConvertibleTo<Coo<ValueType, IndexType>>,
              public ConvertibleTo<Csr<ValueType, IndexType>>,
              public ConvertibleTo<Dense<ValueType>>,
              public DiagonalExtractable<ValueType>,
              public ReadableFromMatrixData<ValueType, IndexType>,
              public WritableToMatrixData<ValueType, IndexType>,
              public EnableAbsoluteComputation<
                  remove_complex<Bccoo<ValueType, IndexType>>> {
    friend class EnableCreateMethod<Bccoo>;
    friend class EnablePolymorphicObject<Bccoo, LinOp>;
    friend class Bccoo<to_complex<ValueType>, IndexType>;

public:
    using EnableLinOp<Bccoo>::convert_to;
    using EnableLinOp<Bccoo>::move_to;
    using ConvertibleTo<
        Bccoo<next_precision<ValueType>, IndexType>>::convert_to;
    using ConvertibleTo<Bccoo<next_precision<ValueType>, IndexType>>::move_to;
    using ConvertibleTo<Coo<ValueType, IndexType>>::convert_to;
    using ConvertibleTo<Coo<ValueType, IndexType>>::move_to;
    using ConvertibleTo<Csr<ValueType, IndexType>>::convert_to;
    using ConvertibleTo<Csr<ValueType, IndexType>>::move_to;
    using ConvertibleTo<Dense<ValueType>>::convert_to;
    using ConvertibleTo<Dense<ValueType>>::move_to;
    using ReadableFromMatrixData<ValueType, IndexType>::read;

    using value_type = ValueType;
    using index_type = IndexType;
    using mat_data = matrix_data<ValueType, IndexType>;
    using absolute_type = remove_complex<Bccoo>;

    friend class Bccoo<next_precision<ValueType>, IndexType>;

    void convert_to(Bccoo<ValueType, IndexType>* result) const override;

    void move_to(Bccoo<ValueType, IndexType>* result) override;

    void convert_to(
        Bccoo<next_precision<ValueType>, IndexType>* result) const override;

    void move_to(Bccoo<next_precision<ValueType>, IndexType>* result) override;

    void convert_to(Coo<ValueType, IndexType>* other) const override;

    void move_to(Coo<ValueType, IndexType>* other) override;

    void convert_to(Csr<ValueType, IndexType>* other) const override;

    void move_to(Csr<ValueType, IndexType>* other) override;

    void convert_to(Dense<ValueType>* other) const override;

    void move_to(Dense<ValueType>* other) override;

    void read(const mat_data& data) override;

    void write(mat_data& data) const override;

    std::unique_ptr<Diagonal<ValueType>> extract_diagonal() const override;

    std::unique_ptr<absolute_type> compute_absolute() const override;

    void compute_absolute_inplace() override;


    /**
     * Returns the minimum row indices of the first element of each block.
     *
     * @return the minimum row indices of the first element of each block.
     */
    index_type* get_rows() noexcept { return rows_.get_data(); }

    /**
     * @copydoc Bccoo::get_rows()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type* get_const_rows() const noexcept
    {
        return rows_.get_const_data();
    }

    /**
     * Returns the minimum col indices of the first element of each block.
     * Only for block compression.
     *
     * @return the minimum col indices of the first element of each block.
     * Only for block compression.
     */
    index_type* get_cols() noexcept { return cols_.get_data(); }

    /**
     * @copydoc Bccoo::get_cols()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant
     *       version, so always prefer this version.
     */
    const index_type* get_const_cols() const noexcept
    {
        return cols_.get_const_data();
    }

    /**
     * Returns the type indices of the first element of each block.
     * Only for block compression.
     *
     * @return the type indices of the first element of each block.
     * Only for block compression.
     */
    uint8* get_types() noexcept { return types_.get_data(); }

    /**
     * @copydoc Bccoo::get_types()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant
     *       version, so always prefer this version.
     */
    const uint8* get_const_types() const noexcept
    {
        return types_.get_const_data();
    }

    /**
     * Returns the offsets related to the first entry of each block.
     *
     * @return the offsets related to the first entry of each block.
     */
    size_type* get_offsets() noexcept { return offsets_.get_data(); }

    /**
     * @copydoc Bccoo::get_offsets()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const size_type* get_const_offsets() const noexcept
    {
        return offsets_.get_const_data();
    }

    /**
     * Returns the vector where the data of each block are stored.
     *
     * @return the vector where the data of each block are stored.
     */
    uint8* get_chunk() noexcept { return chunk_.get_data(); }

    /**
     * @copydoc Bccoo::get_chunk()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const uint8* get_const_chunk() const noexcept
    {
        return chunk_.get_const_data();
    }

    /**
     * Returns the number of elements explicitly stored in the matrix.
     *
     * @return the number of elements explicitly stored in the matrix
     */
    index_type get_num_stored_elements() const noexcept
    {
        return num_nonzeros_;
    }

    /**
     * Returns the block size used in the definition of the matrix.
     *
     * @return the block size used in the definition of the matrix.
     */
    index_type get_block_size() const noexcept { return block_size_; }

    /**
     * Returns the number of blocks used in the definition of the matrix.
     *
     * @return the number of blocks used in the definition of the matrix.
     */
    index_type get_num_blocks() const noexcept { return rows_.get_num_elems(); }

    /**
     * Returns the number of bytes of chunk vector used in the definition of the
     * matrix.
     *
     * @return the number of bytes of chunk vector used in the definition of the
     * matrix.
     */
    size_type get_num_bytes() const noexcept { return chunk_.get_num_elems(); }

    /**
     * Returns the compression used in the definition of the matrix.
     *
     * @return the compression used in the definition of the matrix.
     */
    bccoo::compression get_compression() const noexcept { return compression_; }

    /**
     * Returns if the default compression is used
     *
     * @returns if the default compression is used
     */
    bool use_default_compression() const noexcept
    {
        return compression_ == bccoo::compression::def_value;
    }

    /**
     * Returns if the element compression is used
     *
     * @returns if the element compression is used
     */
    bool use_element_compression() const noexcept
    {
        return compression_ == bccoo::compression::element;
    }

    /**
     * Returns if the block compression is used
     *
     * @returns if the block compression is used
     */
    bool use_block_compression() const noexcept
    {
        return compression_ == bccoo::compression::block;
    }

    /**
     * Returns if the object is initialized
     *
     * @returns if the object is initialized
     */
    bool is_initialized() const
    {
        return (compression_ != bccoo::compression::def_value &&
                block_size_ > 0);
    }


    /**
     * Applies Bccoo matrix axpy to a vector (or a sequence of vectors).
     *
     * Performs the operation x = Bccoo * b + x
     *
     * @param b  the input vector(s) on which the operator is applied
     * @param x  the output vector(s) where the result is stored
     *
     * @return this
     */
    LinOp* apply2(const LinOp* b, LinOp* x)
    {
        this->validate_application_parameters(b, x);
        auto exec = this->get_executor();
        this->apply2_impl(make_temporary_clone(exec, b).get(),
                          make_temporary_clone(exec, x).get());
        return this;
    }

    /**
     * @copydoc apply2(const LinOp *, LinOp *)
     */
    const LinOp* apply2(const LinOp* b, LinOp* x) const
    {
        this->validate_application_parameters(b, x);
        auto exec = this->get_executor();
        this->apply2_impl(make_temporary_clone(exec, b).get(),
                          make_temporary_clone(exec, x).get());
        return this;
    }

    /**
     * Performs the operation x = alpha * Bccoo * b + x.
     *
     * @param alpha  scaling of the result of Bccoo * b
     * @param b  vector(s) on which the operator is applied
     * @param x  output vector(s)
     *
     * @return this
     */
    LinOp* apply2(const LinOp* alpha, const LinOp* b, LinOp* x)
    {
        this->validate_application_parameters(b, x);
        GKO_ASSERT_EQUAL_DIMENSIONS(alpha, dim<2>(1, 1));
        auto exec = this->get_executor();
        this->apply2_impl(make_temporary_clone(exec, alpha).get(),
                          make_temporary_clone(exec, b).get(),
                          make_temporary_clone(exec, x).get());
        return this;
    }

    /**
     * @copydoc apply2(const LinOp *, const LinOp *, LinOp *)
     */
    const LinOp* apply2(const LinOp* alpha, const LinOp* b, LinOp* x) const
    {
        this->validate_application_parameters(b, x);
        GKO_ASSERT_EQUAL_DIMENSIONS(alpha, dim<2>(1, 1));
        auto exec = this->get_executor();
        this->apply2_impl(make_temporary_clone(exec, alpha).get(),
                          make_temporary_clone(exec, b).get(),
                          make_temporary_clone(exec, x).get());
        return this;
    }

protected:
    /**
     * Creates an empty BCCOO matrix
     *
     * @param exec  Executor associated to the matrix
     */
    Bccoo(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Bccoo>(exec, dim<2>{}),
          rows_(exec, 0),
          cols_(exec, 0),
          types_(exec, 0),
          offsets_(exec, 0),
          chunk_(exec, 0),
          num_nonzeros_{0},
          block_size_{0},
          compression_{bccoo::compression::def_value}
    {}

    /**
     * Creates an empty BCCOO matrix, fixing compression level and blocksize
     *
     * @param exec  Executor associated to the matrix
     */
    Bccoo(std::shared_ptr<const Executor> exec, index_type block_size,
          bccoo::compression compression)
        : EnableLinOp<Bccoo>(exec, dim<2>{}),
          rows_(exec, 0),
          cols_(exec, 0),
          types_(exec, 0),
          offsets_(exec, 0),
          chunk_(exec, 0),
          num_nonzeros_{0},
          block_size_{block_size},
          compression_{compression}
    {}

    /**
     * Creates an uninitialized BCCOO matrix of the specified sizes.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param num_nonzeros  number of nonzeros
     * @param block_size    number of nonzeros in each block
     * @param num_bytes     number of bytes
     * @param compression   compression used in the definition
     */
    Bccoo(std::shared_ptr<const Executor> exec, const dim<2>& size,
          index_type num_nonzeros, index_type block_size, size_type num_bytes,
          bccoo::compression compression)
        : EnableLinOp<Bccoo>(exec, size),
          rows_(exec,
                (block_size == 0) ? 0 : ceildiv(num_nonzeros, block_size)),
          cols_(exec, ((compression == bccoo::compression::element) ||
                       (block_size == 0))
                          ? 0
                          : ceildiv(num_nonzeros, block_size)),
          types_(exec, ((compression == bccoo::compression::element) ||
                        (block_size == 0))
                           ? 0
                           : ceildiv(num_nonzeros, block_size)),
          offsets_(exec, (block_size == 0)
                             ? 0
                             : ceildiv(num_nonzeros, block_size) + 1),
          chunk_(exec, num_bytes),
          num_nonzeros_{num_nonzeros},
          block_size_{block_size},
          compression_{compression}
    {}

    /**
     * Creates an element compression variant of the BCCOO matrix from
     * already allocated (and initialized) rows, offsets and chunk arrays.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param chunk  array of matrix indexes and matrix values
     * @param offsets  array of positions of the first entry of each block in
     *                 chunk array
     * @param rows  array of row index of the first entry of each block in
     *              chunk array
     * @param num_nonzeros  number of nonzeros
     * @param block_size    number of nonzeros in each block
     *
     * @note If one of `chunk`, `offsets` or `rows` is not an rvalue, not
     *       an array of uint8, IndexType or IndexType, respectively, or
     *       is on the wrong executor, an internal copy of that array will be
     *       created, and the original array chunk will not be used in the
     *       matrix.
     */
    Bccoo(std::shared_ptr<const Executor> exec, const dim<2>& size,
          array<uint8> chunk, array<size_type> offsets, array<IndexType> rows,
          index_type num_nonzeros, index_type block_size)
        : EnableLinOp<Bccoo>(exec, size),
          chunk_{exec, std::move(chunk)},
          offsets_{exec, std::move(offsets)},
          rows_{exec, std::move(rows)},
          num_nonzeros_{num_nonzeros},
          block_size_{block_size},
          compression_{bccoo::compression::element}
    {
        GKO_ASSERT_EQ(rows_.get_num_elems() + 1, offsets_.get_num_elems());
    }

    /**
     * Creates a block compression variant of the BCCOO matrix from already
     * allocated (and initialized) rows, cols, types, offsets and chunk arrays.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param chunk   array of matrix indexes and matrix values
     * @param offsets array of positions of the first entry of each block in
     *                chunk array
     * @param types   array of compression type for each block in
     *                chunk array
     * @param cols    array of minimum column indices for each block in
     *                chunk array
     * @param rows    array of minimum row indices for each block in
     *                chunk array
     * @param num_nonzeros  number of nonzeros
     * @param block_size    number of nonzeros in each block
     *
     * @note If one of `chunk`, `offsets`, `types`, `cols` or `rows` is not
     * 			 an rvalue, not an array of uint8, IndexType, uint8,
     *       IndexType or IndexType, respectively, or is on the wrong executor,
     *       an internal copy of that array will be created, and the original
     *       array will not be used in the matrix.
     */
    Bccoo(std::shared_ptr<const Executor> exec, const dim<2>& size,
          array<uint8> chunk, array<size_type> offsets, array<uint8> types,
          array<IndexType> cols, array<IndexType> rows, index_type num_nonzeros,
          index_type block_size)
        : EnableLinOp<Bccoo>(exec, size),
          chunk_{exec, std::move(chunk)},
          offsets_{exec, std::move(offsets)},
          types_{exec, std::move(types)},
          cols_{exec, std::move(cols)},
          rows_{exec, std::move(rows)},
          num_nonzeros_{num_nonzeros},
          block_size_{block_size},
          compression_{bccoo::compression::block}
    {
        GKO_ASSERT_EQ(rows_.get_num_elems() + 1, offsets_.get_num_elems());
        GKO_ASSERT_EQ(rows_.get_num_elems(), cols_.get_num_elems());
        GKO_ASSERT_EQ(rows_.get_num_elems(), types_.get_num_elems());
    }

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    void apply2_impl(const LinOp* b, LinOp* x) const;

    void apply2_impl(const LinOp* alpha, const LinOp* b, LinOp* x) const;

private:
    array<index_type> rows_;
    array<index_type> cols_;
    array<uint8> types_;
    array<size_type> offsets_;
    array<uint8> chunk_;
    index_type block_size_;
    index_type num_nonzeros_;
    bccoo::compression compression_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_BCCOO_HPP_
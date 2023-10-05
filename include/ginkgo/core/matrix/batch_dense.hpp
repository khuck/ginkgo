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

#ifndef GKO_PUBLIC_CORE_MATRIX_BATCH_DENSE_HPP_
#define GKO_PUBLIC_CORE_MATRIX_BATCH_DENSE_HPP_


#include <initializer_list>
#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace batch {
namespace matrix {


/**
 * Dense is a batch matrix format which explicitly stores all values of the
 * matrix in each of the batches.
 *
 * The values in each of the batches are stored in row-major format (values
 * belonging to the same row appear consecutive in the memory). Optionally, rows
 * can be padded for better memory access.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @note While this format is not very useful for storing sparse matrices, it
 *       is often suitable to store vectors, and sets of vectors.
 * @ingroup batch_dense
 * @ingroup mat_formats
 * @ingroup BatchLinOp
 */
template <typename ValueType = default_precision>
class Dense : public EnableBatchLinOp<Dense<ValueType>>,
              public EnableCreateMethod<Dense<ValueType>>,
              public ConvertibleTo<Dense<next_precision<ValueType>>> {
    friend class EnableCreateMethod<Dense>;
    friend class EnablePolymorphicObject<Dense, BatchLinOp>;
    friend class Dense<to_complex<ValueType>>;
    friend class Dense<next_precision<ValueType>>;

public:
    using EnableBatchLinOp<Dense>::convert_to;
    using EnableBatchLinOp<Dense>::move_to;

    using value_type = ValueType;
    using index_type = int32;
    using transposed_type = Dense<ValueType>;
    using unbatch_type = gko::matrix::Dense<ValueType>;
    using absolute_type = remove_complex<Dense>;
    using complex_type = to_complex<Dense>;

    /**
     * Creates a Dense matrix with the configuration of another Dense
     * matrix.
     *
     * @param other  The other matrix whose configuration needs to copied.
     */
    static std::unique_ptr<Dense> create_with_config_of(
        ptr_param<const Dense> other);

    void convert_to(Dense<next_precision<ValueType>>* result) const override;

    void move_to(Dense<next_precision<ValueType>>* result) override;


    /**
     * Creates a mutable view (of matrix::Dense type) of one item of the Batch
     * MultiVector<value_type> object. Does not perform any deep copies, but
     * only returns a view of the data.
     *
     * @param item_id  The index of the batch item
     *
     * @return  a matrix::Dense object with the data from the batch item at the
     *          given index.
     */
    std::unique_ptr<unbatch_type> create_view_for_item(size_type item_id);

    /**
     * @copydoc create_view_for_item(size_type)
     */
    std::unique_ptr<const unbatch_type> create_const_view_for_item(
        size_type item_id) const;

    /**
     * Returns a pointer to the array of values of the multi-vector
     *
     * @return the pointer to the array of values
     */
    value_type* get_values() noexcept { return values_.get_data(); }

    /**
     * @copydoc get_values()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type* get_const_values() const noexcept
    {
        return values_.get_const_data();
    }

    /**
     * Returns a single element for a particular batch item.
     *
     * @param batch_id  the batch item index to be queried
     * @param row  the row of the requested element
     * @param col  the column of the requested element
     *
     * @note  the method has to be called on the same Executor the vector is
     *        stored at (e.g. trying to call this method on a GPU multi-vector
     *        from the OMP results in a runtime error)
     */
    value_type& at(size_type batch_id, size_type row, size_type col)
    {
        GKO_ASSERT(batch_id < this->get_num_batch_items());
        return values_.get_data()[linearize_index(batch_id, row, col)];
    }

    /**
     * @copydoc MultiVector::at(size_type, size_type, size_type)
     */
    value_type at(size_type batch_id, size_type row, size_type col) const
    {
        GKO_ASSERT(batch_id < this->get_num_batch_items());
        return values_.get_const_data()[linearize_index(batch_id, row, col)];
    }

    /**
     * Returns a single element for a particular batch item.
     *
     * Useful for iterating across all elements of the vector.
     * However, it is less efficient than the two-parameter variant of this
     * method.
     *
     * @param batch_id  the batch item index to be queried
     * @param idx  a linear index of the requested element
     *
     * @note  the method has to be called on the same Executor the vector is
     *        stored at (e.g. trying to call this method on a GPU multi-vector
     *        from the OMP results in a runtime error)
     */
    ValueType& at(size_type batch_id, size_type idx) noexcept
    {
        return values_.get_data()[linearize_index(batch_id, idx)];
    }

    /**
     * @copydoc MultiVector::at(size_type, size_type, size_type)
     */
    ValueType at(size_type batch_id, size_type idx) const noexcept
    {
        return values_.get_const_data()[linearize_index(batch_id, idx)];
    }

    /**
     * Returns a pointer to the array of values of the multi-vector for a
     * specific batch item.
     *
     * @param batch_id  the id of the batch item.
     *
     * @return the pointer to the array of values
     */
    value_type* get_values_for_item(size_type batch_id) noexcept
    {
        GKO_ASSERT(batch_id < this->get_num_batch_items());
        return values_.get_data() +
               this->get_size().get_cumulative_offset(batch_id);
    }

    /**
     * @copydoc get_values_for_item(size_type)
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type* get_const_values_for_item(
        size_type batch_id) const noexcept
    {
        GKO_ASSERT(batch_id < this->get_num_batch_items());
        return values_.get_const_data() +
               this->get_size().get_cumulative_offset(batch_id);
    }

    /**
     * Returns the number of elements explicitly stored in the batch matrix,
     * cumulative across all the batch items.
     *
     * @return the number of elements explicitly stored in the vector,
     *         cumulative across all the batch items
     */
    size_type get_num_stored_elements() const noexcept
    {
        return values_.get_num_elems();
    }


    /**
     * Creates a constant (immutable) batch dense matrix from a constant
     * array.
     *
     * @param exec  the executor to create the vector on
     * @param size  the dimensions of the vector
     * @param values  the value array of the vector
     *
     * @return A smart pointer to the constant multi-vector wrapping the input
     * array (if it resides on the same executor as the vector) or a copy of the
     * array on the correct executor.
     */
    static std::unique_ptr<const Dense<value_type>> create_const(
        std::shared_ptr<const Executor> exec, const batch_dim<2>& sizes,
        gko::detail::const_array_view<ValueType>&& values);


    void apply(const MultiVector<value_type>* b,
               MultiVector<value_type>* x) const
    {
        this->apply_impl(b, x);
    }

    void apply(const MultiVector<value_type>* alpha,
               const MultiVector<value_type>* b,
               const MultiVector<value_type>* beta,
               MultiVector<value_type>* x) const
    {
        this->apply_impl(alpha, b, beta, x);
    }

private:
    inline size_type compute_num_elems(const batch_dim<2>& size)
    {
        return size.get_cumulative_offset(size.get_num_batch_items());
    }

protected:
    /**
     * Creates an uninitialized Dense matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     */
    Dense(std::shared_ptr<const Executor> exec,
          const batch_dim<2>& size = batch_dim<2>{});

    /**
     * Creates a Dense matrix from an already allocated (and initialized)
     * array.
     *
     * @tparam ValuesArray  type of array of values
     *
     * @param exec  Executor associated to the matrix
     * @param size  sizes of the batch matrices in a batch_dim object
     * @param values  array of matrix values
     * @param strides  stride of the rows (i.e. offset between the first
     *                  elements of two consecutive rows, expressed as the
     *                  number of matrix elements)
     *
     * @note If `values` is not an rvalue, not an array of ValueType, or is on
     *       the wrong executor, an internal copy will be created, and the
     *       original array data will not be used in the matrix.
     */
    template <typename ValuesArray>
    Dense(std::shared_ptr<const Executor> exec, const batch_dim<2>& size,
          ValuesArray&& values)
        : EnableBatchLinOp<Dense>(exec, size),
          values_{exec, std::forward<ValuesArray>(values)}
    {
        // Ensure that the values array has the correct size
        auto num_elems = compute_num_elems(size);
        GKO_ENSURE_IN_BOUNDS(num_elems, values_.get_num_elems() + 1);
    }

    /**
     * Creates a Dense matrix with the same configuration as the callers
     * matrix.
     *
     * @returns a Dense matrix with the same configuration as the caller.
     */
    std::unique_ptr<Dense> create_with_same_config() const;

    virtual void apply_impl(const MultiVector<value_type>* b,
                            MultiVector<value_type>* x) const;

    virtual void apply_impl(const MultiVector<value_type>* alpha,
                            const MultiVector<value_type>* b,
                            const MultiVector<value_type>* beta,
                            MultiVector<value_type>* x) const;

    size_type linearize_index(size_type batch, size_type row,
                              size_type col) const noexcept
    {
        return this->get_size().get_cumulative_offset(batch) +
               row * this->get_size().get_common_size()[1] + col;
    }

    size_type linearize_index(size_type batch, size_type idx) const noexcept
    {
        return linearize_index(batch, idx / this->get_common_size()[1],
                               idx % this->get_common_size()[1]);
    }

private:
    array<value_type> values_;
};


}  // namespace matrix
}  // namespace batch
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_BATCH_DENSE_HPP_

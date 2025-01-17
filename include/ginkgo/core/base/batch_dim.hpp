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

#ifndef GKO_PUBLIC_CORE_BASE_BATCH_DIM_HPP_
#define GKO_PUBLIC_CORE_BASE_BATCH_DIM_HPP_


#include <iostream>


#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {


/**
 * A type representing the dimensions of a multidimensional batch object.
 *
 * @tparam Dimensionality  number of dimensions of the object
 * @tparam DimensionType  datatype used to represent each dimension
 *
 * @ingroup batch_dim
 */
template <size_type Dimensionality = 2, typename DimensionType = size_type>
struct batch_dim {
    static constexpr size_type dimensionality = Dimensionality;
    using dimension_type = DimensionType;

    /**
     * Get the number of batch items stored
     *
     * @return num_batch_items
     */
    size_type get_num_batch_items() const { return num_batch_items_; }

    /**
     * Get the common size of the batch items
     *
     * @return common_size
     */
    dim<dimensionality, dimension_type> get_common_size() const
    {
        return common_size_;
    }

    /**
     * Checks if two batch_dim objects are equal.
     *
     * @param x  first object
     * @param y  second object
     *
     * @return true if and only if all dimensions of both objects are equal.
     */
    friend bool operator==(const batch_dim& x, const batch_dim& y)
    {
        return x.num_batch_items_ == y.num_batch_items_ &&
               x.common_size_ == y.common_size_;
    }


    /**
     * Checks if two batch_dim objects are different.
     *
     * @tparam Dimensionality  number of dimensions of the dim objects
     * @tparam DimensionType  datatype used to represent each dimension
     *
     * @param x  first object
     * @param y  second object
     *
     * @return `!(x == y)`
     */
    friend bool operator!=(const batch_dim<Dimensionality, DimensionType>& x,
                           const batch_dim<Dimensionality, DimensionType>& y)
    {
        return !(x == y);
    }


    /**
     * The default constructor
     */
    batch_dim()
        : common_size_(dim<dimensionality, dimension_type>{}),
          num_batch_items_(0)
    {}

    /**
     * Creates a batch_dim object which stores a uniform size for all batch
     * entries.
     *
     * @param num_batch_items  the number of batch items to be stored
     * @param common_size  the common size of all the batch items stored
     *
     * @note  Use this constructor when uniform batches need to be stored.
     */
    explicit batch_dim(const size_type num_batch_items,
                       const dim<dimensionality, dimension_type>& common_size)
        : common_size_(common_size), num_batch_items_(num_batch_items)
    {}

private:
    size_type num_batch_items_{};
    dim<dimensionality, dimension_type> common_size_{};
};


/**
 * Returns a batch_dim object with its dimensions swapped for batched operators
 *
 * @tparam DimensionType  datatype used to represent each dimension
 *
 * @param dimensions original object
 *
 * @return a batch_dim object with dimensions swapped
 */
template <typename DimensionType>
inline batch_dim<2, DimensionType> transpose(
    const batch_dim<2, DimensionType>& input)
{
    return batch_dim<2, DimensionType>(input.get_num_batch_items(),
                                       transpose(input.get_common_size()));
}


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_BATCH_DIM_HPP_

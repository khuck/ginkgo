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

#include <ginkgo/core/matrix/batch_dense.hpp>


#include <algorithm>
#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/batch_dense_kernels.hpp"


namespace gko {
namespace batch {
namespace matrix {
namespace dense {
namespace {


GKO_REGISTER_OPERATION(simple_apply, batch_dense::simple_apply);
GKO_REGISTER_OPERATION(advanced_apply, batch_dense::advanced_apply);


}  // namespace
}  // namespace dense


template <typename ValueType>
std::unique_ptr<gko::matrix::Dense<ValueType>>
Dense<ValueType>::create_view_for_item(size_type item_id)
{
    auto exec = this->get_executor();
    auto num_rows = this->get_common_size()[0];
    auto stride = this->get_common_size()[1];
    auto mat = unbatch_type::create(
        exec, this->get_common_size(),
        make_array_view(exec, num_rows * stride,
                        this->get_values_for_item(item_id)),
        stride);
    return mat;
}


template <typename ValueType>
std::unique_ptr<const gko::matrix::Dense<ValueType>>
Dense<ValueType>::create_const_view_for_item(size_type item_id) const
{
    auto exec = this->get_executor();
    auto num_rows = this->get_common_size()[0];
    auto stride = this->get_common_size()[1];
    auto mat = unbatch_type::create_const(
        exec, this->get_common_size(),
        make_const_array_view(exec, num_rows * stride,
                              this->get_const_values_for_item(item_id)),
        stride);
    return mat;
}


template <typename ValueType>
std::unique_ptr<Dense<ValueType>> Dense<ValueType>::create_with_config_of(
    ptr_param<const Dense<ValueType>> other)
{
    return Dense<ValueType>::create(other->get_executor(), other->get_size());
}


template <typename ValueType>
std::unique_ptr<const Dense<ValueType>> Dense<ValueType>::create_const(
    std::shared_ptr<const Executor> exec, const batch_dim<2>& sizes,
    gko::detail::const_array_view<ValueType>&& values)
{
    // cast const-ness away, but return a const object afterwards,
    // so we can ensure that no modifications take place.
    return std::unique_ptr<const Dense>(new Dense{
        exec, sizes, gko::detail::array_const_cast(std::move(values))});
}


template <typename ValueType>
Dense<ValueType>::Dense(std::shared_ptr<const Executor> exec,
                        const batch_dim<2>& size)
    : EnableBatchLinOp<Dense<ValueType>>(exec, size),
      values_(exec, compute_num_elems(size))
{}


template <typename ValueType>
void Dense<ValueType>::apply_impl(const MultiVector<ValueType>* b,
                                  MultiVector<ValueType>* x) const
{
    this->validate_application_parameters(b, x);
    this->get_executor()->run(dense::make_simple_apply(this, b, x));
}


template <typename ValueType>
void Dense<ValueType>::apply_impl(const MultiVector<ValueType>* alpha,
                                  const MultiVector<ValueType>* b,
                                  const MultiVector<ValueType>* beta,
                                  MultiVector<ValueType>* x) const
{
    this->validate_application_parameters(alpha, b, beta, x);
    this->get_executor()->run(
        dense::make_advanced_apply(alpha, this, b, beta, x));
}


template <typename ValueType>
void Dense<ValueType>::convert_to(
    Dense<next_precision<ValueType>>* result) const
{
    result->values_ = this->values_;
    result->set_size(this->get_size());
}


template <typename ValueType>
void Dense<ValueType>::move_to(Dense<next_precision<ValueType>>* result)
{
    this->convert_to(result);
}


#define GKO_DECLARE_BATCH_DENSE_MATRIX(_type) class Dense<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_MATRIX);


}  // namespace matrix
}  // namespace batch
}  // namespace gko

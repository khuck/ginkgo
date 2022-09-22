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

#ifndef GKO_CORE_DISTRIBUTED_HELPERS_HPP_
#define GKO_CORE_DISTRIBUTED_HELPERS_HPP_


#include <memory>


#include <ginkgo/config.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/dispatch_helper.hpp"


namespace gko {
namespace detail {


template <typename ValueType>
std::unique_ptr<matrix::Dense<ValueType>> create_with_config_of(
    const matrix::Dense<ValueType>* mtx)
{
    return matrix::Dense<ValueType>::create(mtx->get_executor(),
                                            mtx->get_size(), mtx->get_stride());
}


template <typename ValueType>
const matrix::Dense<ValueType>* get_local(const matrix::Dense<ValueType>* mtx)
{
    return mtx;
}


template <typename ValueType>
matrix::Dense<ValueType>* get_local(matrix::Dense<ValueType>* mtx)
{
    return mtx;
}


#if GINKGO_BUILD_MPI


template <typename ValueType>
std::unique_ptr<experimental::distributed::Vector<ValueType>>
create_with_config_of(const experimental::distributed::Vector<ValueType>* mtx)
{
    return experimental::distributed::Vector<ValueType>::create(
        mtx->get_executor(), mtx->get_communicator(), mtx->get_size(),
        mtx->get_local_vector()->get_size(),
        mtx->get_local_vector()->get_stride());
}


template <typename ValueType>
matrix::Dense<ValueType>* get_local(
    experimental::distributed::Vector<ValueType>* mtx)
{
    return const_cast<matrix::Dense<ValueType>*>(mtx->get_local_vector());
}


template <typename ValueType>
const matrix::Dense<ValueType>* get_local(
    const experimental::distributed::Vector<ValueType>* mtx)
{
    return mtx->get_local_vector();
}


#endif


template <typename Arg>
bool is_distributed(Arg* linop)
{
#if GINKGO_BUILD_MPI
    return dynamic_cast<const experimental::distributed::DistributedBase*>(
        linop);
#else
    return false;
#endif
}


template <typename Arg, typename... Rest>
bool is_distributed(Arg* linop, Rest*... rest)
{
#if GINKGO_BUILD_MPI
    bool is_distributed_value =
        dynamic_cast<const experimental::distributed::DistributedBase*>(linop);
    GKO_ASSERT(is_distributed_value == is_distributed(rest...));
    return is_distributed_value;
#else
    return false;
#endif
}


/**
 * Cast an input linop to the correct underlying vector type (dense/distributed)
 * and passes it to the given function.
 *
 * @tparam ValueType  The value type of the underlying dense or distributed
 * vector.
 * @tparam T  The linop type, either LinOp, or const LinOp.
 * @tparam F  The function type.
 * @tparam Args  The types for the additional arguments of f.
 *
 * @param linop  The linop to be casted into either a dense or distributed
 *               vector.
 * @param f  The function that is to be called with the correctly casted linop.
 * @param args  The additional arguments of f.
 */
template <typename ValueType, typename T, typename F, typename... Args>
void vector_dispatch(T* linop, F&& f, Args&&... args)
{
#if GINKGO_BUILD_MPI
    if (is_distributed(linop)) {
        using type = std::conditional_t<
            std::is_const<T>::value,
            const experimental::distributed::Vector<ValueType>,
            experimental::distributed::Vector<ValueType>>;
        f(dynamic_cast<type*>(linop), std::forward<Args>(args)...);
    } else
#endif
    {
        using type = std::conditional_t<std::is_const<T>::value,
                                        const matrix::Dense<ValueType>,
                                        matrix::Dense<ValueType>>;
        if (auto concrete_linop = dynamic_cast<type*>(linop)) {
            f(concrete_linop, std::forward<Args>(args)...);
        } else {
            GKO_NOT_SUPPORTED(linop);
        }
    }
}


/**
 * Helper to extract a submatrix.
 *
 * @note  global_size is unused, since it can be inferred from rows and cols.
 */
template <typename ValueType>
std::unique_ptr<matrix::Dense<ValueType>> create_submatrix_helper(
    matrix::Dense<ValueType>* mtx, dim<2> global_size, span rows, span cols)
{
    return mtx->create_submatrix(rows, cols);
}


#if GINKGO_BUILD_MPI


/**
 * Helper to extract a submatrix.
 *
 * @param global_size  the global_size of the submatrix
 * @param rows  the rows of the submatrix in local indices
 * @param cols  the columns of the submatrix in local indices
 */
template <typename ValueType>
std::unique_ptr<experimental::distributed::Vector<ValueType>>
create_submatrix_helper(experimental::distributed::Vector<ValueType>* mtx,
                        dim<2> global_size, span rows, span cols)
{
    const auto exec = mtx->get_executor();
    auto local_view = matrix::Dense<ValueType>::create(
        exec, mtx->get_local_vector()->get_size(),
        make_array_view(exec,
                        mtx->get_local_vector()->get_num_stored_elements(),
                        mtx->get_local_values()),
        mtx->get_local_vector()->get_stride());
    return experimental::distributed::Vector<ValueType>::create(
        exec, mtx->get_communicator(), global_size,
        local_view->create_submatrix(rows, cols));
}


#endif


}  // namespace detail
}  // namespace gko


#endif  // GKO_CORE_DISTRIBUTED_HELPERS_HPP_

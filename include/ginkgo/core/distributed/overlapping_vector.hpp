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

#ifndef OVERLAPPING_VECTOR_OVERLAPPING_VECTOR_HPP
#define OVERLAPPING_VECTOR_OVERLAPPING_VECTOR_HPP

#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/lin_op.hpp>
#include <ginkgo/core/distributed/overlapping_partition.hpp>
#include <ginkgo/core/distributed/sparse_communicator.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include <variant>


namespace gko::experimental::distributed {


enum class partition_of_unity {
    unique,  // alternative: restricted, shared indices are owned by exactly one
             // process
    shared,  // alternative: additive, shared indices are mutually owned by all
             // involved processes
};


template <typename ValueType>
struct overlapping_vector : public EnableLinOp<overlapping_vector<ValueType>> {
    using value_type = ValueType;
    using local_vector_type = matrix::Dense<value_type>;

    size_type get_stride() const { return buffer_->get_stride(); }

    size_type get_num_stored_elems() const
    {
        return buffer_->get_num_stored_elements();
    }

    auto make_consistent(transformation mode)
    {
        return sparse_comm_->communicate(as_dense(), mode);
    }

    template <typename F, typename = std::enable_if_t<
                              std::is_invocable_v<F, double, double>>>
    void make_consistent(F&& transformation);

    /**
     * could add non-const versions with custom deleter to write back changes
     */
    std::unique_ptr<const local_vector_type> extract_local()
    {
        return get_local(as_dense().get(), sparse_comm_->get_partition().get());
    }

    std::unique_ptr<const local_vector_type> extract_non_local()
    {
        return get_non_local(as_dense().get(),
                             sparse_comm_->get_partition().get());
    }

    void apply_impl(const LinOp* b, LinOp* x) const override {}
    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {}

    // needs to be shared, because mpi::request might outlive this
    template <typename... Args>
    static std::shared_ptr<overlapping_vector> create(Args&&... args)
    {
        return std::shared_ptr<overlapping_vector>{
            new overlapping_vector{std::forward<Args>(args)...}};
    }

    overlapping_vector(std::shared_ptr<const Executor> exec,
                       std::shared_ptr<sparse_communicator> sparse_comm = {},
                       std::unique_ptr<local_vector_type> local_vector = {})
        : EnableLinOp<overlapping_vector<ValueType>>(
              exec, {sparse_comm->get_partition()->get_size(), 1}),
          sparse_comm_(std::move(sparse_comm)),
          buffer_(local_vector_type::create(exec))
    {
        buffer_->move_from(std::move(local_vector));
    }

    std::shared_ptr<const local_vector_type> as_dense() const
    {
        return buffer_;
    }

    std::shared_ptr<local_vector_type> as_dense() { return buffer_; }

    std::shared_ptr<sparse_communicator> sparse_comm_;
    // contains local+nonlocal values
    // might switch to dense directly
    std::shared_ptr<local_vector_type> buffer_;
};


}  // namespace gko::experimental::distributed

#endif
#endif  // OVERLAPPING_VECTOR_OVERLAPPING_VECTOR_HPP

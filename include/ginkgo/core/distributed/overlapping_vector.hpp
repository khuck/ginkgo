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
#include <ginkgo/core/matrix/dense.hpp>


#include <variant>


namespace gko::experimental::distributed {

enum class transformation { set, add };

/**
 * maybe allow for processes owning multiple parts by mapping target_ids_ to
 * rank?
 */
mpi::communicator create_neighborhood_comm(
    mpi::communicator base, const array<comm_index_type>& sources,
    const array<comm_index_type>& destinations)
{
    auto in_degree = static_cast<comm_index_type>(sources.get_num_elems());
    auto out_degree =
        static_cast<comm_index_type>(destinations.get_num_elems());

    auto sources_host =
        make_temporary_clone(sources.get_executor()->get_master(), &sources);
    auto destinations_host = make_temporary_clone(
        destinations.get_executor()->get_master(), &destinations);

    // adjacent constructor guarantees that querying sources/destinations
    // will result in the array having the same order as defined here
    MPI_Comm new_comm;
    MPI_Dist_graph_create_adjacent(
        base.get(), in_degree, sources_host->get_const_data(), MPI_UNWEIGHTED,
        out_degree, destinations_host->get_const_data(), MPI_UNWEIGHTED,
        MPI_INFO_NULL, false, &new_comm);
    mpi::communicator neighbor_comm{new_comm};  // need to make this owning

    return neighbor_comm;
}
template <typename IndexType>
mpi::communicator create_neighborhood_comm(
    mpi::communicator base, const overlapping_partition<IndexType>* part)
{
    return create_neighborhood_comm(base, part->get_recv_indices().target_ids_,
                                    part->get_send_indices().target_ids_);
}

template <typename ValueType>
struct interleaved_deleter {
    using vector_type = gko::matrix::Dense<ValueType>;

    void operator()(vector_type* ptr)
    {
        if (mode == transformation::set) {
            // normal scatter
        }
        if (mode == transformation::add) {
            // scatter with add
            auto host_exec = ptr->get_executor()->get_master();
            auto host_ptr = make_temporary_clone(host_exec, ptr);
            auto offset = 0;
            for (auto cur_idxs : idxs.sets) {
                auto full_idxs = cur_idxs.to_global_indices();
                full_idxs.set_executor(host_exec);
                for (int i = 0; i < full_idxs.get_num_elems(); ++i) {
                    auto row = full_idxs.get_const_data()[i];
                    for (int col = 0; col < ptr->get_size()[1]; ++col) {
                        original->at(row, col) += host_ptr->at(i + offset, col);
                    }
                }
                offset += cur_idxs.get_num_local_indices();
            }
        }
        delete ptr;
    }

    interleaved_deleter(std::unique_ptr<vector_type>&& original,
                        overlap_indices<int32>::interleaved idxs,
                        transformation mode)
        : original(std::move(original)), idxs(std::move(idxs)), mode(mode)
    {}

    interleaved_deleter(const interleaved_deleter& other)
        : original(make_dense_view(other.original)),
          idxs(other.idxs),
          mode(other.mode)
    {}

    std::unique_ptr<vector_type> original;
    overlap_indices<int32>::interleaved idxs;  // figure out something lighter
    transformation mode;
};

template <typename ValueType>
struct blocked_deleter {
    using vector_type = gko::matrix::Dense<ValueType>;

    void operator()(vector_type* ptr)
    {
        if (mode == transformation::set) {
            // do nothing
        }
        if (mode == transformation::add) {
            // need to put the 1.0 into outside storage for reuse
            original->add_scaled(
                gko::initialize<vector_type>({1.0}, original->get_executor()),
                ptr);
        }
        delete ptr;
    }

    blocked_deleter(std::unique_ptr<vector_type>&& original,
                    transformation mode)
        : original(std::move(original)), mode(mode)
    {}

    blocked_deleter(const blocked_deleter& other)
        : original(make_dense_view(other.original)), mode(other.mode)
    {}

    std::unique_ptr<vector_type> original;
    transformation mode;
};

/**
 * perhaps fix index type to int32?
 * since that is only local indices it might be enough
 */
struct sparse_communication {
    using partition_type = overlapping_partition<int32>;
    using overlap_idxs_type = overlap_indices<int32>;
    /**
     * throw if index set size is larger than int32
     * should comm be a neighborhood comm, or should we make it into one?
     */
    sparse_communication(
        mpi::communicator comm,
        std::shared_ptr<const overlapping_partition<int32>> part)
        : default_comm_(create_neighborhood_comm(
              comm, part->get_recv_indices().target_ids_,
              part->get_send_indices().target_ids_)),
          inverse_comm_(create_neighborhood_comm(
              comm, part->get_send_indices().target_ids_,
              part->get_recv_indices().target_ids_)),
          part_(std::move(part)),
          send_sizes_(comm.size()),
          send_offsets_(comm.size() + 1),
          recv_sizes_(comm.size()),
          recv_offsets_(comm.size() + 1)
    {
        auto exec = part_->get_executor();  // should be exec of part_
        auto host_exec = exec->get_master();
        auto fill_size_offsets = [&](std::vector<int>& sizes,
                                     std::vector<int>& offsets,
                                     const auto& overlap) {
            std::visit(
                overloaded{
                    [&](const typename overlap_idxs_type::blocked& idxs) {
                        for (int i = 0; i < idxs.intervals.size(); ++i) {
                            sizes[i] = idxs.intervals[i].length();
                        }
                        std::partial_sum(sizes.begin(), sizes.end(),
                                         offsets.begin() + 1);
                    },
                    [&](const typename overlap_idxs_type::interleaved& idxs) {
                        for (int i = 0; i < idxs.sets.size(); ++i) {
                            sizes[i] = idxs.sets[i].get_num_local_indices();
                        }
                        std::partial_sum(sizes.begin(), sizes.end(),
                                         offsets.begin() + 1);
                    }},
                overlap.idxs_);
        };
        fill_size_offsets(recv_sizes_, recv_offsets_,
                          part_->get_recv_indices());
        fill_size_offsets(send_sizes_, send_offsets_,
                          part_->get_send_indices());
    }

    static std::shared_ptr<sparse_communication> create(
        mpi::communicator comm,
        std::shared_ptr<const overlapping_partition<int32>> part)
    {
        return std::shared_ptr<sparse_communication>{
            new sparse_communication(std::move(comm), std::move(part))};
    }


    /**
     * thread safety: only one thread can execute this concurrently
     */
    template <typename ValueType>
    auto communicate(matrix::Dense<ValueType>* local_vector,
                     transformation mode) const
    {
        return communicate_impl_(default_comm_.get(), part_->get_send_indices(),
                                 send_sizes_, send_offsets_,
                                 part_->get_recv_indices(), recv_sizes_,
                                 recv_offsets_, local_vector, mode);
    }

    /**
     * inverts the roles of sender/receiver compared to communicate
     */
    template <typename ValueType>
    auto communicate_inverse(matrix::Dense<ValueType>* local_vector,
                             transformation mode) const
    {
        return communicate_impl_(inverse_comm_.get(), part_->get_recv_indices(),
                                 recv_sizes_, recv_offsets_,
                                 part_->get_send_indices(), send_sizes_,
                                 send_offsets_, local_vector, mode);
    }

    template <typename ValueType>
    auto communicate_impl_(MPI_Comm comm, const overlap_idxs_type& send_idxs,
                           const std::vector<comm_index_type>& send_sizes,
                           const std::vector<comm_index_type>& send_offsets,
                           const overlap_idxs_type& recv_idxs,
                           const std::vector<comm_index_type>& recv_sizes,
                           const std::vector<comm_index_type>& recv_offsets,
                           matrix::Dense<ValueType>* local_vector,
                           transformation mode) const
    {
        GKO_ASSERT(this->part_->get_size() == local_vector->get_size()[0]);

        using vector_type = matrix::Dense<ValueType>;

        auto exec = local_vector->get_executor();

        auto get_overlap_block = [&](const overlap_idxs_type& idxs) {
            return local_vector->create_submatrix(
                {static_cast<size_type>(
                     this->part_->get_local_indices().get_size()),
                 static_cast<size_type>(idxs.get_size())},
                {0, local_vector->get_size()[1]});
        };

        // automatically copies back/adds if necessary
        using recv_handle_t =
            std::unique_ptr<vector_type, std::function<void(vector_type*)>>;
        auto recv_handle = [&] {
            if (mode == transformation::set &&
                std::holds_alternative<typename overlap_idxs_type::blocked>(
                    recv_idxs.idxs_)) {
                return recv_handle_t{
                    get_overlap_block(recv_idxs).release(),
                    blocked_deleter{get_overlap_block(recv_idxs), mode}};
            }

            recv_buffer_.init<ValueType>(
                exec, {recv_idxs.get_num_elems(), local_vector->get_size()[1]});

            if (std::holds_alternative<typename overlap_idxs_type::blocked>(
                    recv_idxs.idxs_)) {
                return recv_handle_t{
                    make_dense_view(recv_buffer_.get<ValueType>()).release(),
                    blocked_deleter{get_overlap_block(recv_idxs), mode}};
            } else {
                return recv_handle_t{
                    make_dense_view(recv_buffer_.get<ValueType>()).release(),
                    interleaved_deleter{
                        make_dense_view(local_vector),
                        std::get<overlap_idxs_type::interleaved>(
                            recv_idxs.idxs_),
                        mode}};
            }
        }();
        auto send_handle = [&] {
            if (std::holds_alternative<typename overlap_idxs_type::blocked>(
                    send_idxs.idxs_)) {
                return get_overlap_block(send_idxs);
            } else {
                send_buffer_.init<ValueType>(
                    exec,
                    {send_idxs.get_num_elems(), local_vector->get_size()[1]});

                size_type offset = 0;
                auto idxs = std::get<typename overlap_idxs_type::interleaved>(
                    send_idxs.idxs_);
                for (int i = 0; i < idxs.sets.size(); ++i) {
                    // need direct support for index_set
                    auto full_idxs = idxs.sets[i].to_global_indices();
                    local_vector->row_gather(
                        &full_idxs,
                        send_buffer_.get<ValueType>()->create_submatrix(
                            {offset, offset + full_idxs.get_num_elems()},
                            {0, local_vector->get_size()[1]}));
                    offset += full_idxs.get_num_elems();
                }

                return make_dense_view(send_buffer_.get<ValueType>());
            }
        }();
        auto recv_ptr = recv_handle->get_values();
        auto send_ptr = send_handle->get_values();

        // request deletes recv_handle on successful wait (or at destructor)
        mpi::request req(
            [h = std::move(recv_handle)](MPI_Request) mutable { h.reset(); });
        MPI_Ineighbor_alltoallv(send_ptr, send_sizes.data(),
                                send_offsets.data(), MPI_DOUBLE, recv_ptr,
                                recv_sizes.data(), recv_offsets.data(),
                                MPI_DOUBLE, comm, req.get());
        return req;
    }

    mpi::communicator default_comm_;
    mpi::communicator inverse_comm_;

    std::shared_ptr<const overlapping_partition<int32>> part_;

    std::vector<comm_index_type> send_sizes_;
    std::vector<comm_index_type> send_offsets_;
    std::vector<comm_index_type> recv_sizes_;
    std::vector<comm_index_type> recv_offsets_;

    // need mutex for these, can only be released by mpi request
    gko::detail::DenseCache2 recv_buffer_;
    gko::detail::DenseCache2 send_buffer_;
};


enum class partition_of_unity {
    unique,  // alternative: restricted, shared indices are owned by exactly one
             // process
    shared,  // alternative: additive, shared indices are mutually owned by all
             // involved processes
};


template <typename ValueType>
struct overlapping_vector
    : public EnableLinOp<overlapping_vector<ValueType>>,
      public gko::EnableCreateMethod<overlapping_vector<ValueType>> {
    using value_type = ValueType;
    using local_vector_type = matrix::Dense<value_type>;

    size_type get_stride() const { return stride_; }

    size_type get_num_stored_elems() const { return buffer_.get_num_elems(); }

    auto make_consistent(transformation mode)
    {
        return sparse_comm_->communicate(as_dense().get(), mode);
    }

    template <typename F, typename = std::enable_if_t<
                              std::is_invocable_v<F, double, double>>>
    void make_consistent(F&& transformation);

    /**
     * could add non-const versions with custom deleter to write back changes
     */
    std::unique_ptr<const local_vector_type> extract_local()
    {
        return sparse_comm_->part_->extract_local(as_dense());
    }

    std::unique_ptr<const local_vector_type> extract_non_local()
    {
        return sparse_comm_->part_->extract_non_local(as_dense());
    }

    void apply_impl(const LinOp* b, LinOp* x) const override {}
    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {}

    overlapping_vector(std::shared_ptr<const Executor> exec,
                       std::shared_ptr<sparse_communication> sparse_comm = {},
                       std::unique_ptr<local_vector_type> local_vector = {})
        : EnableLinOp<overlapping_vector<ValueType>>(
              exec, {sparse_comm->part_->get_size(), 1}),
          sparse_comm_(std::move(sparse_comm)),
          buffer_(exec, make_array_view(local_vector->get_executor(),
                                        local_vector->get_num_stored_elements(),
                                        local_vector->get_values())),
          stride_(local_vector->get_stride())
    {}

    std::unique_ptr<const local_vector_type> as_dense() const
    {
        return local_vector_type::create_const(
            this->get_executor(), this->get_size(), buffer_.as_const_view(),
            this->get_stride());
    }

    std::unique_ptr<local_vector_type> as_dense()
    {
        return local_vector_type::create(this->get_executor(), this->get_size(),
                                         buffer_.as_view(), this->get_stride());
    }

    std::shared_ptr<sparse_communication> sparse_comm_;
    // contains local+nonlocal values
    // might switch to dense directly
    array<double> buffer_;
    size_type stride_;
};


}  // namespace gko::experimental::distributed

#endif
#endif  // OVERLAPPING_VECTOR_OVERLAPPING_VECTOR_HPP

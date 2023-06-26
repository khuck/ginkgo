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
#include <ginkgo/core/matrix/dense.hpp>


#include <variant>


namespace gko::experimental::distributed {
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
// explicit deduction guide (not needed as of C++20)
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

enum class transformation { set, add };

/**
 * actually this should be a partition of unity
 * @tparam IndexType
 */
template <typename IndexType>
struct overlapping_partition {
    using index_type = IndexType;
    using mask_type = uint8;

    struct overlap_indices {
        struct blocked {
            blocked(std::vector<span> intervals)
                : intervals(std::move(intervals)),
                  num_local_indices(std::accumulate(
                      this->intervals.begin(), this->intervals.end(), 0,
                      [](const auto& a, const auto& b) {
                          return a + b.length();
                      })),
                  size(this->intervals.empty()
                           ? 0
                           : std::max_element(this->intervals.begin(),
                                              this->intervals.end(),
                                              [](const auto& a, const auto& b) {
                                                  return a.end < b.end;
                                              })
                                 ->end)
            {}

            // conformity with index_set interface (necessary?)
            index_type get_size() const { return size; }
            index_type get_num_local_indices() const
            {
                return num_local_indices;
            }

            size_type get_num_subsets() const { return intervals.size(); }

            std::vector<span> intervals;  // a single span per target id
            size_type num_local_indices;
            index_type size;
        };

        // ids with same index subset
        struct interleaved {
            interleaved(std::vector<index_set<index_type>> sets)
                : sets(std::move(sets)),
                  num_local_indices(
                      std::accumulate(this->sets.begin(), this->sets.end(), 0,
                                      [](const auto& a, const auto& b) {
                                          return a + b.get_num_local_indices();
                                      })),
                  size(this->sets.empty()
                           ? 0
                           : std::max_element(
                                 this->sets.begin(), this->sets.end(),
                                 [](const auto& a, const auto& b) {
                                     return a.get_size() < b.get_size();
                                 })
                                 ->get_size())
            {}

            // conformity with index_set interface (necessary?)
            index_type get_size() const { return size; }
            index_type get_num_local_indices() const
            {
                return num_local_indices;
            }

            std::vector<index_set<index_type>>
                sets;  // a single index set per target id
            size_type num_local_indices;
            index_type size;
        };

        array<comm_index_type> target_ids;
        std::variant<blocked, interleaved> idxs;
    };

    const index_set<index_type>& get_local_indices() const
    {
        return local_idxs_;
    }

    const overlap_indices& get_send_indices() const
    {
        return overlap_send_idxs_;
    }

    const overlap_indices& get_recv_indices() const
    {
        return overlap_recv_idxs_;
    }

    size_type get_overlap_num_elems(const overlap_indices& idxs) const
    {
        return std::visit(
            [](const auto& idxs) {
                return static_cast<size_type>(idxs.get_num_local_indices());
            },
            idxs.idxs);
    }

    index_type get_overlap_size(const overlap_indices& idxs) const
    {
        return std::visit(
            overloaded{
                [this](const typename overlap_indices::blocked& block) {
                    return std::max(static_cast<index_type>(block.get_size()),
                                    local_idxs_.get_size());
                },
                [](const typename overlap_indices::interleaved& interleaved) {
                    return static_cast<index_type>(interleaved.get_size());
                }},
            idxs.idxs);
    }

    size_type get_size() const
    {
        return std::max(local_idxs_.get_size(),
                        std::max(get_overlap_size(overlap_send_idxs_),
                                 get_overlap_size(overlap_recv_idxs_)));
    }

    // only recv indices are relevant, since they are written to
    bool has_grouped_indices() const
    {
        return std::holds_alternative<typename overlap_indices::blocked>(
            overlap_recv_idxs_.idxs);
    }

    std::shared_ptr<const Executor> get_executor() const
    {
        return local_idxs_.get_executor();
    }

    template <typename ValueType>
    std::unique_ptr<const gko::matrix::Dense<ValueType>> extract_local(
        std::unique_ptr<gko::matrix::Dense<ValueType>> vector) const
    {
        GKO_ASSERT(vector->get_size()[0] == get_size());
        if (has_grouped_indices()) {
            return vector->create_submatrix(span{0, local_idxs_.get_size()},
                                            span{0, vector->get_size()[1]});
        } else {
            // not yet done
            return nullptr;
        }
    }

    template <typename ValueType>
    std::unique_ptr<const gko::matrix::Dense<ValueType>> extract_non_local(
        std::unique_ptr<gko::matrix::Dense<ValueType>> vector) const
    {
        GKO_ASSERT(vector->get_size()[0] == get_size());
        if (has_grouped_indices()) {
            return vector->create_submatrix(
                span{local_idxs_.get_size(),
                     get_overlap_size(overlap_recv_idxs_)},
                span{0, vector->get_size()[1]});
        } else {
            // not yet done
            return nullptr;
        }
    }

    /*
     * Indices are grouped first by local indices and then receiving overlapping
     * indices
     * the recv indices of different target-ids can't overlap, need other
     * constructor for that
     */
    static std::shared_ptr<overlapping_partition> build_from_grouped_recv1(
        std::shared_ptr<const Executor> exec, size_type local_size,
        std::vector<std::pair<index_set<index_type>, comm_index_type>>
            send_idxs,
        array<comm_index_type> target_id, array<size_type> group_size)
    {
        std::vector<index_set<index_type>> send_index_sets(
            send_idxs.size(), index_set<index_type>(exec));
        array<comm_index_type> send_target_ids(exec->get_master(),
                                               send_idxs.size());

        for (int i = 0; i < send_idxs.size(); ++i) {
            send_index_sets[i] = std::move(send_idxs[i].first);
            send_target_ids.get_data()[i] = send_idxs[i].second;
        }
        send_target_ids.set_executor(exec);

        return build_from_grouped_recv2(
            std::move(exec), local_size,
            std::make_pair(std::move(send_index_sets),
                           std::move(send_target_ids)),
            std::move(target_id), std::move(group_size));
    }

    /**
     * does the order of target id imply the received order of the blocked
     * indices?
     */
    static std::shared_ptr<overlapping_partition> build_from_grouped_recv2(
        std::shared_ptr<const Executor> exec, size_type local_size,
        std::pair<std::vector<index_set<index_type>>, array<comm_index_type>>
            send_idxs,
        array<comm_index_type> target_id, array<size_type> group_size)
    {
        // make sure shared indices are a subset of local indices
        GKO_ASSERT(send_idxs.first.size() == 0 ||
                   local_size >=
                       std::max_element(send_idxs.first.begin(),
                                        send_idxs.first.end(),
                                        [](const auto& a, const auto& b) {
                                            return a.get_size() < b.get_size();
                                        })
                           ->get_size());
        index_set<index_type> local_idxs(exec, gko::span{0, local_size});

        // need to compute partial sum
        auto recv_size = reduce_add(group_size);
        // need to create a subset for each target id
        // brute force creating index sets until better constructor is available
        std::vector<span> intervals;
        auto offset = local_size;
        for (int gid = 0; gid < group_size.get_num_elems(); ++gid) {
            auto current_size = group_size.get_const_data()[gid];
            intervals.emplace_back(offset, offset + current_size);
            offset += current_size;
        }

        return std::shared_ptr<overlapping_partition>{new overlapping_partition{
            std::move(local_idxs),
            {std::move(send_idxs.second), std::move(send_idxs.first)},
            {std::move(target_id), std::move(intervals)}}};
    }

private:
    overlapping_partition(index_set<index_type> local_idxs,
                          overlap_indices overlap_send_idxs,
                          overlap_indices overlap_recv_idxs)
        : local_idxs_(std::move(local_idxs)),
          overlap_send_idxs_(std::move(overlap_send_idxs)),
          overlap_recv_idxs_(std::move(overlap_recv_idxs))
    {}

    // owned by this process (exclusively or shared)
    index_set<index_type> local_idxs_;
    // owned by this and used by other processes (subset of local_idxs_)
    overlap_indices overlap_send_idxs_;
    // owned by other processes (doesn't exclude this also owning them)
    overlap_indices overlap_recv_idxs_;

    // store local multiplicity, i.e. if the index is owned by this process,
    // by how many is it owned in total, otherwise the multiplicity is zero
};

/**
 * maybe allow for processes owning multiple parts by mapping target_ids to
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
    return create_neighborhood_comm(base, part->get_recv_indices().target_ids,
                                    part->get_send_indices().target_ids);
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

    interleaved_deleter(
        std::unique_ptr<vector_type>&& original,
        overlapping_partition<int32>::overlap_indices::interleaved idxs,
        transformation mode)
        : original(std::move(original)), idxs(std::move(idxs)), mode(mode)
    {}

    interleaved_deleter(const interleaved_deleter& other)
        : original(make_dense_view(other.original)),
          idxs(other.idxs),
          mode(other.mode)
    {}

    std::unique_ptr<vector_type> original;
    overlapping_partition<int32>::overlap_indices::interleaved
        idxs;  // figure out something lighter
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
    /**
     * throw if index set size is larger than int32
     * should comm be a neighborhood comm, or should we make it into one?
     */
    sparse_communication(
        mpi::communicator comm,
        std::shared_ptr<const overlapping_partition<int32>> part)
        : default_comm_(create_neighborhood_comm(
              comm, part->get_recv_indices().target_ids,
              part->get_send_indices().target_ids)),
          inverse_comm_(create_neighborhood_comm(
              comm, part->get_send_indices().target_ids,
              part->get_recv_indices().target_ids)),
          part_(std::move(part)),
          send_sizes_(comm.size()),
          send_offsets_(comm.size() + 1),
          recv_sizes_(comm.size()),
          recv_offsets_(comm.size() + 1)
    {
        using partition_type = overlapping_partition<int32>;
        auto exec = part_->get_executor();  // should be exec of part_
        auto host_exec = exec->get_master();
        auto fill_size_offsets = [&](std::vector<int>& sizes,
                                     std::vector<int>& offsets,
                                     const auto& overlap) {
            std::visit(
                overloaded{
                    [&](const typename partition_type::overlap_indices::blocked&
                            idxs) {
                        for (int i = 0; i < idxs.intervals.size(); ++i) {
                            sizes[i] = idxs.intervals[i].length();
                        }
                        std::partial_sum(sizes.begin(), sizes.end(),
                                         offsets.begin() + 1);
                    },
                    [&](const typename partition_type::overlap_indices::
                            interleaved& idxs) {
                        for (int i = 0; i < idxs.sets.size(); ++i) {
                            sizes[i] = idxs.sets[i].get_num_local_indices();
                        }
                        std::partial_sum(sizes.begin(), sizes.end(),
                                         offsets.begin() + 1);
                    }},
                overlap.idxs);
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

    using partition_type = overlapping_partition<int32>;

    /**
     * thread safety: only one thread can execute this concurrently
     */
    template <typename ValueType>
    auto communicate(matrix::Dense<ValueType>* local_vector,
                     transformation mode) const
    {
        GKO_ASSERT(this->part_->get_size() == local_vector->get_size()[0]);

        using vector_type = matrix::Dense<ValueType>;
        auto recv_idxs = part_->get_recv_indices();
        auto send_idxs = part_->get_send_indices();

        auto exec = local_vector->get_executor();

        auto get_overlap_block =
            [&](const typename partition_type::overlap_indices& idxs) {
                return local_vector->create_submatrix(
                    {static_cast<size_type>(
                         this->part_->get_local_indices().get_size()),
                     static_cast<size_type>(part_->get_overlap_size(idxs))},
                    {0, local_vector->get_size()[1]});
            };

        // automatically copies back/adds if necessary
        using recv_handle_t =
            std::unique_ptr<vector_type, std::function<void(vector_type*)>>;
        auto recv_handle = [&] {
            if (mode == transformation::set &&
                std::holds_alternative<
                    typename partition_type::overlap_indices::blocked>(
                    recv_idxs.idxs)) {
                return recv_handle_t{
                    get_overlap_block(recv_idxs).release(),
                    blocked_deleter{get_overlap_block(recv_idxs), mode}};
            }

            recv_buffer_.init<ValueType>(
                exec, {this->part_->get_overlap_num_elems(recv_idxs),
                       local_vector->get_size()[1]});

            if (std::holds_alternative<
                    typename partition_type::overlap_indices::blocked>(
                    recv_idxs.idxs)) {
                return recv_handle_t{
                    make_dense_view(recv_buffer_.get<ValueType>()).release(),
                    blocked_deleter{get_overlap_block(recv_idxs), mode}};
            } else {
                return recv_handle_t{
                    make_dense_view(recv_buffer_.get<ValueType>()).release(),
                    interleaved_deleter{
                        make_dense_view(local_vector),
                        std::get<partition_type ::overlap_indices::interleaved>(
                            recv_idxs.idxs),
                        mode}};
            }
        }();
        auto send_handle = [&] {
            if (std::holds_alternative<
                    typename partition_type::overlap_indices::blocked>(
                    send_idxs.idxs)) {
                return get_overlap_block(send_idxs);
            } else {
                send_buffer_.init<ValueType>(
                    exec, {this->part_->get_overlap_num_elems(send_idxs),
                           local_vector->get_size()[1]});

                size_type offset = 0;
                auto idxs = std::get<
                    typename partition_type::overlap_indices::interleaved>(
                    send_idxs.idxs);
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

        // request deletes recv_handle on successful wait
        mpi::request req(
            [h = std::move(recv_handle)](MPI_Request) mutable { h.reset(); });
        MPI_Ineighbor_alltoallv(send_ptr, send_sizes_.data(),
                                send_offsets_.data(), MPI_DOUBLE, recv_ptr,
                                recv_sizes_.data(), recv_offsets_.data(),
                                MPI_DOUBLE, default_comm_.get(), req.get());
        return req;
    }

    /**
     * inverts the roles of sender/receiver compared to communicate
     */
    template <typename ValueType>
    auto communicate_inverse(matrix::Dense<ValueType>* local_vector,
                             transformation mode) const
    {
        GKO_ASSERT(this->part_->get_size() == local_vector->get_size()[0]);

        using vector_type = matrix::Dense<ValueType>;
        auto recv_idxs = part_->get_send_indices();
        auto send_idxs = part_->get_recv_indices();

        auto exec = local_vector->get_executor();

        auto get_overlap_block =
            [&](const typename partition_type::overlap_indices& idxs) {
                return local_vector->create_submatrix(
                    {static_cast<size_type>(
                         this->part_->get_local_indices().get_size()),
                     static_cast<size_type>(part_->get_overlap_size(idxs))},
                    {0, local_vector->get_size()[1]});
            };

        // automatically copies back/adds if necessary
        using recv_handle_t =
            std::unique_ptr<vector_type, std::function<void(vector_type*)>>;
        auto recv_handle = [&] {
            if (mode == transformation::set &&
                std::holds_alternative<
                    typename partition_type::overlap_indices::blocked>(
                    recv_idxs.idxs)) {
                return recv_handle_t{
                    get_overlap_block(recv_idxs).release(),
                    blocked_deleter{get_overlap_block(recv_idxs), mode}};
            }

            recv_buffer_.init<ValueType>(
                exec, {this->part_->get_overlap_num_elems(recv_idxs),
                       local_vector->get_size()[1]});

            if (std::holds_alternative<
                    typename partition_type::overlap_indices::blocked>(
                    recv_idxs.idxs)) {
                return recv_handle_t{
                    make_dense_view(recv_buffer_.get<ValueType>()).release(),
                    blocked_deleter{get_overlap_block(recv_idxs), mode}};
            } else {
                return recv_handle_t{
                    make_dense_view(recv_buffer_.get<ValueType>()).release(),
                    interleaved_deleter{
                        make_dense_view(local_vector),
                        std::get<partition_type ::overlap_indices::interleaved>(
                            recv_idxs.idxs),
                        mode}};
            }
        }();
        auto send_handle = [&] {
            if (std::holds_alternative<
                    typename partition_type::overlap_indices::blocked>(
                    send_idxs.idxs)) {
                return get_overlap_block(send_idxs);
            } else {
                send_buffer_.init<ValueType>(
                    exec, {this->part_->get_overlap_num_elems(send_idxs),
                           local_vector->get_size()[1]});

                size_type offset = 0;
                auto idxs = std::get<
                    typename partition_type::overlap_indices::interleaved>(
                    send_idxs.idxs);
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

        // request deletes recv_handle on successful wait
        mpi::request req(
            [h = std::move(recv_handle)](MPI_Request) mutable { h.reset(); });
        MPI_Ineighbor_alltoallv(send_ptr, recv_sizes_.data(),
                                recv_offsets_.data(), MPI_DOUBLE, recv_ptr,
                                send_sizes_.data(), send_offsets_.data(),
                                MPI_DOUBLE, inverse_comm_.get(), req.get());
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

/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <ginkgo/core/distributed/repartition.hpp>
#include <numeric>


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/coo.hpp>


namespace gko {
namespace distributed {


struct all_to_all_pattern {
    using comm_vector = std::vector<comm_index_type>;
    comm_vector send_sizes;
    comm_vector send_offsets;
    comm_vector recv_sizes;
    comm_vector recv_offsets;
};


template <typename LocalIndexType, typename GlobalIndexType>
all_to_all_pattern build_communication_pattern(
    mpi::communicator from_comm,
    std::shared_ptr<Partition<LocalIndexType, GlobalIndexType>> from_part,
    std::shared_ptr<Partition<LocalIndexType, GlobalIndexType>> to_part,
    const array<GlobalIndexType>& indices)
{
    const auto from_n_parts = from_part->get_num_parts();

    auto host_to_part =
        gko::clone(to_part->get_executor()->get_master(), to_part.get());

    const auto* to_ranges = host_to_part->get_range_bounds();
    const auto* to_pid = host_to_part->get_part_ids();
    const auto to_n_ranges = host_to_part->get_num_ranges();

    using comm_vector = std::vector<comm_index_type>;
    comm_vector send_sizes(from_n_parts);
    comm_vector send_offsets(from_n_parts + 1);
    comm_vector recv_sizes(from_n_parts);
    comm_vector recv_offsets(from_n_parts + 1);

    auto find_to_range = [&](GlobalIndexType idx) {
        auto it =
            std::upper_bound(to_ranges + 1, to_ranges + to_n_ranges + 1, idx);
        return std::distance(to_ranges + 1, it);
    };

    for (size_type i = 0; i < indices.get_num_elems(); ++i) {
        auto to_range = find_to_range(indices.get_const_data()[i]);
        auto recv_pid = to_pid[to_range];
        send_sizes[recv_pid]++;
    }
    std::partial_sum(send_sizes.cbegin(), send_sizes.cend(),
                     send_offsets.begin() + 1);

    from_comm.all_to_all(send_sizes.data(), 1, recv_sizes.data(), 1);
    std::partial_sum(recv_sizes.cbegin(), recv_sizes.cend(),
                     recv_offsets.begin() + 1);

    return all_to_all_pattern{send_sizes, send_offsets, recv_sizes,
                              recv_offsets};
}


template <typename LocalIndexType, typename GlobalIndexType>
repartitioner<LocalIndexType, GlobalIndexType>::repartitioner(
    mpi::communicator from_comm,
    std::shared_ptr<Partition<LocalIndexType, GlobalIndexType>> from_partition,
    std::shared_ptr<Partition<LocalIndexType, GlobalIndexType>> to_partition)
    : from_partition_(std::move(from_partition)),
      to_partition_(std::move(to_partition)),
      from_comm_(from_comm),
      to_comm_(from_comm),
      to_has_data_(false)
{
    auto host_from_partition = gko::clone(
        from_partition_->get_executor()->get_master(), from_partition_.get());
    const auto* old_ranges = host_from_partition->get_range_bounds();
    const auto* old_pid = host_from_partition->get_part_ids();
    const auto old_n_ranges = host_from_partition->get_num_ranges();
    const auto old_n_parts = host_from_partition->get_num_parts();

    const auto new_n_parts = to_partition_->get_num_parts();

    GKO_ASSERT(new_n_parts <= old_n_parts);
    GKO_ASSERT(from_partition_->get_size() == to_partition_->get_size());

    const auto rank = from_comm_.rank();
    to_has_data_ = rank < new_n_parts;

    if (new_n_parts < old_n_parts) {
        to_comm_ =
            mpi::communicator(from_comm_.get(), to_has_data_, from_comm_.rank(),
                              to_partition_->get_executor());
    } else {
        to_comm_ = from_comm_;
    }

    array<GlobalIndexType> owned_global_idxs(
        host_from_partition->get_executor(),
        host_from_partition->get_part_size(rank));
    {
        size_type local_idx = 0;
        for (size_type range_idx = 0; range_idx < old_n_ranges; ++range_idx) {
            if (old_pid[range_idx] == rank) {
                for (GlobalIndexType global_idx = old_ranges[range_idx];
                     global_idx < old_ranges[range_idx + 1]; ++global_idx) {
                    owned_global_idxs.get_data()[local_idx++] = global_idx;
                }
            }
        }
    }
    auto pattern = build_communication_pattern(
        from_comm_, from_partition_, to_partition_, owned_global_idxs);

    default_send_sizes_ =
        std::make_shared<std::vector<comm_index_type>>(pattern.send_sizes);
    default_send_offsets_ =
        std::make_shared<std::vector<comm_index_type>>(pattern.send_offsets);
    default_recv_sizes_ =
        std::make_shared<std::vector<comm_index_type>>(pattern.recv_sizes);
    default_recv_offsets_ =
        std::make_shared<std::vector<comm_index_type>>(pattern.recv_offsets);
}


template <typename LocalIndexType, typename GlobalIndexType>
template <typename ValueType>
void repartitioner<LocalIndexType, GlobalIndexType>::gather(
    const Vector<ValueType>* from, Vector<ValueType>* to) const
{
    if (from->get_communicator() != from_comm_) {
        throw GKO_MPI_ERROR(MPI_ERR_COMM);
    }
    // Todo: figure out if necessary to test parts for equality

    if (!from_partition_->has_ordered_parts()) {
        GKO_NOT_IMPLEMENTED;
    }
    std::shared_ptr<std::vector<comm_index_type>> send_sizes;
    std::shared_ptr<std::vector<comm_index_type>> send_offsets;
    std::shared_ptr<std::vector<comm_index_type>> recv_sizes;
    std::shared_ptr<std::vector<comm_index_type>> recv_offsets;

    if (from->get_size()[1] == 1) {
        send_sizes = default_send_sizes_;
        send_offsets = default_send_offsets_;
        recv_sizes = default_recv_sizes_;
        recv_offsets = default_recv_offsets_;
    } else {
        GKO_NOT_IMPLEMENTED;
    }

    auto tmp = Vector<ValueType>::create(
        to_comm_, dim<2>{to_partition_->get_size(), 1},
        dim<2>{static_cast<size_type>(recv_offsets->back()), 1});

    const auto* send_buffer = from->get_local_vector()->get_const_values();
    auto* recv_buffer = tmp->get_local_values();

    if (to_partition_->get_num_parts() > 1) {
        from_comm_.all_to_all_v(send_buffer, send_sizes->data(),
                                send_offsets->data(), recv_buffer,
                                recv_sizes->data(), recv_offsets->data());
    } else {
        const comm_index_type root = 0;
        from_comm_.gather_v(send_buffer, (*send_sizes)[root], recv_buffer,
                            recv_sizes->data(), recv_offsets->data(), root);
    }

    if (to_has_data()) {
        tmp->move_to(to);
    } else {
        *to = *Vector<ValueType>::create(to_comm_);
    }
}

#define GKO_DECLARE_REPETITIONER_GATHER(_value_type, _index_type_l,        \
                                        _index_type_g)                     \
    void repartitioner<_index_type_l, _index_type_g>::gather<_value_type>( \
        const Vector<_value_type>* from, Vector<_value_type>* to) const
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_REPETITIONER_GATHER);


template <typename LocalIndexType, typename GlobalIndexType>
template <typename ValueType>
void repartitioner<LocalIndexType, GlobalIndexType>::scatter(
    const Vector<ValueType>* to, Vector<ValueType>* from) const
{
    if (to->get_communicator() != to_comm_) {
        throw GKO_MPI_ERROR(MPI_ERR_COMM);
    }

    if (!to_partition_->has_ordered_parts()) {
        GKO_NOT_IMPLEMENTED;
    }

    std::shared_ptr<std::vector<comm_index_type>> send_sizes;
    std::shared_ptr<std::vector<comm_index_type>> send_offsets;
    std::shared_ptr<std::vector<comm_index_type>> recv_sizes;
    std::shared_ptr<std::vector<comm_index_type>> recv_offsets;

    if (to->get_size()[1] == 1 || !to_has_data()) {
        send_sizes = default_recv_sizes_;
        send_offsets = default_recv_offsets_;
        recv_sizes = default_send_sizes_;
        recv_offsets = default_send_offsets_;
    } else {
        GKO_NOT_IMPLEMENTED;
    }

    auto tmp = Vector<ValueType>::create(
        from_comm_, dim<2>{from_partition_->get_size(), 1},
        dim<2>{static_cast<size_type>(recv_offsets->back()), 1});

    const auto* send_buffer = to->get_local_vector()->get_const_values();
    auto* recv_buffer = tmp->get_local_values();

    if (to_partition_->get_num_parts() > 1) {
        from_comm_.all_to_all_v(send_buffer, send_sizes->data(),
                                send_offsets->data(), recv_buffer,
                                recv_sizes->data(), recv_offsets->data());
    } else {
        const comm_index_type root = 0;
        from_comm_.scatter_v(send_buffer, send_sizes->data(),
                             send_offsets->data(), recv_buffer,
                             (*recv_sizes)[root], root);
    }

    tmp->move_to(from);
}

#define GKO_DECLARE_REPETITIONER_SCATTER(_value_type, _index_type_l,        \
                                         _index_type_g)                     \
    void repartitioner<_index_type_l, _index_type_g>::scatter<_value_type>( \
        const Vector<_value_type>* to, Vector<_value_type>* from) const
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_REPETITIONER_SCATTER);


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
auto write_local(std::shared_ptr<const Executor> exec,
                 const Matrix<ValueType, LocalIndexType, GlobalIndexType>* mat,
                 const Partition<LocalIndexType, GlobalIndexType>* part)
{
    using md_local = device_matrix_data<ValueType, LocalIndexType>;
    using md_global = device_matrix_data<ValueType, GlobalIndexType>;

    md_local local_md(exec->get_master());
    md_local non_local_md(exec->get_master());
    auto coo_local =
        matrix::Coo<ValueType, LocalIndexType>::create(mat->get_executor());
    auto coo_non_local =
        matrix::Coo<ValueType, LocalIndexType>::create(mat->get_executor());
    as<ConvertibleTo<matrix::Coo<ValueType, LocalIndexType>>>(
        mat->get_local_matrix())
        ->convert_to(coo_local.get());
    as<ConvertibleTo<matrix::Coo<ValueType, LocalIndexType>>>(
        mat->get_non_local_matrix())
        ->convert_to(coo_non_local.get());
    coo_local->write(local_md);
    coo_non_local->write(non_local_md);

    auto local_nnz = local_md.get_num_elems();
    auto non_local_nnz = non_local_md.get_num_elems();
    md_global data{exec->get_master(),
                   dim<2>{local_md.get_size()[0],
                          local_md.get_size()[1] + non_local_md.get_size()[1]},
                   local_nnz + non_local_nnz};

    auto host_part =
        gko::make_temporary_clone(part->get_executor()->get_master(), part);
    array<GlobalIndexType> map_local_to_global(
        exec->get_master(),
        part->get_part_size(mat->get_communicator().rank()));
    int local_idx = 0;
    for (int rid = 0; rid < host_part->get_num_ranges(); ++rid) {
        if (host_part->get_part_ids()[rid] == mat->get_communicator().rank()) {
            for (int i = host_part->get_range_bounds()[rid];
                 i < host_part->get_range_bounds()[rid + 1]; ++i) {
                map_local_to_global.get_data()[local_idx++] = i;
            }
        }
    }

    for (int i = 0; i < local_nnz; ++i) {
        data.get_row_idxs()[i] =
            map_local_to_global.get_data()[local_md.get_row_idxs()[i]];
        data.get_col_idxs()[i] =
            map_local_to_global.get_data()[local_md.get_col_idxs()[i]];
        data.get_values()[i] = local_md.get_values()[i];
    }
    for (int i = 0; i < non_local_nnz; ++i) {
        data.get_row_idxs()[local_nnz + i] =
            map_local_to_global.get_data()[non_local_md.get_row_idxs()[i]];
        data.get_col_idxs()[local_nnz + i] =
            mat->get_non_local_to_global()[non_local_md.get_col_idxs()[i]];
        data.get_values()[local_nnz + i] = non_local_md.get_values()[i];
    }

    data = md_global{exec, data};
    data.sort_row_major();
    return data;
}


template <typename LocalIndexType, typename GlobalIndexType>
template <typename ValueType>
void repartitioner<LocalIndexType, GlobalIndexType>::gather(
    const Matrix<ValueType, LocalIndexType, GlobalIndexType>* from,
    Matrix<ValueType, LocalIndexType, GlobalIndexType>* to) const
{
    if (from->get_communicator() != from_comm_ ||
        to->get_communicator() != to_comm_) {
        throw GKO_MPI_ERROR(MPI_ERR_COMM);
    }

    using md_global = device_matrix_data<ValueType, GlobalIndexType>;
    md_global local_data =
        write_local(from->get_executor(), from, from_partition_.get());
    auto local_arrays = std::move(local_data).empty_out();

    auto pattern = build_communication_pattern(
        from_comm_, from_partition_, to_partition_, local_arrays.row_idxs);

    const auto new_local_nnz = pattern.recv_offsets.back();
    array<GlobalIndexType> recv_rows(to->get_executor(), new_local_nnz);
    array<GlobalIndexType> recv_cols(to->get_executor(), new_local_nnz);
    array<ValueType> recv_values(to->get_executor(), new_local_nnz);

    auto communicate = [&](const auto* send_buffer, auto* recv_buffer) {
        if (to_partition_->get_num_parts() > 1) {
            from_comm_.all_to_all_v(send_buffer, pattern.send_sizes.data(),
                                    pattern.send_offsets.data(), recv_buffer,
                                    pattern.recv_sizes.data(),
                                    pattern.recv_offsets.data());
        } else {
            const comm_index_type root = 0;
            from_comm_.gather_v(send_buffer, pattern.send_sizes[root],
                                recv_buffer, pattern.recv_sizes.data(),
                                pattern.recv_offsets.data(), root);
        }
    };
    communicate(local_arrays.row_idxs.get_const_data(), recv_rows.get_data());
    communicate(local_arrays.col_idxs.get_const_data(), recv_cols.get_data());
    communicate(local_arrays.values.get_const_data(), recv_values.get_data());

    md_global new_local_data(to->get_executor(), from->get_size(),
                             std::move(recv_rows), std::move(recv_cols),
                             std::move(recv_values));
    new_local_data.sort_row_major();

    auto tmp =
        Matrix<ValueType, LocalIndexType, GlobalIndexType>::create(to_comm_);
    if (to_has_data_) {
        tmp->read_distributed(new_local_data, to_partition_.get());
    }
    tmp->move_to(to);
}

#define GKO_DECLARE_REPETITIONER_GATHER_MATRIX(_value_type, _index_type_l, \
                                               _index_type_g)              \
    void repartitioner<_index_type_l, _index_type_g>::gather<_value_type>( \
        const Matrix<_value_type, _index_type_l, _index_type_g>* to,       \
        Matrix<_value_type, _index_type_l, _index_type_g>* from) const
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_REPETITIONER_GATHER_MATRIX);

#define GKO_DECLARE_REPARTITIONER(_type_l, _type_g) \
    class repartitioner<_type_l, _type_g>
GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(GKO_DECLARE_REPARTITIONER);


}  // namespace distributed
}  // namespace gko


#endif

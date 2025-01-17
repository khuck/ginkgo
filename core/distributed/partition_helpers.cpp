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

#include <ginkgo/core/distributed/partition_helpers.hpp>


#include <numeric>


#include <ginkgo/core/distributed/partition.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "core/distributed/partition_helpers_kernels.hpp"


namespace gko {
namespace experimental {
namespace distributed {
namespace components {
namespace {


GKO_REGISTER_OPERATION(fill_seq_array, components::fill_seq_array);


}  // namespace
}  // namespace components


namespace partition_helpers {
namespace {


GKO_REGISTER_OPERATION(sort_by_range_start,
                       partition_helpers::sort_by_range_start);
GKO_REGISTER_OPERATION(check_consecutive_ranges,
                       partition_helpers::check_consecutive_ranges);
GKO_REGISTER_OPERATION(compress_ranges, partition_helpers::compress_ranges);


}  // namespace
}  // namespace partition_helpers


template <typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<Partition<LocalIndexType, GlobalIndexType>>
build_partition_from_local_range(std::shared_ptr<const Executor> exec,
                                 mpi::communicator comm, span local_range)
{
    std::array<GlobalIndexType, 2> range{
        static_cast<GlobalIndexType>(local_range.begin),
        static_cast<GlobalIndexType>(local_range.end)};

    // make all range_start_ends available on each rank
    // note: not all combination of MPI + GPU library seem to support
    // mixing host and device buffers, e.g. OpenMPI 4.0.5 and Rocm 4.0
    auto mpi_exec = exec->get_master();
    array<GlobalIndexType> ranges_start_end(mpi_exec, comm.size() * 2);
    ranges_start_end.fill(invalid_index<GlobalIndexType>());
    comm.all_gather(mpi_exec, range.data(), 2, ranges_start_end.get_data(), 2);
    ranges_start_end.set_executor(exec);

    // make_sort_by_range_start
    array<comm_index_type> part_ids(exec, comm.size());
    exec->run(components::make_fill_seq_array(part_ids.get_data(),
                                              part_ids.get_num_elems()));
    exec->run(partition_helpers::make_sort_by_range_start(ranges_start_end,
                                                          part_ids));

    // check for consistency
    bool consecutive_ranges = false;
    exec->run(partition_helpers::make_check_consecutive_ranges(
        ranges_start_end, consecutive_ranges));
    if (!consecutive_ranges) {
        GKO_INVALID_STATE("The partition contains gaps.");
    }

    // join (now consecutive) starts and ends into combined array
    array<GlobalIndexType> ranges(exec, comm.size() + 1);
    exec->run(
        partition_helpers::make_compress_ranges(ranges_start_end, ranges));

    return Partition<LocalIndexType, GlobalIndexType>::build_from_contiguous(
        exec, ranges, part_ids);
}

#define GKO_DECLARE_BUILD_PARTITION_FROM_LOCAL_RANGE(_local_type,          \
                                                     _global_type)         \
    std::unique_ptr<Partition<_local_type, _global_type>>                  \
    build_partition_from_local_range(std::shared_ptr<const Executor> exec, \
                                     mpi::communicator comm, span local_range)
GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_BUILD_PARTITION_FROM_LOCAL_RANGE);


template <typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<Partition<LocalIndexType, GlobalIndexType>>
build_partition_from_local_size(std::shared_ptr<const Executor> exec,
                                mpi::communicator comm, size_type local_size)
{
    auto local_size_gi = static_cast<GlobalIndexType>(local_size);
    array<GlobalIndexType> sizes(exec->get_master(), comm.size());
    comm.all_gather(exec, &local_size_gi, 1, sizes.get_data(), 1);

    array<GlobalIndexType> offsets(exec->get_master(), comm.size() + 1);
    offsets.get_data()[0] = 0;
    std::partial_sum(sizes.get_data(), sizes.get_data() + comm.size(),
                     offsets.get_data() + 1);

    return Partition<LocalIndexType, GlobalIndexType>::build_from_contiguous(
        exec, offsets);
}

#define GKO_DECLARE_BUILD_PARTITION_FROM_LOCAL_SIZE(_local_type, _global_type) \
    std::unique_ptr<Partition<_local_type, _global_type>>                      \
    build_partition_from_local_size(std::shared_ptr<const Executor> exec,      \
                                    mpi::communicator comm,                    \
                                    size_type local_range)
GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_BUILD_PARTITION_FROM_LOCAL_SIZE);


}  // namespace distributed
}  // namespace experimental
}  // namespace gko

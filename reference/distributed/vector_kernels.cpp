/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include "core/distributed/vector_kernels.hpp"


#include "core/base/allocator.hpp"
#include "core/components/prefix_sum_kernels.hpp"

namespace gko {
namespace kernels {
namespace reference {
namespace distributed_vector {


template <typename ValueType, typename LocalIndexType>
void build_local(
    std::shared_ptr<const DefaultExecutor> exec,
    const Array<matrix_data_entry<ValueType, global_index_type>>& input,
    const distributed::Partition<LocalIndexType>* partition,
    comm_index_type local_part,
    Array<matrix_data_entry<ValueType, LocalIndexType>>& local_data,
    Array<global_index_type>& local_to_global, ValueType deduction_help)
{
    using range_index_type = global_index_type;
    using part_index_type = comm_index_type;
    using global_nonzero = matrix_data_entry<ValueType, global_index_type>;
    using local_nonzero = matrix_data_entry<ValueType, LocalIndexType>;
    auto input_data = input.get_const_data();
    auto range_bounds = partition->get_const_range_bounds();
    auto range_parts = partition->get_const_part_ids();
    auto range_ranks = partition->get_range_ranks();
    auto num_parts = partition->get_num_parts();
    auto num_ranges = partition->get_num_ranges();

    // helpers for retrieving range info
    struct range_info {
        range_index_type index{};
        global_index_type begin{};
        global_index_type end{};
        LocalIndexType base_rank{};
        part_index_type part{};
    };
    auto find_range = [&](global_index_type idx) {
        auto it = std::upper_bound(range_bounds + 1,
                                   range_bounds + num_ranges + 1, idx);
        return std::distance(range_bounds + 1, it);
    };
    auto update_range = [&](global_index_type idx, range_info& info) {
        if (idx < info.begin || idx >= info.end) {
            info.index = find_range(idx);
            info.begin = range_bounds[info.index];
            info.end = range_bounds[info.index + 1];
            info.base_rank = range_ranks[info.index];
            info.part = range_parts[info.index];
            // assert(info.index < num_ranges);
        }
        // assert(idx >= info.begin && idx < info.end);
    };
    auto map_to_local = [&](global_index_type idx,
                            range_info info) -> LocalIndexType {
        return static_cast<LocalIndexType>(idx - info.begin) + info.base_rank;
    };

    range_info row_range{};
    size_type count = std::count_if(
        input_data, input_data + input.get_num_elems(), [&](const auto& entry) {
            auto range_idx = find_range(entry.row);
            return range_parts[range_idx] == local_part;
        });

    local_to_global.resize_and_reset(partition->get_part_size(local_part));
    std::fill_n(local_to_global.get_data(), local_to_global.get_num_elems(),
                -1);

    local_data.resize_and_reset(count);
    size_type idx{};
    for (size_type i = 0; i < input.get_num_elems(); ++i) {
        auto entry = input_data[i];
        update_range(entry.row, row_range);
        // skip non-local rows
        if (row_range.part != local_part) {
            continue;
        }
        local_data.get_data()[idx] = {// map global row idx to local row idx
                                      map_to_local(entry.row, row_range),
                                      static_cast<LocalIndexType>(entry.column),
                                      entry.value};
        local_to_global.get_data()[map_to_local(entry.row, row_range)] =
            entry.row;
        idx++;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BUILD_LOCAL);


}  // namespace distributed_vector
}  // namespace reference
}  // namespace kernels
}  // namespace gko

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


#include "core/distributed/partition_helpers_kernels.hpp"
#include <numeric>

#include "core/base/iterator_factory.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace partition_helpers {


template <typename GlobalIndexType>
void sort_by_range_start(
    std::shared_ptr<const DefaultExecutor> exec,
    array<GlobalIndexType>& range_start_ends,
    array<experimental::distributed::comm_index_type>& part_ids)
{
    struct range {
        GlobalIndexType idxs[2];
    };

    auto part_ids_d = part_ids.get_data();
    auto num_parts = part_ids.get_num_elems();
    auto range_it = reinterpret_cast<range*>(range_start_ends.get_data());
    auto sort_it = detail::make_zip_iterator(range_it, part_ids_d);
    std::sort(sort_it, sort_it + num_parts, [](const auto& a, const auto& b) {
        return std::get<0>(a).idxs[0] < std::get<0>(b).idxs[0];
    });
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_PARTITION_HELPERS_SORT_BY_RANGE_START);


template <typename GlobalIndexType>
void check_consecutive_ranges(std::shared_ptr<const DefaultExecutor> exec,
                              const array<GlobalIndexType>& range_start_ends,
                              bool* result)
{
    struct end_start {
        GlobalIndexType end;
        GlobalIndexType start;
    };

    auto num_parts = range_start_ends.get_num_elems() / 2;
    auto range_it = reinterpret_cast<const end_start*>(
        range_start_ends.get_const_data() + 1);

    if (num_parts) {
        *result =
            std::all_of(range_it, range_it + num_parts - 1,
                        [](const end_start& r) { return r.end == r.start; });
    } else {
        *result = true;
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_PARTITION_HELPERS_CHECK_CONSECUTIVE_RANGES);


template <typename GlobalIndexType>
void compress_ranges(std::shared_ptr<const DefaultExecutor> exec,
                     const array<GlobalIndexType>& range_start_ends,
                     array<GlobalIndexType>& range_offsets)
{
    range_offsets.get_data()[0] = range_start_ends.get_const_data()[0];
    for (int i = 0; i < range_offsets.get_num_elems() - 1; ++i) {
        range_offsets.get_data()[i + 1] =
            range_start_ends.get_const_data()[2 * i + 1];
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_PARTITION_HELPERS_COMPRESS_RANGES);


}  // namespace partition_helpers
}  // namespace reference
}  // namespace kernels
}  // namespace gko
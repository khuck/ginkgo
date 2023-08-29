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

#include <ginkgo/core/base/index_set.hpp>


#include <algorithm>
#include <iostream>
#include <mutex>
#include <vector>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/index_set_kernels.hpp"


namespace gko {
namespace idx_set {


GKO_REGISTER_OPERATION(to_global_indices, idx_set::to_global_indices);
GKO_REGISTER_OPERATION(populate_subsets, idx_set::populate_subsets);
GKO_REGISTER_OPERATION(global_to_local, idx_set::global_to_local);
GKO_REGISTER_OPERATION(local_to_global, idx_set::local_to_global);


}  // namespace idx_set


template <typename IndexType>
void index_set<IndexType>::populate_subsets(
    const gko::array<IndexType>& indices, const bool is_sorted)
{
    auto exec = this->get_executor();
    this->num_stored_indices_ = indices.size();
    exec->run(idx_set::make_populate_subsets(
        this->index_space_size_, &indices, &this->subsets_begin_,
        &this->subsets_end_, &this->superset_cumulative_indices_, is_sorted));
}


template <typename IndexType>
bool index_set<IndexType>::contains(const IndexType input_index) const
{
    auto local_index = this->get_local_index(input_index);
    return local_index != invalid_index<IndexType>();
}


template <typename IndexType>
IndexType index_set<IndexType>::get_global_index(const IndexType index) const
{
    auto exec = this->get_executor();
    const auto local_idx =
        array<IndexType>(exec, std::initializer_list<IndexType>{index});
    auto global_idx =
        array<IndexType>(exec, this->map_local_to_global(local_idx, true));

    return exec->copy_val_to_host(global_idx.data());
}


template <typename IndexType>
IndexType index_set<IndexType>::get_local_index(const IndexType index) const
{
    auto exec = this->get_executor();
    const auto global_idx =
        array<IndexType>(exec, std::initializer_list<IndexType>{index});
    auto local_idx =
        array<IndexType>(exec, this->map_global_to_local(global_idx, true));

    return exec->copy_val_to_host(local_idx.data());
}


template <typename IndexType>
array<IndexType> index_set<IndexType>::to_global_indices() const
{
    auto exec = this->get_executor();
    auto num_elems =
        exec->copy_val_to_host(this->superset_cumulative_indices_.const_data() +
                               this->superset_cumulative_indices_.size() - 1);
    auto decomp_indices = gko::array<IndexType>(exec, num_elems);
    exec->run(idx_set::make_to_global_indices(
        this->get_num_subsets(), this->get_subsets_begin(),
        this->get_subsets_end(), this->get_superset_indices(),
        decomp_indices.data()));

    return decomp_indices;
}


template <typename IndexType>
array<IndexType> index_set<IndexType>::map_local_to_global(
    const array<IndexType>& local_indices, const bool is_sorted) const
{
    auto exec = this->get_executor();
    auto global_indices = gko::array<IndexType>(exec, local_indices.size());

    GKO_ASSERT(this->get_num_subsets() >= 1);
    exec->run(idx_set::make_local_to_global(
        this->get_num_subsets(), this->get_subsets_begin(),
        this->get_superset_indices(),
        static_cast<IndexType>(local_indices.size()),
        local_indices.const_data(), global_indices.data(), is_sorted));
    return global_indices;
}


template <typename IndexType>
array<IndexType> index_set<IndexType>::map_global_to_local(
    const array<IndexType>& global_indices, const bool is_sorted) const
{
    auto exec = this->get_executor();
    auto local_indices = gko::array<IndexType>(exec, global_indices.size());

    GKO_ASSERT(this->get_num_subsets() >= 1);
    exec->run(idx_set::make_global_to_local(
        this->index_space_size_, this->get_num_subsets(),
        this->get_subsets_begin(), this->get_subsets_end(),
        this->get_superset_indices(),
        static_cast<IndexType>(local_indices.size()),
        global_indices.const_data(), local_indices.data(), is_sorted));
    return local_indices;
}


#define GKO_DECLARE_INDEX_SET(_type) class index_set<_type>
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_INDEX_SET);


}  // namespace gko

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

#ifndef GKO_CORE_DISTRIBUTED_VECTOR_KERNELS_HPP_
#define GKO_CORE_DISTRIBUTED_VECTOR_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/distributed/partition.hpp>


namespace gko {
namespace kernels {


#define GKO_DECLARE_BUILD_LOCAL(ValueType, LocalIndexType)                   \
    void build_local(                                                        \
        std::shared_ptr<const DefaultExecutor> exec,                         \
        const Array<matrix_data_entry<ValueType, global_index_type>>& input, \
        const distributed::Partition<LocalIndexType>* partition,             \
        comm_index_type local_part,                                          \
        Array<matrix_data_entry<ValueType, LocalIndexType>>& local_data,     \
        ValueType deduction_help)


#define GKO_DECLARE_ALL_AS_TEMPLATES                          \
    using global_index_type = distributed::global_index_type; \
    using comm_index_type = distributed::comm_index_type;     \
    template <typename ValueType, typename LocalIndexType>    \
    GKO_DECLARE_BUILD_LOCAL(ValueType, LocalIndexType)


namespace omp {
namespace distributed_vector {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace distributed_vector
}  // namespace omp


namespace cuda {
namespace distributed_vector {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace distributed_vector
}  // namespace cuda


namespace reference {
namespace distributed_vector {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace distributed_vector
}  // namespace reference


namespace hip {
namespace distributed_vector {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace distributed_vector
}  // namespace hip


namespace dpcpp {
namespace distributed_vector {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace distributed_vector
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_DISTRIBUTED_VECTOR_KERNELS_HPP_

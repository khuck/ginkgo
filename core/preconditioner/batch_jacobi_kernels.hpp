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

#ifndef GKO_CORE_PRECONDITIONER_BATCH_JACOBI_KERNELS_HPP_
#define GKO_CORE_PRECONDITIONER_BATCH_JACOBI_KERNELS_HPP_


#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


namespace gko {
namespace kernels {


/**
 * @fn batch_jacobi_apply
 *
 * This kernel builds a Jacobi preconditioner for each matrix in
 * the input batch of matrices and applies them to the corresponding vectors
 * in the input vector batches.
 *
 * These functions are mostly meant only for experimentation and testing.
 *
 * @param exec  The executor on which to run the kernel.
 * @param a  The batch of matrices for which to build the preconditioner.
 * @param b  The batch of input (RHS) vectors.
 * @param x  The batch of output (solution) vectors.
 */
#define GKO_DECLARE_BATCH_JACOBI_KERNEL(_type)                           \
    void batch_jacobi_apply(std::shared_ptr<const DefaultExecutor> exec, \
                            const matrix::BatchCsr<_type> *a,            \
                            const matrix::BatchDense<_type> *b,          \
                            matrix::BatchDense<_type> *x)


#define GKO_DECLARE_ALL_AS_TEMPLATES \
    template <typename ValueType>    \
    GKO_DECLARE_BATCH_JACOBI_KERNEL(ValueType)


namespace omp {
namespace batch_jacobi {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_jacobi
}  // namespace omp


namespace cuda {
namespace batch_jacobi {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_jacobi
}  // namespace cuda


namespace reference {
namespace batch_jacobi {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_jacobi
}  // namespace reference


namespace hip {
namespace batch_jacobi {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_jacobi
}  // namespace hip


namespace dpcpp {
namespace batch_jacobi {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_jacobi
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif
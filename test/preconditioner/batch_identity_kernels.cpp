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

#include <limits>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>


#include "core/preconditioner/batch_identity_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"


namespace {


template <typename T>
class BatchIdentity : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type, int>;
    using BDense = gko::matrix::BatchDense<value_type>;

    BatchIdentity()
        : exec(gko::ReferenceExecutor::create()),
          cuexec(gko::CudaExecutor::create(0, exec)),
          ref_mtx(gko::test::generate_uniform_batch_random_matrix<Mtx>(
              nbatch, nrows, nrows,
              std::uniform_int_distribution<>(1, nrows - 1),
              std::normal_distribution<real_type>(), std::ranlux48(34), true,
              exec)),
          cu_mtx(Mtx::create(cuexec))
    {
        cu_mtx->copy_from(ref_mtx.get());
    }

    void TearDown()
    {
        if (cuexec != nullptr) {
            ASSERT_NO_THROW(cuexec->synchronize());
        }
    }

    std::shared_ptr<gko::ReferenceExecutor> exec;
    std::shared_ptr<const gko::CudaExecutor> cuexec;

    const size_t nbatch = 10;
    const int nrows = 50;
    std::unique_ptr<Mtx> ref_mtx;
    std::unique_ptr<Mtx> cu_mtx;


    void check_identity(const int nrhs)
    {
        auto ref_b = gko::test::generate_uniform_batch_random_matrix<BDense>(
            nbatch, nrows, nrhs, std::uniform_int_distribution<>(nrhs, nrhs),
            std::normal_distribution<real_type>(), std::ranlux48(34), false,
            exec);
        auto cu_b = BDense::create(cuexec);
        cu_b->copy_from(ref_b.get());
        auto ref_x = BDense::create(
            exec, gko::batch_dim<>(nbatch, gko::dim<2>(nrows, nrhs)));
        auto cu_x = BDense::create(
            cuexec, gko::batch_dim<>(nbatch, gko::dim<2>(nrows, nrhs)));

        gko::kernels::cuda::batch_identity::batch_identity_apply(
            cuexec, cu_mtx.get(), cu_b.get(), cu_x.get());
        gko::kernels::reference::batch_identity::batch_identity_apply(
            exec, ref_mtx.get(), ref_b.get(), ref_x.get());

        cuexec->synchronize();
        GKO_ASSERT_BATCH_MTX_NEAR(ref_x, cu_x, 0.0);
    }
};

TYPED_TEST_SUITE(BatchIdentity, gko::test::ValueTypes);


TYPED_TEST(BatchIdentity, ApplySingleIsEquivalentToReference)
{
    this->check_identity(1);
}


}  // namespace

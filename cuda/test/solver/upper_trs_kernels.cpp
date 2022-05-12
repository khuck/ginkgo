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

#include <memory>
#include <random>


#include <cuda.h>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/triangular.hpp>


#include "core/solver/upper_trs_kernels.hpp"
#include "cuda/test/utils.hpp"


namespace {


class UpperTrs : public ::testing::Test {
protected:
    using CsrMtx = gko::matrix::Csr<double, gko::int32>;
    using Mtx = gko::matrix::Dense<double>;
    using Mtx2 = gko::matrix::Dense<float>;

    UpperTrs() : rand_engine(30) {}

    void SetUp()
    {
        ASSERT_GT(gko::CudaExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        cuda = gko::CudaExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (cuda != nullptr) {
            ASSERT_NO_THROW(cuda->synchronize());
        }
    }

    std::unique_ptr<Mtx> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    std::unique_ptr<Mtx> gen_u_mtx(int size)
    {
        return gko::test::generate_random_upper_triangular_matrix<Mtx>(
            size, false, std::uniform_int_distribution<>(size, size),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void initialize_data(int m, int n)
    {
        mtx = gen_u_mtx(m);
        b = gen_mtx(m, n);
        x = gen_mtx(m, n);
        x2 = Mtx2::create(ref);
        x2->copy_from(x.get());
        csr_mtx = CsrMtx::create(ref);
        mtx->convert_to(csr_mtx.get());
        d_csr_mtx = CsrMtx::create(cuda);
        d_x = gko::clone(cuda, x);
        d_x2 = gko::clone(cuda, x2);
        d_csr_mtx->copy_from(csr_mtx.get());
        d_b = gko::clone(cuda, b);
        b2 = Mtx2::create(ref);
        b2->copy_from(b.get());
        d_b2 = gko::clone(cuda, b2);
    }

    std::shared_ptr<Mtx> b;
    std::shared_ptr<Mtx2> b2;
    std::shared_ptr<Mtx> x;
    std::shared_ptr<Mtx2> x2;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<CsrMtx> csr_mtx;
    std::shared_ptr<Mtx> d_b;
    std::shared_ptr<Mtx2> d_b2;
    std::shared_ptr<Mtx> d_x;
    std::shared_ptr<Mtx2> d_x2;
    std::shared_ptr<CsrMtx> d_csr_mtx;
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::CudaExecutor> cuda;
    std::default_random_engine rand_engine;
};


TEST_F(UpperTrs, CudaUpperTrsFlagCheckIsCorrect)
{
    bool trans_flag = true;
    bool expected_flag = false;

    gko::kernels::cuda::upper_trs::should_perform_transpose(cuda, trans_flag);

    ASSERT_EQ(expected_flag, trans_flag);
}


TEST_F(UpperTrs, CudaSingleRhsApplySyncfreelibIsEquivalentToRef)
{
    initialize_data(50, 1);
    auto upper_trs_factory = gko::solver::UpperTrs<>::build().on(ref);
    auto d_upper_trs_factory =
        gko::solver::UpperTrs<>::build()
            .with_algorithm(gko::solver::trisolve_algorithm::syncfree)
            .on(cuda);
    auto solver = upper_trs_factory->generate(csr_mtx);
    auto d_solver = d_upper_trs_factory->generate(d_csr_mtx);

    solver->apply(b.get(), x.get());
    d_solver->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


TEST_F(UpperTrs, CudaSingleRhsApplyIsEquivalentToRef)
{
    initialize_data(50, 1);
    auto upper_trs_factory = gko::solver::UpperTrs<>::build().on(ref);
    auto d_upper_trs_factory = gko::solver::UpperTrs<>::build().on(cuda);
    auto solver = upper_trs_factory->generate(csr_mtx);
    auto d_solver = d_upper_trs_factory->generate(d_csr_mtx);

    solver->apply(b.get(), x.get());
    d_solver->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


TEST_F(UpperTrs, CudaSingleRhsMixedApplyIsEquivalentToRef1)
{
    initialize_data(50, 1);
    auto upper_trs_factory = gko::solver::UpperTrs<>::build().on(ref);
    auto d_upper_trs_factory = gko::solver::UpperTrs<>::build().on(cuda);
    auto solver = upper_trs_factory->generate(csr_mtx);
    auto d_solver = d_upper_trs_factory->generate(d_csr_mtx);

    solver->apply(b2.get(), x2.get());
    d_solver->apply(d_b2.get(), d_x2.get());

    GKO_ASSERT_MTX_NEAR(d_x2, x2, 1e-6);
}


TEST_F(UpperTrs, CudaSingleRhsMixedApplyIsEquivalentToRef2)
{
    initialize_data(50, 1);
    auto upper_trs_factory = gko::solver::UpperTrs<>::build().on(ref);
    auto d_upper_trs_factory = gko::solver::UpperTrs<>::build().on(cuda);
    auto solver = upper_trs_factory->generate(csr_mtx);
    auto d_solver = d_upper_trs_factory->generate(d_csr_mtx);

    solver->apply(b2.get(), x.get());
    d_solver->apply(d_b2.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


TEST_F(UpperTrs, CudaSingleRhsMixedApplyIsEquivalentToRef3)
{
    initialize_data(50, 1);
    auto upper_trs_factory = gko::solver::UpperTrs<>::build().on(ref);
    auto d_upper_trs_factory = gko::solver::UpperTrs<>::build().on(cuda);
    auto solver = upper_trs_factory->generate(csr_mtx);
    auto d_solver = d_upper_trs_factory->generate(d_csr_mtx);

    solver->apply(b.get(), x2.get());
    d_solver->apply(d_b.get(), d_x2.get());

    GKO_ASSERT_MTX_NEAR(d_x2, x2, 1e-6);
}


TEST_F(UpperTrs, CudaMultipleRhsApplySyncfreeIsEquivalentToRef)
{
    initialize_data(50, 3);
    auto upper_trs_factory =
        gko::solver::UpperTrs<>::build().with_num_rhs(3u).on(ref);
    auto d_upper_trs_factory =
        gko::solver::UpperTrs<>::build()
            .with_algorithm(gko::solver::trisolve_algorithm::syncfree)
            .with_num_rhs(3u)
            .on(cuda);
    auto solver = upper_trs_factory->generate(csr_mtx);
    auto d_solver = d_upper_trs_factory->generate(d_csr_mtx);
    auto db_strided = Mtx::create(cuda, b->get_size(), 4);
    d_b->convert_to(db_strided.get());
    auto dx_strided = Mtx::create(cuda, x->get_size(), 5);

    solver->apply(b.get(), x.get());
    d_solver->apply(db_strided.get(), dx_strided.get());

    GKO_ASSERT_MTX_NEAR(dx_strided, x, 1e-14);
}


TEST_F(UpperTrs, CudaMultipleRhsApplyIsEquivalentToRef)
{
    initialize_data(50, 3);
    auto upper_trs_factory =
        gko::solver::UpperTrs<>::build().with_num_rhs(3u).on(ref);
    auto d_upper_trs_factory =
        gko::solver::UpperTrs<>::build().with_num_rhs(3u).on(cuda);
    auto solver = upper_trs_factory->generate(csr_mtx);
    auto d_solver = d_upper_trs_factory->generate(d_csr_mtx);
    auto db_strided = Mtx::create(cuda, b->get_size(), 4);
    d_b->convert_to(db_strided.get());
    // The cuSPARSE Generic SpSM implementation uses the wrong stride here
    // so the input and output stride need to match
#if CUDA_VERSION >= 11031
    auto dx_strided = Mtx::create(cuda, x->get_size(), 4);
#else
    auto dx_strided = Mtx::create(cuda, x->get_size(), 5);
#endif

    solver->apply(b.get(), x.get());
    d_solver->apply(db_strided.get(), dx_strided.get());

    GKO_ASSERT_MTX_NEAR(dx_strided, x, 1e-14);
}


TEST_F(UpperTrs, CudaMultipleRhsMixedApplyIsEquivalentToRef1)
{
    initialize_data(50, 3);
    auto upper_trs_factory =
        gko::solver::UpperTrs<>::build().with_num_rhs(3u).on(ref);
    auto d_upper_trs_factory =
        gko::solver::UpperTrs<>::build().with_num_rhs(3u).on(cuda);
    auto solver = upper_trs_factory->generate(csr_mtx);
    auto d_solver = d_upper_trs_factory->generate(d_csr_mtx);
    auto db2_strided = Mtx2::create(cuda, b->get_size(), 4);
    d_b2->convert_to(db2_strided.get());
    auto dx2_strided = Mtx2::create(cuda, x2->get_size(), 5);

    solver->apply(b2.get(), x2.get());
    d_solver->apply(db2_strided.get(), dx2_strided.get());

    GKO_ASSERT_MTX_NEAR(dx2_strided, x2, 1e-14);
}


TEST_F(UpperTrs, CudaMultipleRhsMixedApplyIsEquivalentToRef2)
{
    initialize_data(50, 3);
    auto upper_trs_factory =
        gko::solver::UpperTrs<>::build().with_num_rhs(3u).on(ref);
    auto d_upper_trs_factory =
        gko::solver::UpperTrs<>::build().with_num_rhs(3u).on(cuda);
    auto solver = upper_trs_factory->generate(csr_mtx);
    auto d_solver = d_upper_trs_factory->generate(d_csr_mtx);
    auto db2_strided = Mtx2::create(cuda, b->get_size(), 4);
    d_b2->convert_to(db2_strided.get());
    auto dx_strided = Mtx::create(cuda, x->get_size(), 5);

    solver->apply(b2.get(), x.get());
    d_solver->apply(db2_strided.get(), dx_strided.get());

    GKO_ASSERT_MTX_NEAR(dx_strided, x, 1e-14);
}


TEST_F(UpperTrs, CudaApplyThrowsWithWrongNumRHS)
{
    initialize_data(50, 3);
    auto d_lower_trs_factory = gko::solver::UpperTrs<>::build().on(cuda);
    auto d_solver = d_lower_trs_factory->generate(d_csr_mtx);

    ASSERT_THROW(d_solver->apply(d_b2.get(), d_x.get()), gko::ValueMismatch);
}


TEST_F(UpperTrs, CudaMultipleRhsMixedApplyIsEquivalentToRef3)
{
    initialize_data(50, 3);
    auto upper_trs_factory =
        gko::solver::UpperTrs<>::build().with_num_rhs(3u).on(ref);
    auto d_upper_trs_factory =
        gko::solver::UpperTrs<>::build().with_num_rhs(3u).on(cuda);
    auto solver = upper_trs_factory->generate(csr_mtx);
    auto d_solver = d_upper_trs_factory->generate(d_csr_mtx);
    auto db_strided = Mtx::create(cuda, b->get_size(), 4);
    d_b->convert_to(db_strided.get());
    auto dx2_strided = Mtx2::create(cuda, x2->get_size(), 5);

    solver->apply(b.get(), x2.get());
    d_solver->apply(db_strided.get(), dx2_strided.get());

    GKO_ASSERT_MTX_NEAR(dx2_strided, x2, 1e-14);
}


}  // namespace

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

#include <ginkgo/core/solver/bicg.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/solver/bicg_kernels.hpp"
#include "core/test/utils.hpp"
#include "matrices/config.hpp"
#include "test/utils/executor.hpp"


namespace {


class Bicg : public ::testing::Test {
protected:
    using value_type = gko::default_precision;
    using index_type = gko::int32;
    using Mtx = gko::matrix::Dense<>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    Bicg() : rand_engine(30) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        init_executor(ref, exec);

        std::string file_name(gko::matrices::location_ani1_mtx);
        auto input_file = std::ifstream(file_name, std::ios::in);
        if (!input_file) {
            FAIL() << "Could not find the file \"" << file_name
                   << "\", which is required for this test.\n";
        }
        csr = gko::read<Csr>(input_file, ref);
        auto csr_hip_temp = Csr::create(exec);
        csr_hip_temp->copy_from(gko::lend(csr));
        d_csr = gko::give(csr_hip_temp);
    }

    void TearDown()
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    std::unique_ptr<Mtx> gen_mtx(gko::size_type num_rows,
                                 gko::size_type num_cols)
    {
        return gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void initialize_data()
    {
        gko::size_type m = 597;
        gko::size_type n = 43;
        b_full = gen_mtx(m, n + 2);
        b = b_full->create_submatrix(gko::span{0, m}, gko::span{0, n});
        r = gen_mtx(m, n);
        z = gen_mtx(m, n);
        p = gen_mtx(m, n);
        q = gen_mtx(m, n);
        r2 = gen_mtx(m, n);
        z2 = gen_mtx(m, n);
        p2 = gen_mtx(m, n);
        q2 = gen_mtx(m, n);
        x_full = gen_mtx(m, n + 3);
        x = x_full->create_submatrix(gko::span{0, m}, gko::span{0, n});
        beta = gen_mtx(1, n);
        prev_rho = gen_mtx(1, n);
        rho = gen_mtx(1, n);
        // check correct handling for zero values
        beta->at(2) = 0.0;
        prev_rho->at(2) = 0.0;
        stop_status = std::unique_ptr<gko::Array<gko::stopping_status>>(
            new gko::Array<gko::stopping_status>(ref, n));
        for (size_t i = 0; i < stop_status->get_num_elems(); ++i) {
            stop_status->get_data()[i].reset();
        }
        // check correct handling for stopped columns
        stop_status->get_data()[1].stop(1);

        d_b = Mtx::create(exec);
        d_b->copy_from(b.get());
        d_r = Mtx::create(exec);
        d_r->copy_from(r.get());
        d_z = Mtx::create(exec);
        d_z->copy_from(z.get());
        d_p = Mtx::create(exec);
        d_p->copy_from(p.get());
        d_q = Mtx::create(exec);
        d_q->copy_from(q.get());
        d_r2 = Mtx::create(exec);
        d_r2->copy_from(r2.get());
        d_z2 = Mtx::create(exec);
        d_z2->copy_from(z2.get());
        d_p2 = Mtx::create(exec);
        d_p2->copy_from(p2.get());
        d_q2 = Mtx::create(exec);
        d_q2->copy_from(q2.get());
        d_x = Mtx::create(exec);
        d_x->copy_from(x.get());
        d_beta = Mtx::create(exec);
        d_beta->copy_from(beta.get());
        d_prev_rho = Mtx::create(exec);
        d_prev_rho->copy_from(prev_rho.get());
        d_rho = Mtx::create(exec);
        d_rho->copy_from(rho.get());
        d_stop_status = std::unique_ptr<gko::Array<gko::stopping_status>>(
            new gko::Array<gko::stopping_status>(exec, n));
        *d_stop_status = *stop_status;
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;

    std::ranlux48 rand_engine;

    std::unique_ptr<Mtx> b_full;
    std::unique_ptr<Mtx> b;
    std::unique_ptr<Mtx> r;
    std::unique_ptr<Mtx> z;
    std::unique_ptr<Mtx> p;
    std::unique_ptr<Mtx> q;
    std::unique_ptr<Mtx> r2;
    std::unique_ptr<Mtx> z2;
    std::unique_ptr<Mtx> p2;
    std::unique_ptr<Mtx> q2;
    std::unique_ptr<Mtx> x_full;
    std::unique_ptr<Mtx> x;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Mtx> prev_rho;
    std::unique_ptr<Mtx> rho;
    std::shared_ptr<Csr> csr;
    std::unique_ptr<gko::Array<gko::stopping_status>> stop_status;

    std::unique_ptr<Mtx> d_b;
    std::unique_ptr<Mtx> d_r;
    std::unique_ptr<Mtx> d_z;
    std::unique_ptr<Mtx> d_p;
    std::unique_ptr<Mtx> d_q;
    std::unique_ptr<Mtx> d_r2;
    std::unique_ptr<Mtx> d_z2;
    std::unique_ptr<Mtx> d_p2;
    std::unique_ptr<Mtx> d_q2;
    std::unique_ptr<Mtx> d_x;
    std::unique_ptr<Mtx> d_beta;
    std::unique_ptr<Mtx> d_prev_rho;
    std::unique_ptr<Mtx> d_rho;
    std::shared_ptr<Csr> d_csr;
    std::unique_ptr<gko::Array<gko::stopping_status>> d_stop_status;
};


TEST_F(Bicg, BicgInitializeIsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::bicg::initialize(
        ref, b.get(), r.get(), z.get(), p.get(), q.get(), prev_rho.get(),
        rho.get(), r2.get(), z2.get(), p2.get(), q2.get(), stop_status.get());
    gko::kernels::EXEC_NAMESPACE::bicg::initialize(
        exec, d_b.get(), d_r.get(), d_z.get(), d_p.get(), d_q.get(),
        d_prev_rho.get(), d_rho.get(), d_r2.get(), d_z2.get(), d_p2.get(),
        d_q2.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_r, r, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_z, z, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_p, p, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_q, q, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_r2, r2, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_z2, z2, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_p2, p2, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_q2, q2, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_prev_rho, prev_rho, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_rho, rho, 1e-14);
    GKO_ASSERT_ARRAY_EQ(*d_stop_status, *stop_status);
}


TEST_F(Bicg, BicgStep1IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::bicg::step_1(ref, p.get(), z.get(), p2.get(),
                                          z2.get(), rho.get(), prev_rho.get(),
                                          stop_status.get());
    gko::kernels::EXEC_NAMESPACE::bicg::step_1(
        exec, d_p.get(), d_z.get(), d_p2.get(), d_z2.get(), d_rho.get(),
        d_prev_rho.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_p, p, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_z, z, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_p2, p2, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_z2, z2, 1e-14);
}


TEST_F(Bicg, BicgStep2IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::bicg::step_2(
        ref, x.get(), r.get(), r2.get(), p.get(), q.get(), q2.get(), beta.get(),
        rho.get(), stop_status.get());
    gko::kernels::EXEC_NAMESPACE::bicg::step_2(
        exec, d_x.get(), d_r.get(), d_r2.get(), d_p.get(), d_q.get(),
        d_q2.get(), d_beta.get(), d_rho.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_r, r, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_r2, r2, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_p, p, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_q, q, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_q2, q2, 1e-14);
}


TEST_F(Bicg, ApplyWithSpdMatrixIsEquivalentToRef)
{
    auto mtx = gen_mtx(50, 50);
    gko::test::make_hpd(mtx.get());
    auto x = gen_mtx(50, 3);
    auto b = gen_mtx(50, 3);
    auto d_mtx = Mtx::create(exec);
    d_mtx->copy_from(mtx.get());
    auto d_x = Mtx::create(exec);
    d_x->copy_from(x.get());
    auto d_b = Mtx::create(exec);
    d_b->copy_from(b.get());
    auto bicg_factory =
        gko::solver::Bicg<>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(50u).on(ref),
                gko::stop::ResidualNorm<>::build()
                    .with_reduction_factor(1e-14)
                    .on(ref))
            .on(ref);
    auto d_bicg_factory =
        gko::solver::Bicg<>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(50u).on(exec),
                gko::stop::ResidualNorm<>::build()
                    .with_reduction_factor(1e-14)
                    .on(exec))
            .on(exec);
    auto solver = bicg_factory->generate(std::move(mtx));
    auto d_solver = d_bicg_factory->generate(std::move(d_mtx));

    solver->apply(b.get(), x.get());
    d_solver->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


TEST_F(Bicg, ApplyWithSuiteSparseMatrixIsEquivalentToRef)
{
    auto x = gen_mtx(36, 1);
    auto b = gen_mtx(36, 1);
    auto d_x = Mtx::create(exec);
    d_x->copy_from(x.get());
    auto d_b = Mtx::create(exec);
    d_b->copy_from(b.get());
    auto bicg_factory =
        gko::solver::Bicg<>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(50u).on(ref),
                gko::stop::ResidualNorm<>::build()
                    .with_reduction_factor(1e-14)
                    .on(ref))
            .on(ref);
    auto d_bicg_factory =
        gko::solver::Bicg<>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(50u).on(exec),
                gko::stop::ResidualNorm<>::build()
                    .with_reduction_factor(1e-14)
                    .on(exec))
            .on(exec);
    auto solver = bicg_factory->generate(std::move(csr));
    auto d_solver = d_bicg_factory->generate(std::move(d_csr));

    solver->apply(b.get(), x.get());
    d_solver->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


}  // namespace

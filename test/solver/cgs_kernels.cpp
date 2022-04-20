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

#include "core/solver/cgs_kernels.hpp"


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/cgs.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/test/utils.hpp"
#include "core/utils/matrix_utils.hpp"
#include "test/utils/executor.hpp"


namespace {


class Cgs : public ::testing::Test {
protected:
#if GINKGO_COMMON_SINGLE_MODE
    using value_type = float;
#else
    using value_type = double;
#endif
    using index_type = int;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Cgs<value_type>;

    Cgs() : rand_engine(30) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        init_executor(ref, exec);

        auto data = gko::matrix_data<value_type, index_type>(
            gko::dim<2>{123, 123},
            std::normal_distribution<value_type>(-1.0, 1.0), rand_engine);
        gko::test::make_diag_dominant(data);
        mtx = Mtx::create(ref, data.size, 125);
        mtx->read(data);
        d_mtx = gko::clone(exec, mtx);
        exec_cgs_factory =
            Solver::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(246u).on(exec),
                    gko::stop::ResidualNorm<value_type>::build()
                        .with_reduction_factor(::r<value_type>::value)
                        .on(exec))
                .on(exec);
        ref_cgs_factory =
            Solver::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(246u).on(ref),
                    gko::stop::ResidualNorm<value_type>::build()
                        .with_reduction_factor(::r<value_type>::value)
                        .on(ref))
                .on(ref);
    }

    void TearDown()
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    std::unique_ptr<Mtx> gen_mtx(gko::size_type num_rows,
                                 gko::size_type num_cols, gko::size_type stride)
    {
        auto tmp_mtx = gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<value_type>(-1.0, 1.0), rand_engine, ref);
        auto result = Mtx::create(ref, gko::dim<2>{num_rows, num_cols}, stride);
        result->copy_from(tmp_mtx.get());
        return result;
    }

    void initialize_data()
    {
        gko::size_type m = 597;
        gko::size_type n = 43;
        b = gen_mtx(m, n, n + 2);
        r = gen_mtx(m, n, n + 2);
        r_tld = gen_mtx(m, n, n + 2);
        p = gen_mtx(m, n, n + 2);
        q = gen_mtx(m, n, n + 2);
        u = gen_mtx(m, n, n + 2);
        u_hat = gen_mtx(m, n, n + 2);
        v_hat = gen_mtx(m, n, n + 2);
        t = gen_mtx(m, n, n + 2);
        x = gen_mtx(m, n, n + 3);
        alpha = gen_mtx(1, n, n);
        beta = gen_mtx(1, n, n);
        gamma = gen_mtx(1, n, n);
        rho = gen_mtx(1, n, n);
        rho_prev = gen_mtx(1, n, n);
        // check correct handling for zero values
        gamma->at(2) = 0.0;
        rho_prev->at(2) = 0.0;
        stop_status =
            std::make_unique<gko::Array<gko::stopping_status>>(ref, n);
        for (size_t i = 0; i < stop_status->get_num_elems(); ++i) {
            stop_status->get_data()[i].reset();
        }
        // check correct handling for stopped columns
        stop_status->get_data()[1].stop(1);

        d_b = gko::clone(exec, b);
        d_r = gko::clone(exec, r);
        d_r_tld = gko::clone(exec, r_tld);
        d_p = gko::clone(exec, p);
        d_q = gko::clone(exec, q);
        d_u = gko::clone(exec, u);
        d_u_hat = gko::clone(exec, u_hat);
        d_v_hat = gko::clone(exec, v_hat);
        d_t = gko::clone(exec, t);
        d_x = gko::clone(exec, x);
        d_alpha = gko::clone(exec, alpha);
        d_beta = gko::clone(exec, beta);
        d_gamma = gko::clone(exec, gamma);
        d_rho_prev = gko::clone(exec, rho_prev);
        d_rho = gko::clone(exec, rho);
        d_stop_status = std::make_unique<gko::Array<gko::stopping_status>>(
            exec, *stop_status);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;

    std::default_random_engine rand_engine;

    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> d_mtx;
    std::unique_ptr<Solver::Factory> exec_cgs_factory;
    std::unique_ptr<Solver::Factory> ref_cgs_factory;

    std::unique_ptr<Mtx> b;
    std::unique_ptr<Mtx> r;
    std::unique_ptr<Mtx> r_tld;
    std::unique_ptr<Mtx> t;
    std::unique_ptr<Mtx> p;
    std::unique_ptr<Mtx> q;
    std::unique_ptr<Mtx> u;
    std::unique_ptr<Mtx> u_hat;
    std::unique_ptr<Mtx> v_hat;
    std::unique_ptr<Mtx> x;
    std::unique_ptr<Mtx> alpha;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Mtx> gamma;
    std::unique_ptr<Mtx> rho;
    std::unique_ptr<Mtx> rho_prev;
    std::unique_ptr<gko::Array<gko::stopping_status>> stop_status;

    std::unique_ptr<Mtx> d_b;
    std::unique_ptr<Mtx> d_r;
    std::unique_ptr<Mtx> d_r_tld;
    std::unique_ptr<Mtx> d_t;
    std::unique_ptr<Mtx> d_p;
    std::unique_ptr<Mtx> d_q;
    std::unique_ptr<Mtx> d_u;
    std::unique_ptr<Mtx> d_u_hat;
    std::unique_ptr<Mtx> d_v_hat;
    std::unique_ptr<Mtx> d_x;
    std::unique_ptr<Mtx> d_alpha;
    std::unique_ptr<Mtx> d_beta;
    std::unique_ptr<Mtx> d_gamma;
    std::unique_ptr<Mtx> d_rho;
    std::unique_ptr<Mtx> d_rho_prev;
    std::unique_ptr<gko::Array<gko::stopping_status>> d_stop_status;
};


TEST_F(Cgs, CgsInitializeIsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::cgs::initialize(
        ref, b.get(), r.get(), r_tld.get(), p.get(), q.get(), u.get(),
        u_hat.get(), v_hat.get(), t.get(), alpha.get(), beta.get(), gamma.get(),
        rho_prev.get(), rho.get(), stop_status.get());
    gko::kernels::EXEC_NAMESPACE::cgs::initialize(
        exec, d_b.get(), d_r.get(), d_r_tld.get(), d_p.get(), d_q.get(),
        d_u.get(), d_u_hat.get(), d_v_hat.get(), d_t.get(), d_alpha.get(),
        d_beta.get(), d_gamma.get(), d_rho_prev.get(), d_rho.get(),
        d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_r, r, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_r_tld, r_tld, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_p, p, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_q, q, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_u, u, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_t, t, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_u_hat, u_hat, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_v_hat, v_hat, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_rho_prev, rho_prev, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_rho, rho, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_alpha, alpha, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_beta, beta, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_gamma, gamma, ::r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(*d_stop_status, *stop_status);
}


TEST_F(Cgs, CgsStep1IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::cgs::step_1(ref, r.get(), u.get(), p.get(),
                                         q.get(), beta.get(), rho.get(),
                                         rho_prev.get(), stop_status.get());
    gko::kernels::EXEC_NAMESPACE::cgs::step_1(
        exec, d_r.get(), d_u.get(), d_p.get(), d_q.get(), d_beta.get(),
        d_rho.get(), d_rho_prev.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_beta, beta, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_u, u, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_p, p, ::r<value_type>::value);
}


TEST_F(Cgs, CgsStep2IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::cgs::step_2(ref, u.get(), v_hat.get(), q.get(),
                                         t.get(), alpha.get(), rho.get(),
                                         gamma.get(), stop_status.get());
    gko::kernels::EXEC_NAMESPACE::cgs::step_2(
        exec, d_u.get(), d_v_hat.get(), d_q.get(), d_t.get(), d_alpha.get(),
        d_rho.get(), d_gamma.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_alpha, alpha, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_t, t, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_q, q, ::r<value_type>::value);
}


TEST_F(Cgs, CgsStep3IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::cgs::step_3(ref, t.get(), u_hat.get(), r.get(),
                                         x.get(), alpha.get(),
                                         stop_status.get());
    gko::kernels::EXEC_NAMESPACE::cgs::step_3(
        exec, d_t.get(), d_u_hat.get(), d_r.get(), d_x.get(), d_alpha.get(),
        d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_r, r, ::r<value_type>::value);
}


TEST_F(Cgs, CgsApplyOneRHSIsEquivalentToRef)
{
    int m = 123;
    int n = 1;
    auto ref_solver = ref_cgs_factory->generate(mtx);
    auto exec_solver = exec_cgs_factory->generate(d_mtx);
    auto b = gen_mtx(m, n, n + 4);
    auto x = gen_mtx(m, n, n + 2);
    auto d_b = gko::clone(exec, b);
    auto d_x = gko::clone(exec, x);

    ref_solver->apply(b.get(), x.get());
    exec_solver->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_b, b, ::r<value_type>::value * 1e3);
    GKO_ASSERT_MTX_NEAR(d_x, x, ::r<value_type>::value * 1e3);
}


TEST_F(Cgs, CgsApplyMultipleRHSIsEquivalentToRef)
{
    int m = 123;
    int n = 16;
    auto exec_solver = exec_cgs_factory->generate(d_mtx);
    auto ref_solver = ref_cgs_factory->generate(mtx);
    auto b = gen_mtx(m, n, n + 1);
    auto x = gen_mtx(m, n, n);
    auto d_b = gko::clone(exec, b);
    auto d_x = gko::clone(exec, x);

    ref_solver->apply(b.get(), x.get());
    exec_solver->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_b, b, ::r<value_type>::value * 5e3);
    GKO_ASSERT_MTX_NEAR(d_x, x, ::r<value_type>::value * 5e3);
}

}  // namespace

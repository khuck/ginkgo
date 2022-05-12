
#include <ginkgo/core/solver/minres.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>


#include "core/solver/minres_kernels.hpp"
#include "core/test/utils.hpp"

namespace {

template <typename T>
class Minres : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Minres<value_type>;

    Minres()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{1, 2, 3, 4}, {2, 5, 6, 7}, {3, 6, 0, 0}, {4, 7, 0, 0}}, exec)),
          zero(gko::initialize<Mtx>(I<I<value_type>>{{0, 0}, {0, 0}}, exec)),
          zero_scalar(gko::initialize<Mtx>(I<I<value_type>>{{0, 0}}, exec)),
          small_x(gko::clone(zero)),
          small_r(gko::clone(zero)),
          small_z(gko::clone(zero)),
          small_p(gko::clone(zero)),
          small_q(gko::clone(zero)),
          small_z_tilde(gko::clone(zero)),
          small_q_tilde(gko::clone(zero)),
          small_p_prev(gko::clone(zero)),
          small_q_prev(gko::clone(zero)),
          alpha(gko::clone(zero_scalar)),
          beta(gko::clone(zero_scalar)),
          gamma(gko::clone(zero_scalar)),
          delta(gko::clone(zero_scalar)),
          eta_next(gko::clone(zero_scalar)),
          eta(gko::clone(zero_scalar)),
          tau(gko::clone(zero_scalar)),
          cos_prev(gko::clone(zero_scalar)),
          cos(gko::clone(zero_scalar)),
          sin_prev(gko::clone(zero_scalar)),
          sin(gko::clone(zero_scalar)),
          stopped(),
          non_stopped(),
          small_stop(exec, 2),
          minres_factory(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(40u).on(
                          exec),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_reduction_factor(r<value_type>::value / 2)
                          .on(exec))
                  .on(exec))
    {
        stopped.stop(1);
        non_stopped.reset();
        std::fill_n(small_stop.get_data(), small_stop.get_num_elems(),
                    non_stopped);
    }


    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<Mtx> mtx;

    std::unique_ptr<Mtx> zero;
    std::unique_ptr<Mtx> zero_scalar;
    std::unique_ptr<Mtx> small_x;
    std::unique_ptr<Mtx> small_r;
    std::unique_ptr<Mtx> small_z;
    std::unique_ptr<Mtx> small_p;
    std::unique_ptr<Mtx> small_q;
    std::unique_ptr<Mtx> small_z_tilde;
    std::unique_ptr<Mtx> small_q_tilde;
    std::unique_ptr<Mtx> small_p_prev;
    std::unique_ptr<Mtx> small_q_prev;

    std::unique_ptr<Mtx> alpha;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Mtx> gamma;
    std::unique_ptr<Mtx> delta;
    std::unique_ptr<Mtx> eta_next;
    std::unique_ptr<Mtx> eta;
    std::unique_ptr<Mtx> tau;
    std::unique_ptr<Mtx> cos_prev;
    std::unique_ptr<Mtx> cos;
    std::unique_ptr<Mtx> sin_prev;
    std::unique_ptr<Mtx> sin;

    gko::stopping_status stopped;
    gko::stopping_status non_stopped;
    gko::Array<gko::stopping_status> small_stop;

    std::unique_ptr<typename Solver::Factory> minres_factory;
};

TYPED_TEST_SUITE(Minres, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Minres, KernelInitialize)
{
    using Mtx = typename TestFixture::Mtx;
    using vt = typename TestFixture::value_type;
    this->small_r =
        gko::initialize<Mtx>(I<I<vt>>({{1, 2}, {3, 4}}), this->exec);
    this->small_z = gko::initialize<Mtx>(I<I<vt>>{{4, 3}, {2, 1}}, this->exec);
    this->beta = gko::initialize<Mtx>(I<I<vt>>{{4, 25}}, this->exec);
    this->small_p->fill(1);
    this->small_q->fill(0);
    this->small_p_prev->fill(1);
    this->small_q_prev->fill(1);
    this->gamma->fill(1);
    this->delta->fill(1);
    this->cos_prev->fill(1);
    this->cos->fill(10);
    this->sin_prev->fill(1);
    this->sin_prev->fill(1);
    this->eta_next->fill(1);
    this->eta->fill(1);
    std::fill_n(this->small_stop.get_data(), this->small_stop.get_num_elems(),
                this->stopped);

    gko::kernels::reference::minres::initialize(
        this->exec, this->small_r.get(), this->small_z.get(),
        this->small_p.get(), this->small_p_prev.get(), this->small_q.get(),
        this->small_q_prev.get(), this->beta.get(), this->gamma.get(),
        this->delta.get(), this->cos_prev.get(), this->cos.get(),
        this->sin_prev.get(), this->sin.get(), this->eta_next.get(),
        this->eta.get(), &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_q, l({{1. / 2, 2. / 5}, {3. / 2, 4. / 5}}),
                        r<vt>::value);
    GKO_ASSERT_MTX_NEAR(this->small_z, l({{4. / 2, 3. / 5}, {2. / 2, 1. / 5}}),
                        r<vt>::value);
    GKO_ASSERT_MTX_NEAR(this->small_p, this->zero, 0);
    GKO_ASSERT_MTX_NEAR(this->small_p_prev, this->zero, 0);
    GKO_ASSERT_MTX_NEAR(this->small_q_prev, this->zero, 0);
    GKO_ASSERT_MTX_NEAR(this->beta, l({{2., 5.}}), 0);
    GKO_ASSERT_MTX_NEAR(this->delta, this->zero_scalar, 0);
    GKO_ASSERT_MTX_NEAR(this->gamma, this->zero_scalar, 0);
    GKO_ASSERT_MTX_NEAR(this->cos_prev, this->zero_scalar, 0);
    GKO_ASSERT_MTX_NEAR(this->cos, l({{1., 1.}}), 0);
    GKO_ASSERT_MTX_NEAR(this->sin_prev, this->zero_scalar, 0);
    GKO_ASSERT_MTX_NEAR(this->sin, this->zero_scalar, 0);
    GKO_ASSERT_MTX_NEAR(this->eta, this->beta, 0);
    GKO_ASSERT_MTX_NEAR(this->eta_next, this->beta, 0);
    ASSERT_EQ(this->small_stop.get_data()[0], this->non_stopped);
    ASSERT_EQ(this->small_stop.get_data()[1], this->non_stopped);
}


TYPED_TEST(Minres, KernelStep1)
{
    using Mtx = typename TestFixture::Mtx;
    using vt = typename TestFixture::value_type;
    this->beta = gko::initialize<Mtx>(I<I<vt>>{{4, 9}}, this->exec);
    this->small_q = gko::initialize<Mtx>(I<I<vt>>{{4, 9}, {7, 11}}, this->exec);
    this->small_q_tilde =
        gko::initialize<Mtx>(I<I<vt>>{{1, 3}, {4, 5}}, this->exec);
    this->small_p_prev->fill(1.);
    this->alpha = gko::initialize<Mtx>(I<I<vt>>{{1, 2}}, this->exec);
    this->gamma = gko::initialize<Mtx>(I<I<vt>>{{3, 6}}, this->exec);
    this->delta = gko::initialize<Mtx>(I<I<vt>>{{4, 5}}, this->exec);
    this->cos_prev = gko::initialize<Mtx>(I<I<vt>>{{0.1, 0.2}}, this->exec);
    this->cos = gko::initialize<Mtx>(I<I<vt>>{{4, 1}}, this->exec);
    this->sin_prev = gko::initialize<Mtx>(I<I<vt>>{{3, 7}}, this->exec);
    this->sin = gko::initialize<Mtx>(I<I<vt>>{{8, 3}}, this->exec);
    this->eta_next = gko::initialize<Mtx>(I<I<vt>>{{6, 4}}, this->exec);
    this->eta->fill(0.);
    this->tau = gko::initialize<Mtx>(I<I<vt>>{{4, 3}}, this->exec);
    this->small_p = gko::initialize<Mtx>(I<I<vt>>{{3, 4}, {-5, 2}}, this->exec);
    this->small_p_prev->fill(1.);
    this->small_z = gko::initialize<Mtx>(I<I<vt>>{{6, 1}, {7, 3}}, this->exec);
    this->small_z_tilde =
        gko::initialize<Mtx>(I<I<vt>>{{4, 1}, {2, 5}}, this->exec);
    this->small_x = gko::initialize<Mtx>(I<I<vt>>{{5, 6}, {9, 3}}, this->exec);
    auto old_small_q = gko::clone(this->small_q);
    auto old_cos = gko::clone(this->cos);
    auto old_sin = gko::clone(this->sin);
    auto old_eta_next = gko::clone(this->eta_next);
    auto old_p = gko::clone(this->small_p);

    gko::kernels::reference::minres::step_1(
        this->exec, this->small_x.get(), this->small_p.get(),
        this->small_p_prev.get(), this->small_z.get(),
        this->small_z_tilde.get(), this->small_q.get(),
        this->small_q_prev.get(), this->small_q_tilde.get(), this->alpha.get(),
        this->beta.get(), this->gamma.get(), this->delta.get(),
        this->cos_prev.get(), this->cos.get(), this->sin_prev.get(),
        this->sin.get(), this->eta.get(), this->eta_next.get(), this->tau.get(),
        &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->beta, l({{2., 3.}}), r<vt>::value);
    GKO_ASSERT_MTX_NEAR(this->small_q_prev, old_small_q, 0.);
    GKO_ASSERT_MTX_NEAR(this->small_q, l({{1 / 2., 3 / 3.}, {4 / 2., 5 / 3.}}),
                        r<vt>::value);
    GKO_ASSERT_MTX_NEAR(this->delta, l({{3 * 3., 7 * 6.}}), r<vt>::value);
    GKO_ASSERT_MTX_NEAR(this->gamma,
                        l({{4. * 0.1 * 3 + 8 * 1, 1 * 0.2 * 6 + 3 * 2}}),
                        r<vt>::value);
    GKO_ASSERT_MTX_NEAR(this->alpha, l({{2.561249694973139, 3.4}}),
                        r<vt>::value);
    GKO_ASSERT_MTX_NEAR(this->cos_prev, old_cos, 0.);
    GKO_ASSERT_MTX_NEAR(this->sin_prev, old_sin, 0.);
    GKO_ASSERT_MTX_NEAR(this->cos,
                        l({{0.6246950475544241, -0.47058823529411775}}),
                        r<vt>::value);
    GKO_ASSERT_MTX_NEAR(
        this->sin, l({{0.7808688094430303, 0.8823529411764705}}), r<vt>::value);
    GKO_ASSERT_MTX_NEAR(this->eta, old_eta_next, 0.);
    GKO_ASSERT_MTX_NEAR(this->eta_next,
                        l({{-4.685212856658182, -3.529411764705882}}),
                        r<vt>::value);
    GKO_ASSERT_MTX_NEAR(this->tau, l({{3.123475237772121, 2.6470588235294112}}),
                        r<vt>::value);
    GKO_ASSERT_MTX_NEAR(this->small_p_prev, old_p, 0.);
    GKO_ASSERT_MTX_NEAR(this->small_p,
                        l({{-11.947292784478368, -20.52941176470588},
                           {17.179113807746674, -15.705882352941176}}),
                        r<vt>::value);
    GKO_ASSERT_MTX_NEAR(this->small_x,
                        l({{-39.78048780487805, 44.64359861591696},
                           {73.39024390243904, 32.564013840830455}}),
                        r<vt>::value);
    GKO_ASSERT_MTX_NEAR(this->small_z, l({{4 / 2., 1 / 3.}, {2 / 2., 5 / 3.}}),
                        r<vt>::value);
}

TYPED_TEST(Minres, SolvesSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using vt = typename TestFixture::value_type;
    auto one_op = gko::initialize<Mtx>({gko::one<vt>()}, this->exec);
    auto neg_one_op = gko::initialize<Mtx>({-gko::one<vt>()}, this->exec);
    auto solver = this->minres_factory->template generate(this->mtx);
    auto x = gko::initialize<Mtx>({-1., 2., 3., 4.}, this->exec);
    auto b = Mtx::create(this->exec, x->get_size());
    this->mtx->apply(x.get(), b.get());
    auto solution = gko::clone(x);
    x->fill(0.);

    solver->apply(b.get(), x.get());
    this->mtx->apply(one_op.get(), x.get(), neg_one_op.get(), b.get());

    GKO_ASSERT_MTX_NEAR(b, l({0., 0., 0., 0.}), r<vt>::value);
}


}  // namespace

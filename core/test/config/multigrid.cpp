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

#include <typeinfo>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/multigrid/fixed_coarsening.hpp>
#include <ginkgo/core/multigrid/pgm.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/multigrid.hpp>
#include <ginkgo/core/stop/iteration.hpp>


#include "core/config/config.hpp"
#include "core/test/config/utils.hpp"
#include "core/test/utils.hpp"


using namespace gko::config;


template <typename ExplicitType, typename DefaultType>
struct MultigridLevelConfigTest {
    using explicit_type = ExplicitType;
    using default_type = DefaultType;
    using multigrid_level_config_test = MultigridLevelConfigTest;

    static void change_template(pnode& config)
    {
        config.get_list()["ValueType"] = pnode{"float"};
        config.get_list()["IndexType"] = pnode{"int64"};
    }
};


struct Pgm : MultigridLevelConfigTest<gko::multigrid::Pgm<float, gko::int64>,
                                      gko::multigrid::Pgm<double, int>> {
    static pnode setup_base() { return pnode{{{"Type", pnode{"Pgm"}}}}; }

    template <typename ParamType>
    static void set(pnode& config, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config.get_list()["max_iterations"] = pnode{20};
        param.with_max_iterations(20u);
        config.get_list()["max_unassigned_ratio"] = pnode{0.1};
        param.with_max_unassigned_ratio(0.1);
        config.get_list()["deterministic"] = pnode{true};
        param.with_deterministic(true);
        config.get_list()["skip_sorting"] = pnode{true};
        param.with_skip_sorting(true);
    }

    template <typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        ASSERT_EQ(res_param.max_iterations, ans_param.max_iterations);
        ASSERT_EQ(res_param.max_unassigned_ratio,
                  ans_param.max_unassigned_ratio);
        ASSERT_EQ(res_param.deterministic, ans_param.deterministic);
        ASSERT_EQ(res_param.skip_sorting, ans_param.skip_sorting);
    }
};


struct FixedCoarsening : MultigridLevelConfigTest<
                             gko::multigrid::FixedCoarsening<float, gko::int64>,
                             gko::multigrid::FixedCoarsening<double, int>> {
    static pnode setup_base()
    {
        return pnode{{{"Type", pnode{"FixedCoarsening"}}}};
    }

    template <typename ParamType>
    static void set(pnode& config, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config.get_list()["coarse_rows"] =
            pnode{std::vector<pnode>{{2}, {3}, {5}}};
        param.with_coarse_rows(gko::array<gko::int64>(exec, {2, 3, 5}));
        config.get_list()["skip_sorting"] = pnode{true};
        param.with_skip_sorting(true);
    }

    template <typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        GKO_ASSERT_ARRAY_EQ(res_param.coarse_rows, ans_param.coarse_rows);
        ASSERT_EQ(res_param.skip_sorting, ans_param.skip_sorting);
    }
};


template <typename T>
class MultigridLevel : public ::testing::Test {
protected:
    using Config = T;

    MultigridLevel()
        : exec(gko::ReferenceExecutor::create()),
          td("double", "int"),
          reg(generate_config_map())
    {}

    std::shared_ptr<const gko::Executor> exec;
    type_descriptor td;
    registry reg;
};


using MultigridLevelTypes = ::testing::Types<::Pgm, ::FixedCoarsening>;


TYPED_TEST_SUITE(MultigridLevel, MultigridLevelTypes, TypenameNameGenerator);


TYPED_TEST(MultigridLevel, CreateDefault)
{
    using Config = typename TestFixture::Config;
    auto config = Config::setup_base();

    auto res = build_from_config(config, this->reg, this->exec, this->td);
    auto ans = Config::default_type::build().on(this->exec);

    Config::validate(res.get(), ans.get());
}


TYPED_TEST(MultigridLevel, ExplicitTemplate)
{
    using Config = typename TestFixture::Config;
    auto config = Config::setup_base();
    Config::change_template(config);

    auto res = build_from_config(config, this->reg, this->exec, this->td);
    auto ans = Config::explicit_type::build().on(this->exec);

    Config::validate(res.get(), ans.get());
}


TYPED_TEST(MultigridLevel, Set)
{
    using Config = typename TestFixture::Config;
    auto config = Config::setup_base();
    Config::change_template(config);
    auto param = Config::explicit_type::build();
    Config::set(config, param, this->reg, this->exec);

    auto res = build_from_config(config, this->reg, this->exec, this->td);
    auto ans = param.on(this->exec);

    Config::validate(res.get(), ans.get());
}


using DummyMgLevel = gko::multigrid::Pgm<double, int>;
using DummySmoother = gko::solver::Ir<double>;
using DummyStop = gko::stop::Iteration;

struct MultigridConfig {
    static pnode setup_base() { return pnode{{{"Type", pnode{"Multigrid"}}}}; }

    template <bool from_reg, typename ParamType>
    static void set(pnode& config, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config.get_list()["post_uses_pre"] = pnode{false};
        param.with_post_uses_pre(false);
        config.get_list()["mid_case"] = pnode{"both"};
        param.with_mid_case(gko::solver::multigrid::mid_smooth_type::both);
        config.get_list()["max_levels"] = pnode{20u};
        param.with_max_levels(20u);
        config.get_list()["min_coarse_rows"] = pnode{32u};
        param.with_min_coarse_rows(32u);
        config.get_list()["cycle"] = pnode{"w"};
        param.with_cycle(gko::solver::multigrid::cycle::w);
        config.get_list()["kcycle_base"] = pnode{2u};
        param.with_kcycle_base(2u);
        config.get_list()["kcycle_rel_tol"] = pnode{0.5};
        param.with_kcycle_rel_tol(0.5);
        config.get_list()["smoother_relax"] = pnode{0.3};
        param.with_smoother_relax(0.3);
        config.get_list()["smoother_iters"] = pnode{2u};
        param.with_smoother_iters(2u);
        config.get_list()["default_initial_guess"] = pnode{"provided"};
        param.with_default_initial_guess(
            gko::solver::initial_guess_mode::provided);
        if (from_reg) {
            config.get_list()["criteria"] = pnode{"criterion_factory"};
            param.with_criteria(reg.search_data<gko::stop::CriterionFactory>(
                "criterion_factory"));
            config.get_list()["mg_level"] =
                pnode{std::vector<pnode>{{"mg_level_0"}, {"mg_level_1"}}};
            param.with_mg_level(
                reg.search_data<gko::LinOpFactory>("mg_level_0"),
                reg.search_data<gko::LinOpFactory>("mg_level_1"));
            config.get_list()["pre_smoother"] = pnode{"pre_smoother"};
            param.with_pre_smoother(
                reg.search_data<gko::LinOpFactory>("pre_smoother"));
            config.get_list()["post_smoother"] = pnode{"post_smoother"};
            param.with_post_smoother(
                reg.search_data<gko::LinOpFactory>("post_smoother"));
            config.get_list()["mid_smoother"] = pnode{"mid_smoother"};
            param.with_mid_smoother(
                reg.search_data<gko::LinOpFactory>("mid_smoother"));
            config.get_list()["coarsest_solver"] = pnode{"coarsest_solver"};
            param.with_coarsest_solver(
                reg.search_data<gko::LinOpFactory>("coarsest_solver"));
        } else {
            config.get_list()["criteria"] =
                pnode{{{"Type", pnode{"Iteration"}}}};
            param.with_criteria(DummyStop::build().on(exec));
            config.get_list()["mg_level"] = pnode{std::vector<pnode>{
                pnode{std::map<std::string, pnode>{{"Type", {"Pgm"}}}},
                pnode{std::map<std::string, pnode>{{"Type", {"Pgm"}}}}}};
            param.with_mg_level(DummyMgLevel::build().on(exec),
                                DummyMgLevel::build().on(exec));
            config.get_list()["pre_smoother"] = pnode{{{"Type", pnode{"Ir"}}}};
            param.with_pre_smoother(DummySmoother::build().on(exec));
            config.get_list()["post_smoother"] = pnode{{{"Type", pnode{"Ir"}}}};
            param.with_post_smoother(DummySmoother::build().on(exec));
            config.get_list()["mid_smoother"] = pnode{{{"Type", pnode{"Ir"}}}};
            param.with_mid_smoother(DummySmoother::build().on(exec));
            config.get_list()["coarsest_solver"] =
                pnode{{{"Type", pnode{"Ir"}}}};
            param.with_coarsest_solver(DummySmoother::build().on(exec));
        }
        config.get_list()["level_selector"] = pnode{"first_for_top"};
        param.with_level_selector(
            [](const gko::size_type level, const gko::LinOp*) {
                return level == 0 ? 0 : 1;
            });
        config.get_list()["solver_selector"] = pnode{"first_for_top"};
        param.with_solver_selector(
            [](const gko::size_type level, const gko::LinOp*) {
                return level == 0 ? 0 : 1;
            });
    }

    template <bool from_reg, typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        ASSERT_EQ(res_param.post_uses_pre, ans_param.post_uses_pre);
        ASSERT_EQ(res_param.mid_case, ans_param.mid_case);
        ASSERT_EQ(res_param.max_levels, ans_param.max_levels);
        ASSERT_EQ(res_param.min_coarse_rows, ans_param.min_coarse_rows);
        ASSERT_EQ(res_param.cycle, ans_param.cycle);
        ASSERT_EQ(res_param.kcycle_base, ans_param.kcycle_base);
        ASSERT_EQ(res_param.kcycle_rel_tol, ans_param.kcycle_rel_tol);
        ASSERT_EQ(res_param.smoother_relax, ans_param.smoother_relax);
        ASSERT_EQ(res_param.smoother_iters, ans_param.smoother_iters);
        ASSERT_EQ(res_param.default_initial_guess,
                  ans_param.default_initial_guess);
        if (from_reg) {
            ASSERT_EQ(res_param.criteria, ans_param.criteria);
            ASSERT_EQ(res_param.mg_level, ans_param.mg_level);
            ASSERT_EQ(res_param.pre_smoother, ans_param.pre_smoother);
            ASSERT_EQ(res_param.post_smoother, ans_param.post_smoother);
            ASSERT_EQ(res_param.mid_smoother, ans_param.mid_smoother);
            ASSERT_EQ(res_param.coarsest_solver, ans_param.coarsest_solver);
        } else {
            ASSERT_NE(
                std::dynamic_pointer_cast<const typename DummyStop::Factory>(
                    res_param.criteria.at(0)),
                nullptr);
            ASSERT_NE(
                std::dynamic_pointer_cast<const typename DummyMgLevel::Factory>(
                    res_param.mg_level.at(0)),
                nullptr);
            ASSERT_NE(
                std::dynamic_pointer_cast<const typename DummyMgLevel::Factory>(
                    res_param.mg_level.at(1)),
                nullptr);
            ASSERT_NE(std::dynamic_pointer_cast<
                          const typename DummySmoother::Factory>(
                          res_param.pre_smoother.at(0)),
                      nullptr);
            ASSERT_NE(std::dynamic_pointer_cast<
                          const typename DummySmoother::Factory>(
                          res_param.post_smoother.at(0)),
                      nullptr);
            ASSERT_NE(std::dynamic_pointer_cast<
                          const typename DummySmoother::Factory>(
                          res_param.mid_smoother.at(0)),
                      nullptr);
            ASSERT_NE(std::dynamic_pointer_cast<
                          const typename DummySmoother::Factory>(
                          res_param.coarsest_solver.at(0)),
                      nullptr);
        }
        if (ans_param.level_selector) {
            ASSERT_TRUE(res_param.level_selector);
            for (gko::size_type i = 0; i < 5; i++) {
                ASSERT_EQ(res_param.level_selector(i, nullptr),
                          ans_param.level_selector(i, nullptr));
            }
        }
        if (ans_param.solver_selector) {
            ASSERT_TRUE(res_param.solver_selector);
            for (gko::size_type i = 0; i < 5; i++) {
                ASSERT_EQ(res_param.solver_selector(i, nullptr),
                          ans_param.solver_selector(i, nullptr));
            }
        }
    }
};


class MultigridT : public ::testing::Test {
protected:
    using Config = MultigridConfig;

    MultigridT()
        : exec(gko::ReferenceExecutor::create()),
          td("double", "int"),
          mg_level_0(DummyMgLevel::build().on(exec)),
          mg_level_1(DummyMgLevel::build().on(exec)),
          criterion_factory(DummyStop::build().on(exec)),
          pre_smoother(DummySmoother::build().on(exec)),
          post_smoother(DummySmoother::build().on(exec)),
          mid_smoother(DummySmoother::build().on(exec)),
          coarsest_solver(DummySmoother::build().on(exec)),
          reg(generate_config_map())
    {
        reg.emplace("mg_level_0", mg_level_0);
        reg.emplace("mg_level_1", mg_level_1);
        reg.emplace("criterion_factory", criterion_factory);
        reg.emplace("pre_smoother", pre_smoother);
        reg.emplace("post_smoother", post_smoother);
        reg.emplace("mid_smoother", mid_smoother);
        reg.emplace("coarsest_solver", coarsest_solver);
    }

    std::shared_ptr<const gko::Executor> exec;
    type_descriptor td;
    std::shared_ptr<typename DummyMgLevel::Factory> mg_level_0;
    std::shared_ptr<typename DummyMgLevel::Factory> mg_level_1;
    std::shared_ptr<typename DummyStop::Factory> criterion_factory;
    std::shared_ptr<typename DummySmoother::Factory> pre_smoother;
    std::shared_ptr<typename DummySmoother::Factory> post_smoother;
    std::shared_ptr<typename DummySmoother::Factory> mid_smoother;
    std::shared_ptr<typename DummySmoother::Factory> coarsest_solver;
    registry reg;
};


TEST_F(MultigridT, CreateDefault)
{
    auto config = Config::setup_base();

    auto res = build_from_config(config, this->reg, this->exec, this->td);
    auto ans = gko::solver::Multigrid::build().on(this->exec);

    Config::template validate<true>(res.get(), ans.get());
}


TEST_F(MultigridT, SetFromRegistry)
{
    auto config = Config::setup_base();
    auto param = gko::solver::Multigrid::build();
    Config::template set<true>(config, param, this->reg, this->exec);

    auto res = build_from_config(config, this->reg, this->exec, this->td);
    auto ans = param.on(this->exec);

    Config::template validate<true>(res.get(), ans.get());
}


TEST_F(MultigridT, SetFromConfig)
{
    auto config = Config::setup_base();
    auto param = gko::solver::Multigrid::build();
    Config::template set<false>(config, param, this->reg, this->exec);

    auto res = build_from_config(config, this->reg, this->exec, this->td);
    auto ans = param.on(this->exec);

    Config::template validate<false>(res.get(), ans.get());
}
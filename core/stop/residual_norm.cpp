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

#include <ginkgo/core/stop/residual_norm.hpp>


#include <ginkgo/core/base/precision_dispatch.hpp>


#include "core/base/dispatch_helper.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/distributed/helpers.hpp"
#include "core/stop/residual_norm_kernels.hpp"


namespace gko {
namespace stop {
namespace residual_norm {
namespace {


GKO_REGISTER_OPERATION(residual_norm, residual_norm::residual_norm);


}  // anonymous namespace
}  // namespace residual_norm


namespace implicit_residual_norm {
namespace {


GKO_REGISTER_OPERATION(implicit_residual_norm,
                       implicit_residual_norm::implicit_residual_norm);


}  // anonymous namespace
}  // namespace implicit_residual_norm


template <typename ValueType>
bool any_is_complex()
{
    return false;
}


template <typename ValueType, typename LinOp, typename... Rest>
bool any_is_complex(const LinOp* in, Rest&&... rest)
{
#if GINKGO_BUILD_MPI
    bool is_complex_distributed = dynamic_cast<const ConvertibleTo<
        experimental::distributed::Vector<std::complex<double>>>*>(in);
#else
    bool is_complex_distributed = false;
#endif

    return is_complex<ValueType>() || is_complex_distributed ||
           dynamic_cast<
               const ConvertibleTo<matrix::Dense<std::complex<double>>>*>(in) ||
           any_is_complex<ValueType>(std::forward<Rest>(rest)...);
}


template <typename ValueType, typename Function, typename... LinOps>
void norm_dispatch(Function&& fn, LinOps*... linops)
{
#if GINKGO_BUILD_MPI
    if (gko::detail::is_distributed(linops...)) {
        if (any_is_complex<ValueType>(linops...)) {
            experimental::distributed::precision_dispatch<
                to_complex<ValueType>>(std::forward<Function>(fn), linops...);
        } else {
            experimental::distributed::precision_dispatch<ValueType>(
                std::forward<Function>(fn), linops...);
        }
    } else
#endif
    {
        if (any_is_complex<ValueType>(linops...)) {
            precision_dispatch<to_complex<ValueType>>(
                std::forward<Function>(fn), linops...);
        } else {
            precision_dispatch<ValueType>(std::forward<Function>(fn),
                                          linops...);
        }
    }
}


template <typename ValueType>
ResidualNormBase<ValueType>::ResidualNormBase(
    std::shared_ptr<const gko::Executor> exec, const CriterionArgs& args,
    remove_complex<ValueType> reduction_factor, mode baseline)
    : EnablePolymorphicObject<ResidualNormBase, Criterion>(exec),
      device_storage_{exec, 2},
      reduction_factor_{reduction_factor},
      baseline_{baseline},
      system_matrix_{args.system_matrix},
      b_{args.b},
      one_{gko::initialize<Vector>({1}, exec)},
      neg_one_{gko::initialize<Vector>({-1}, exec)}
{
    switch (baseline_) {
    case mode::initial_resnorm: {
        if (args.initial_residual == nullptr) {
            if (args.system_matrix == nullptr || args.b == nullptr ||
                args.x == nullptr) {
                GKO_NOT_SUPPORTED(nullptr);
            } else {
                this->starting_tau_ =
                    NormVector::create(exec, dim<2>{1, args.b->get_size()[1]});
                auto b_clone = share(args.b->clone());
                args.system_matrix->apply(neg_one_.get(), args.x, one_.get(),
                                          b_clone.get());
                norm_dispatch<ValueType>(
                    [&](auto dense_r) {
                        dense_r->compute_norm2(this->starting_tau_.get());
                    },
                    b_clone.get());
            }
        } else {
            this->starting_tau_ = NormVector::create(
                exec, dim<2>{1, args.initial_residual->get_size()[1]});
            norm_dispatch<ValueType>(
                [&](auto dense_r) {
                    dense_r->compute_norm2(this->starting_tau_.get());
                },
                args.initial_residual);
        }
        break;
    }
    case mode::rhs_norm: {
        if (args.b == nullptr) {
            GKO_NOT_SUPPORTED(nullptr);
        }
        this->starting_tau_ =
            NormVector::create(exec, dim<2>{1, args.b->get_size()[1]});
        norm_dispatch<ValueType>(
            [&](auto dense_r) {
                dense_r->compute_norm2(this->starting_tau_.get());
            },
            args.b.get());
        break;
    }
    case mode::absolute: {
        if (args.b == nullptr) {
            GKO_NOT_SUPPORTED(nullptr);
        }
        this->starting_tau_ =
            NormVector::create(exec, dim<2>{1, args.b->get_size()[1]});
        this->starting_tau_->fill(gko::one<remove_complex<ValueType>>());
        break;
    }
    default:
        GKO_NOT_SUPPORTED(nullptr);
    }
    this->u_dense_tau_ =
        NormVector::create_with_config_of(this->starting_tau_.get());
}


template <typename ValueType>
bool ResidualNormBase<ValueType>::check_impl(
    uint8 stopping_id, bool set_finalized, array<stopping_status>* stop_status,
    bool* one_changed, const Criterion::Updater& updater)
{
    const NormVector* dense_tau;
    if (updater.residual_norm_ != nullptr) {
        dense_tau = as<NormVector>(updater.residual_norm_);
    } else if (updater.residual_ != nullptr) {
        norm_dispatch<ValueType>(
            [&](auto dense_r) { dense_r->compute_norm2(u_dense_tau_.get()); },
            updater.residual_);
        dense_tau = u_dense_tau_.get();
    } else if (updater.solution_ != nullptr && system_matrix_ != nullptr &&
               b_ != nullptr) {
        auto exec = this->get_executor();
        norm_dispatch<ValueType>(
            [&](auto dense_b, auto dense_x) {
                auto dense_r = dense_b->clone();
                system_matrix_->apply(neg_one_.get(), dense_x, one_.get(),
                                      dense_r.get());
                dense_r->compute_norm2(u_dense_tau_.get());
            },
            b_.get(), updater.solution_);
        dense_tau = u_dense_tau_.get();
    } else {
        GKO_NOT_SUPPORTED(nullptr);
    }
    bool all_converged = true;

    this->get_executor()->run(residual_norm::make_residual_norm(
        dense_tau, starting_tau_.get(), reduction_factor_, stopping_id,
        set_finalized, stop_status, &device_storage_, &all_converged,
        one_changed));

    return all_converged;
}


template <typename ValueType>
bool ImplicitResidualNorm<ValueType>::check_impl(
    uint8 stopping_id, bool set_finalized, array<stopping_status>* stop_status,
    bool* one_changed, const Criterion::Updater& updater)
{
    const Vector* dense_tau;
    if (updater.implicit_sq_residual_norm_ != nullptr) {
        dense_tau = as<Vector>(updater.implicit_sq_residual_norm_);
    } else {
        GKO_NOT_SUPPORTED(nullptr);
    }
    bool all_converged = true;

    this->get_executor()->run(
        implicit_residual_norm::make_implicit_residual_norm(
            dense_tau, this->starting_tau_.get(), this->reduction_factor_,
            stopping_id, set_finalized, stop_status, &this->device_storage_,
            &all_converged, one_changed));

    return all_converged;
}


template <typename ValueType>
AutomaticResidualNorm<ValueType>::AutomaticResidualNorm(
    std::shared_ptr<const gko::Executor> exec)
    : EnablePolymorphicObject<AutomaticResidualNorm<ValueType>, Criterion>(exec)
{}


template <typename ValueType>
AutomaticResidualNorm<ValueType>::AutomaticResidualNorm(
    const AutomaticResidualNorm::Factory* factory, const CriterionArgs& args)
    : EnablePolymorphicObject<AutomaticResidualNorm<ValueType>, Criterion>(
          factory->get_executor()),
      parameters_{factory->get_parameters()}
{
    switch (args.check) {
    case residual_norm_criteria::direct:
        actual_criteria_ =
            ResidualNorm<ValueType>::build()
                .with_reduction_factor(parameters_.reduction_factor)
                .with_baseline(parameters_.baseline)
                .on(this->get_executor())
                ->generate(args);
        break;
    case residual_norm_criteria::implicit:
        actual_criteria_ =
            ImplicitResidualNorm<ValueType>::build()
                .with_reduction_factor(parameters_.reduction_factor)
                .with_baseline(parameters_.baseline)
                .on(this->get_executor())
                ->generate(args);
        break;
    default:
        GKO_NOT_SUPPORTED(args.check);
    }
}


template <typename ValueType>
bool AutomaticResidualNorm<ValueType>::check_impl(
    uint8 stoppingId, bool setFinalized, array<stopping_status>* stop_status,
    bool* one_changed, const Criterion::Updater& updater)
{
    return actual_criteria_->check_impl(stoppingId, setFinalized, stop_status,
                                        one_changed, updater);
}


#define GKO_DECLARE_RESIDUAL_NORM(_type) class ResidualNormBase<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_RESIDUAL_NORM);


#define GKO_DECLARE_IMPLICIT_RESIDUAL_NORM(_type) \
    class ImplicitResidualNorm<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IMPLICIT_RESIDUAL_NORM);

#define GKO_DECLARE_AUTOMATIC_RESIDUAL_NORM(_type) \
    class AutomaticResidualNorm<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_AUTOMATIC_RESIDUAL_NORM);


}  // namespace stop
}  // namespace gko

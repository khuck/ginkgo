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

#ifndef GKO_PUBLIC_CORE_SOLVER_CG_HPP_
#define GKO_PUBLIC_CORE_SOLVER_CG_HPP_


#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace solver {


/**
 * CG or the conjugate gradient method is an iterative type Krylov subspace
 * method which is suitable for symmetric positive definite methods.
 *
 * Though this method performs very well for symmetric positive definite
 * matrices, it is in general not suitable for general matrices.
 *
 * The implementation in Ginkgo makes use of the merged kernel to make the best
 * use of data locality. The inner operations in one iteration of CG are merged
 * into 2 separate steps.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Cg : public EnableLinOp<Cg<ValueType>>,
           public EnablePreconditionedIterativeSolver<ValueType, Cg<ValueType>>,
           public Transposable {
    friend class EnableLinOp<Cg>;
    friend class EnablePolymorphicObject<Cg, LinOp>;

public:
    using value_type = ValueType;
    using transposed_type = Cg<ValueType>;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    /**
     * Return true as iterative solvers use the data in x as an initial guess.
     *
     * @return true as iterative solvers use the data in x as an initial guess.
     */
    bool apply_uses_initial_guess() const override { return true; }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Criterion factories.
         */
        std::vector<std::shared_ptr<const stop::CriterionFactory>>
            GKO_FACTORY_PARAMETER_VECTOR(criteria, nullptr);

        /**
         * Preconditioner factory.
         */
        std::shared_ptr<const LinOpFactory> GKO_FACTORY_PARAMETER_SCALAR(
            preconditioner, nullptr);

        /**
         * Already generated preconditioner. If one is provided, the factory
         * `preconditioner` will be ignored.
         */
        std::shared_ptr<const LinOp> GKO_FACTORY_PARAMETER_SCALAR(
            generated_preconditioner, nullptr);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Cg, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    static std::shared_ptr<LinOpFactory> configure(const property_tree& pt,
                                                   const context& ctx,
                                                   const type_config& cfg)
    {}

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override;

    template <typename VectorType>
    void apply_dense_impl(const VectorType* b, VectorType* x) const;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    explicit Cg(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Cg>(std::move(exec))
    {}

    explicit Cg(const Factory* factory,
                std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Cg>(factory->get_executor(),
                          gko::transpose(system_matrix->get_size())),
          EnablePreconditionedIterativeSolver<ValueType, Cg<ValueType>>{
              std::move(system_matrix), factory->get_parameters()},
          parameters_{factory->get_parameters()}
    {}
};


template <typename ValueType>
struct workspace_traits<Cg<ValueType>> {
    using Solver = Cg<ValueType>;
    // number of vectors used by this workspace
    static int num_vectors(const Solver&);
    // number of arrays used by this workspace
    static int num_arrays(const Solver&);
    // array containing the num_vectors names for the workspace vectors
    static std::vector<std::string> op_names(const Solver&);
    // array containing the num_arrays names for the workspace vectors
    static std::vector<std::string> array_names(const Solver&);
    // array containing all varying scalar vectors (independent of problem size)
    static std::vector<int> scalars(const Solver&);
    // array containing all varying vectors (dependent on problem size)
    static std::vector<int> vectors(const Solver&);

    // residual vector
    constexpr static int r = 0;
    // preconditioned residual vector
    constexpr static int z = 1;
    // p vector
    constexpr static int p = 2;
    // q vector
    constexpr static int q = 3;
    // alpha scalar
    constexpr static int alpha = 4;
    // beta scalar
    constexpr static int beta = 5;
    // previous rho scalar
    constexpr static int prev_rho = 6;
    // current rho scalar
    constexpr static int rho = 7;
    // constant 1.0 scalar
    constexpr static int one = 8;
    // constant -1.0 scalar
    constexpr static int minus_one = 9;

    // stopping status array
    constexpr static int stop = 0;
    // reduction tmp array
    constexpr static int tmp = 1;
};


}  // namespace solver


namespace config {

struct Cg {
    static std::shared_ptr<LinOpFactory> configure(const property_tree& pt,
                                                   const context& ctx,
                                                   const type_config& cfg)
    {
        return dispatch<solver::Cg>(pt, ctx, cfg, cfg.value_type);
    }
};

}  // namespace config
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_CG_HPP_

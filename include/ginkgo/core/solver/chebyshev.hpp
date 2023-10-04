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

#ifndef GKO_PUBLIC_CORE_SOLVER_CHEBYSHEV_HPP_
#define GKO_PUBLIC_CORE_SOLVER_CHEBYSHEV_HPP_


#include <vector>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>
#include <ginkgo/core/stop/iteration.hpp>


namespace gko {
namespace solver {


/**
 * Chebyshev iteration is an iterative method that uses another inner
 * solver to approximate the error of the current solution via the current
 * residual. It has another term for the difference of solution. Moreover, this
 * method requires knowledge about the spectrum of the matrix. This
 * implementation follows the algorithm in "Templates for the Solution of Linear
 * Systems: Building Blocks for Iterative Methods, 2nd Edition".
 *
 * ```
 * solution = initial_guess
 * while not converged:
 *     residual = b - A solution
 *     error = preconditioner(A) * residual
 *     solution = solution + alpha_i * error + beta_i * (solution_i -
 * solution_{i-1})
 * ```
 *
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Chebyshev
    : public EnableLinOp<Chebyshev<ValueType>>,
      public EnablePreconditionedIterativeSolver<ValueType,
                                                 Chebyshev<ValueType>>,
      public EnableApplyWithInitialGuess<Chebyshev<ValueType>>,
      public Transposable {
    friend class EnableLinOp<Chebyshev>;
    friend class EnablePolymorphicObject<Chebyshev, LinOp>;
    friend class EnableApplyWithInitialGuess<Chebyshev>;

public:
    using value_type = ValueType;
    using transposed_type = Chebyshev<ValueType>;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    /**
     * Return true as iterative solvers use the data in x as an initial guess.
     *
     * @return true as iterative solvers use the data in x as an initial guess.
     */
    bool apply_uses_initial_guess() const override
    {
        return this->get_default_initial_guess() ==
               initial_guess_mode::provided;
    }

    /**
     * Copy-assigns a Chebyshev solver. Preserves the executor, shallow-copies
     * inner solver, stopping criterion and system matrix. If the executors
     * mismatch, clones inner solver, stopping criterion and system matrix onto
     * this executor.
     */
    Chebyshev& operator=(const Chebyshev&);

    /**
     * Move-assigns a Chebyshev solver. Preserves the executor, moves inner
     * solver, stopping criterion and system matrix. If the executors mismatch,
     * clones inner solver, stopping criterion and system matrix onto this
     * executor. The moved-from object is empty (0x0 and nullptr inner solver,
     * stopping criterion and system matrix)
     */
    Chebyshev& operator=(Chebyshev&&);

    /**
     * Copy-constructs an Chebyshev solver. Inherits the executor,
     * shallow-copies inner solver, stopping criterion and system matrix.
     */
    Chebyshev(const Chebyshev&);

    /**
     * Move-constructs an Chebyshev solver. Preserves the executor, moves inner
     * solver, stopping criterion and system matrix. The moved-from object is
     * empty (0x0 and nullptr inner solver, stopping criterion and system
     * matrix)
     */
    Chebyshev(Chebyshev&&);

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Criterion factories.
         */
        std::vector<std::shared_ptr<const stop::CriterionFactory>>
            GKO_FACTORY_PARAMETER_VECTOR(criteria, nullptr);

        /**
         * Preconditioner factory. If not provided this will
         * result in a non-preconditioned Chebyshev iteration.
         */
        std::shared_ptr<const LinOpFactory> GKO_FACTORY_PARAMETER_SCALAR(
            preconditioner, nullptr);

        /**
         * Already generated preconditioner. If one is provided, the factory
         * `preconditioner` will be ignored.
         */
        std::shared_ptr<const LinOp> GKO_FACTORY_PARAMETER_SCALAR(
            generated_preconditioner, nullptr);

        /**
         * The pair of foci of ellipse, which covers the eigenvalues of
         * preconditioned system. It is usually be {lower bound of eigval, upper
         * bound of eigval} of preconditioned real matrices.
         */
        std::pair<value_type, value_type> GKO_FACTORY_PARAMETER_VECTOR(
            foci, value_type{0}, value_type{1});

        /**
         * Default initial guess mode. The available options are under
         * initial_guess_mode.
         */
        initial_guess_mode GKO_FACTORY_PARAMETER_SCALAR(
            default_initial_guess, initial_guess_mode::provided);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Chebyshev, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override;

    template <typename VectorType>
    void apply_dense_impl(const VectorType* b, VectorType* x,
                          initial_guess_mode guess) const;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    void apply_with_initial_guess_impl(const LinOp* b, LinOp* x,
                                       initial_guess_mode guess) const override;

    void apply_with_initial_guess_impl(const LinOp* alpha, const LinOp* b,
                                       const LinOp* beta, LinOp* x,
                                       initial_guess_mode guess) const override;

    void set_relaxation_factor(
        std::shared_ptr<const matrix::Dense<ValueType>> new_factor);

    explicit Chebyshev(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Chebyshev>(std::move(exec))
    {}

    explicit Chebyshev(const Factory* factory,
                       std::shared_ptr<const LinOp> system_matrix);

private:
    std::shared_ptr<const LinOp> solver_{};
    // num_generated_scalar_ is to track the number of generated scalar alpha
    // and beta.
    mutable size_type num_generated_scalar_;
    // num_max_generation_ is the number of keeping the generated scalar in
    // workspace.
    mutable size_type num_max_generation_;
    ValueType center_;
    ValueType foci_direction_;
};


template <typename ValueType>
struct workspace_traits<Chebyshev<ValueType>> {
    using Solver = Chebyshev<ValueType>;
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
    constexpr static int residual = 0;
    // inner solution vector
    constexpr static int inner_solution = 1;
    // update solution
    constexpr static int update_solution = 2;
    // alpha
    constexpr static int alpha = 3;
    // beta
    constexpr static int beta = 4;
    // constant 1.0 scalar
    constexpr static int one = 5;
    // constant -1.0 scalar
    constexpr static int minus_one = 6;

    // stopping status array
    constexpr static int stop = 0;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_CHEBYSHEV_HPP_
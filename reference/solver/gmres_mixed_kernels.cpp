/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include "core/solver/gmres_mixed_kernels.hpp"


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/solver/gmres_mixed.hpp>

#include <iostream>


#include "core/solver/gmres_mixed_accessor.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The GMRES_MIXED solver namespace.
 *
 * @ingroup gmres_mixed
 */
namespace gmres_mixed {


namespace {


template <typename ValueType, typename Accessor3d>
void finish_arnoldi_CGS2(matrix::Dense<ValueType> *next_krylov_basis,
                         Accessor3d krylov_bases,
                         matrix::Dense<ValueType> *hessenberg_iter,
                         matrix::Dense<ValueType> *buffer_iter,
                         matrix::Dense<remove_complex<ValueType>> *arnoldi_norm,
                         size_type iter, const stopping_status *stop_status)
{
    static_assert(
        std::is_same<ValueType,
                     typename Accessor3d::accessor::arithmetic_type>::value,
        "ValueType must match arithmetic_type of accessor!");
    const remove_complex<ValueType> eta = 1.0 / sqrt(2.0);
    for (size_type i = 0; i < next_krylov_basis->get_size()[1]; ++i) {
        arnoldi_norm->at(0, i) = zero<remove_complex<ValueType>>();
        for (size_type j = 0; j < next_krylov_basis->get_size()[0]; ++j) {
            arnoldi_norm->at(0, i) += squared_norm(next_krylov_basis->at(j, i));
        }
        arnoldi_norm->at(0, i) = sqrt(arnoldi_norm->at(0, i)) * eta;
        // arnoldi_norm->at(0, i) = norm(next_krylov_basis)
        if (stop_status[i].has_stopped()) {
            continue;
        }
        for (size_type k = 0; k < iter + 1; ++k) {
            hessenberg_iter->at(k, i) = zero<ValueType>();
            for (size_type j = 0; j < next_krylov_basis->get_size()[0]; ++j) {
                hessenberg_iter->at(k, i) +=
                    next_krylov_basis->at(j, i) * krylov_bases(k, j, i);
            }
        }
        // for i in 1:iter
        //     hessenberg(iter, i) = next_krylov_basis' * krylov_bases(:, i)
        // end
        for (size_type k = 0; k < iter + 1; ++k) {
            for (size_type j = 0; j < next_krylov_basis->get_size()[0]; ++j) {
                next_krylov_basis->at(j, i) -=
                    hessenberg_iter->at(k, i) * krylov_bases(k, j, i);
            }
        }
        // for i in 1:iter
        //     next_krylov_basis  -= hessenberg(iter, i) * krylov_bases(:, i)
        // end
        arnoldi_norm->at(1, i) = zero<remove_complex<ValueType>>();
        arnoldi_norm->at(2, i) = zero<remove_complex<ValueType>>();
        for (size_type j = 0; j < next_krylov_basis->get_size()[0]; ++j) {
            arnoldi_norm->at(1, i) += squared_norm(next_krylov_basis->at(j, i));
            arnoldi_norm->at(2, i) =
                (arnoldi_norm->at(2, i) >= abs(next_krylov_basis->at(j, i)))
                    ? arnoldi_norm->at(2, i)
                    : abs(next_krylov_basis->at(j, i));
        }
        arnoldi_norm->at(1, i) = sqrt(arnoldi_norm->at(1, i));

        for (size_type l = 1;
             (arnoldi_norm->at(1, i)) < (arnoldi_norm->at(0, i)) && l < 3;
             l++) {
            arnoldi_norm->at(0, i) = eta * arnoldi_norm->at(1, i);
            for (size_type k = 0; k < iter + 1; ++k) {
                buffer_iter->at(k, i) = zero<ValueType>();
                for (size_type j = 0; j < next_krylov_basis->get_size()[0];
                     ++j) {
                    buffer_iter->at(k, i) +=
                        next_krylov_basis->at(j, i) * krylov_bases(k, j, i);
                }
            }
            // for i in 1:iter
            //     buffer(iter, i) = next_krylov_basis' * krylov_bases(:, i)
            // end
            for (size_type k = 0; k < iter + 1; ++k) {
                for (size_type j = 0; j < next_krylov_basis->get_size()[0];
                     ++j) {
                    next_krylov_basis->at(j, i) -=
                        buffer_iter->at(k, i) * krylov_bases(k, j, i);
                }
                hessenberg_iter->at(k, i) += buffer_iter->at(k, i);
            }
            // for i in 1:iter
            //     next_krylov_basis   -= buffer(iter, i) * krylov_bases(:, i)
            //     hessenberg(iter, i) += buffer(iter, i)
            // end
            arnoldi_norm->at(1, i) = zero<remove_complex<ValueType>>();
            arnoldi_norm->at(2, i) = zero<remove_complex<ValueType>>();
            for (size_type j = 0; j < next_krylov_basis->get_size()[0]; ++j) {
                arnoldi_norm->at(1, i) +=
                    squared_norm(next_krylov_basis->at(j, i));
                arnoldi_norm->at(2, i) =
                    (arnoldi_norm->at(2, i) >= abs(next_krylov_basis->at(j, i)))
                        ? arnoldi_norm->at(2, i)
                        : abs(next_krylov_basis->at(j, i));
            }
            arnoldi_norm->at(1, i) = sqrt(arnoldi_norm->at(1, i));
            // nrmN = norm(next_krylov_basis)
        }
        helper_functions_accessor<Accessor3d>::write_scalar(
            krylov_bases, iter + 1, i,
            arnoldi_norm->at(2, i) / arnoldi_norm->at(1, i));
        hessenberg_iter->at(iter + 1, i) = arnoldi_norm->at(1, i);
        // hessenberg(iter, iter + 1) = norm(next_krylov_basis)
        for (size_type j = 0; j < next_krylov_basis->get_size()[0]; ++j) {
            next_krylov_basis->at(j, i) /= hessenberg_iter->at(iter + 1, i);
            krylov_bases(iter + 1, j, i) = next_krylov_basis->at(j, i);
        }
        // next_krylov_basis /= hessenberg(iter, iter + 1)
        // krylov_bases(:, iter + 1) = next_krylov_basis
        // End of arnoldi
    }
}


template <typename ValueType>
void calculate_sin_and_cos(matrix::Dense<ValueType> *givens_sin,
                           matrix::Dense<ValueType> *givens_cos,
                           matrix::Dense<ValueType> *hessenberg_iter,
                           size_type iter, const size_type rhs)
{
    if (hessenberg_iter->at(iter, rhs) == zero<ValueType>()) {
        givens_cos->at(iter, rhs) = zero<ValueType>();
        givens_sin->at(iter, rhs) = one<ValueType>();
    } else {
        auto hypotenuse = sqrt(hessenberg_iter->at(iter, rhs) *
                                   hessenberg_iter->at(iter, rhs) +
                               hessenberg_iter->at(iter + 1, rhs) *
                                   hessenberg_iter->at(iter + 1, rhs));
        givens_cos->at(iter, rhs) =
            abs(hessenberg_iter->at(iter, rhs)) / hypotenuse;
        givens_sin->at(iter, rhs) = givens_cos->at(iter, rhs) *
                                    hessenberg_iter->at(iter + 1, rhs) /
                                    hessenberg_iter->at(iter, rhs);
    }
}


template <typename ValueType>
void givens_rotation(matrix::Dense<ValueType> *next_krylov_basis,
                     matrix::Dense<ValueType> *givens_sin,
                     matrix::Dense<ValueType> *givens_cos,
                     matrix::Dense<ValueType> *hessenberg_iter, size_type iter,
                     const stopping_status *stop_status)
{
    for (size_type i = 0; i < next_krylov_basis->get_size()[1]; ++i) {
        if (stop_status[i].has_stopped()) {
            continue;
        }
        for (size_type j = 0; j < iter; ++j) {
            auto temp = givens_cos->at(j, i) * hessenberg_iter->at(j, i) +
                        givens_sin->at(j, i) * hessenberg_iter->at(j + 1, i);
            hessenberg_iter->at(j + 1, i) =
                -givens_sin->at(j, i) * hessenberg_iter->at(j, i) +
                givens_cos->at(j, i) * hessenberg_iter->at(j + 1, i);
            hessenberg_iter->at(j, i) = temp;
            // temp             =  cos(j)*hessenberg(j) +
            //                     sin(j)*hessenberg(j+1)
            // hessenberg(j+1)  = -sin(j)*hessenberg(j) +
            //                     cos(j)*hessenberg(j+1)
            // hessenberg(j)    =  temp;
        }

        calculate_sin_and_cos(givens_sin, givens_cos, hessenberg_iter, iter, i);

        hessenberg_iter->at(iter, i) =
            givens_cos->at(iter, i) * hessenberg_iter->at(iter, i) +
            givens_sin->at(iter, i) * hessenberg_iter->at(iter + 1, i);
        hessenberg_iter->at(iter + 1, i) = zero<ValueType>();
        // hessenberg(iter)   = cos(iter)*hessenberg(iter) +
        //                      sin(iter)*hessenberg(iter)
        // hessenberg(iter+1) = 0
    }
}


template <typename ValueType>
void calculate_next_residual_norm(
    matrix::Dense<ValueType> *givens_sin, matrix::Dense<ValueType> *givens_cos,
    matrix::Dense<remove_complex<ValueType>> *residual_norm,
    matrix::Dense<ValueType> *residual_norm_collection,
    const matrix::Dense<remove_complex<ValueType>> *b_norm, size_type iter,
    const stopping_status *stop_status)
{
    for (size_type i = 0; i < residual_norm->get_size()[1]; ++i) {
        if (stop_status[i].has_stopped()) {
            continue;
        }
        residual_norm_collection->at(iter + 1, i) =
            -conj(givens_sin->at(iter, i)) *
            residual_norm_collection->at(iter, i);
        residual_norm_collection->at(iter, i) =
            givens_cos->at(iter, i) * residual_norm_collection->at(iter, i);
        residual_norm->at(0, i) =
            abs(residual_norm_collection->at(iter + 1, i));
    }
}


template <typename ValueType>
void solve_upper_triangular(
    const matrix::Dense<ValueType> *residual_norm_collection,
    const matrix::Dense<ValueType> *hessenberg, matrix::Dense<ValueType> *y,
    const size_type *final_iter_nums)
{
    for (size_type k = 0; k < residual_norm_collection->get_size()[1]; ++k) {
        for (int i = final_iter_nums[k] - 1; i >= 0; --i) {
            auto temp = residual_norm_collection->at(i, k);
            for (size_type j = i + 1; j < final_iter_nums[k]; ++j) {
                temp -=
                    hessenberg->at(
                        i, j * residual_norm_collection->get_size()[1] + k) *
                    y->at(j, k);
            }
            y->at(i, k) =
                temp / hessenberg->at(
                           i, i * residual_norm_collection->get_size()[1] + k);
        }
    }
}


template <typename ValueType, typename ConstAccessor3d>
void calculate_qy(ConstAccessor3d krylov_bases,
                  const matrix::Dense<ValueType> *y,
                  matrix::Dense<ValueType> *before_preconditioner,
                  const size_type *final_iter_nums)
{
    static_assert(
        std::is_same<
            ValueType,
            typename ConstAccessor3d::accessor::arithmetic_type>::value,
        "ValueType must match arithmetic_type of accessor!");
    for (size_type k = 0; k < before_preconditioner->get_size()[1]; ++k) {
        for (size_type i = 0; i < before_preconditioner->get_size()[0]; ++i) {
            before_preconditioner->at(i, k) = zero<ValueType>();
            for (size_type j = 0; j < final_iter_nums[k]; ++j) {
                before_preconditioner->at(i, k) +=
                    krylov_bases(j, i, k) * y->at(j, k);
            }
        }
    }
}


}  // namespace


template <typename ValueType>
void initialize_1(std::shared_ptr<const ReferenceExecutor> exec,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<remove_complex<ValueType>> *b_norm,
                  matrix::Dense<ValueType> *residual,
                  matrix::Dense<ValueType> *givens_sin,
                  matrix::Dense<ValueType> *givens_cos,
                  Array<stopping_status> *stop_status, size_type krylov_dim)
{
    for (size_type j = 0; j < b->get_size()[1]; ++j) {
        // Calculate b norm
        b_norm->at(0, j) = zero<remove_complex<ValueType>>();
        for (size_type i = 0; i < b->get_size()[0]; ++i) {
            b_norm->at(0, j) += squared_norm(b->at(i, j));
        }
        b_norm->at(0, j) = sqrt(b_norm->at(0, j));

        for (size_type i = 0; i < b->get_size()[0]; ++i) {
            residual->at(i, j) = b->at(i, j);
        }
        for (size_type i = 0; i < krylov_dim; ++i) {
            givens_sin->at(i, j) = zero<ValueType>();
            givens_cos->at(i, j) = zero<ValueType>();
        }
        stop_status->get_data()[j].reset();
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_GMRES_MIXED_INITIALIZE_1_KERNEL);


template <typename ValueType, typename Accessor3d>
void initialize_2(std::shared_ptr<const ReferenceExecutor> exec,
                  const matrix::Dense<ValueType> *residual,
                  matrix::Dense<remove_complex<ValueType>> *residual_norm,
                  matrix::Dense<ValueType> *residual_norm_collection,
                  matrix::Dense<remove_complex<ValueType>> *arnoldi_norm,
                  Accessor3d krylov_bases,
                  matrix::Dense<ValueType> *next_krylov_basis,
                  Array<size_type> *final_iter_nums, size_type krylov_dim)
{
    static_assert(
        std::is_same<ValueType,
                     typename Accessor3d::accessor::arithmetic_type>::value,
        "ValueType must match arithmetic_type of accessor!");
    for (size_type j = 0; j < residual->get_size()[1]; ++j) {
        // Calculate residual norm
        residual_norm->at(0, j) = zero<remove_complex<ValueType>>();
        arnoldi_norm->at(2, j) = zero<remove_complex<ValueType>>();
        for (size_type i = 0; i < residual->get_size()[0]; ++i) {
            residual_norm->at(0, j) += squared_norm(residual->at(i, j));
            arnoldi_norm->at(2, j) =
                (arnoldi_norm->at(2, j) >= abs(residual->at(i, j)))
                    ? arnoldi_norm->at(2, j)
                    : abs(residual->at(i, j));
        }
        residual_norm->at(0, j) = sqrt(residual_norm->at(0, j));
        helper_functions_accessor<Accessor3d>::write_scalar(
            krylov_bases, {0}, j,
            arnoldi_norm->at(2, j) / residual_norm->at(0, j));

        for (size_type i = 0; i < krylov_dim + 1; ++i) {
            if (i == 0) {
                residual_norm_collection->at(i, j) = residual_norm->at(0, j);
            } else {
                residual_norm_collection->at(i, j) = zero<ValueType>();
            }
        }
        for (size_type i = 0; i < residual->get_size()[0]; ++i) {
            krylov_bases(0, i, j) =
                residual->at(i, j) / residual_norm->at(0, j);
            next_krylov_basis->at(i, j) =
                residual->at(i, j) / residual_norm->at(0, j);
        }
        final_iter_nums->get_data()[j] = 0;
    }

    for (size_type k = 1; k < krylov_dim + 1; ++k) {
        for (size_type j = 0; j < residual->get_size()[1]; ++j) {
            helper_functions_accessor<Accessor3d>::write_scalar(
                krylov_bases, k, j, one<remove_complex<ValueType>>());
            for (size_type i = 0; i < residual->get_size()[0]; ++i) {
                krylov_bases(k, i, j) = zero<ValueType>();
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_GMRES_MIXED_TYPE(
    GKO_DECLARE_GMRES_MIXED_INITIALIZE_2_KERNEL);


// template <typename ValueType, typename ValueTypeKrylovBases, bool
// Reorthogonalization, bool MGS_CGS>
template <typename ValueType, typename Accessor3d>
void step_1(std::shared_ptr<const ReferenceExecutor> exec,
            matrix::Dense<ValueType> *next_krylov_basis,
            matrix::Dense<ValueType> *givens_sin,
            matrix::Dense<ValueType> *givens_cos,
            matrix::Dense<remove_complex<ValueType>> *residual_norm,
            matrix::Dense<ValueType> *residual_norm_collection,
            Accessor3d krylov_bases, matrix::Dense<ValueType> *hessenberg_iter,
            matrix::Dense<ValueType> *buffer_iter,
            const matrix::Dense<remove_complex<ValueType>> *b_norm,
            matrix::Dense<remove_complex<ValueType>> *arnoldi_norm,
            size_type iter, Array<size_type> *final_iter_nums,
            const Array<stopping_status> *stop_status,
            Array<stopping_status> *reorth_status, Array<size_type> *num_reorth,
            int *num_reorth_steps, int *num_reorth_vectors)
{
    static_assert(
        std::is_same<ValueType,
                     typename Accessor3d::accessor::arithmetic_type>::value,
        "ValueType must match arithmetic_type of accessor!");
    for (size_type i = 0; i < final_iter_nums->get_num_elems(); ++i) {
        final_iter_nums->get_data()[i] +=
            (1 - stop_status->get_const_data()[i].has_stopped());
    }
    finish_arnoldi_CGS2(next_krylov_basis, krylov_bases, hessenberg_iter,
                        buffer_iter, arnoldi_norm, iter,
                        stop_status->get_const_data());
    givens_rotation(next_krylov_basis, givens_sin, givens_cos, hessenberg_iter,
                    iter, stop_status->get_const_data());
    calculate_next_residual_norm(givens_sin, givens_cos, residual_norm,
                                 residual_norm_collection, b_norm, iter,
                                 stop_status->get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_GMRES_MIXED_TYPE(
    GKO_DECLARE_GMRES_MIXED_STEP_1_KERNEL);


template <typename ValueType, typename ConstAccessor3d>
void step_2(std::shared_ptr<const ReferenceExecutor> exec,
            const matrix::Dense<ValueType> *residual_norm_collection,
            ConstAccessor3d krylov_bases,
            const matrix::Dense<ValueType> *hessenberg,
            matrix::Dense<ValueType> *y,
            matrix::Dense<ValueType> *before_preconditioner,
            const Array<size_type> *final_iter_nums)
{
    solve_upper_triangular(residual_norm_collection, hessenberg, y,
                           final_iter_nums->get_const_data());
    calculate_qy(krylov_bases, y, before_preconditioner,
                 final_iter_nums->get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_GMRES_MIXED_CONST_TYPE(
    GKO_DECLARE_GMRES_MIXED_STEP_2_KERNEL);


}  // namespace gmres_mixed
}  // namespace reference
}  // namespace kernels
}  // namespace gko

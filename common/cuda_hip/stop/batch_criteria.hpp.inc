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


// to include "cuda|hip/matrix/batch_struct.hpp"
// to include "cuda|hip/matrix/batch_dense_kernels.hpp"

//#include <ginkgo/core/stop/batch_stop_enum.hpp>

namespace stop {

/**
 * A stopping criterion for batch solvers that comprises a
 * maximum iteration count as well as relative residual tolerance.
 *
 * At most `batch_config::max_num_rhs` right-hand-side vectors are supported.
 */
template <typename ValueType>
class RelResidualMaxIter {
public:
    using real_type = remove_complex<ValueType>;
    using bitset_type = config::lane_mask_type;
    static constexpr int max_nrhs = config::warp_size;

    /**
     * Set up the stopping criterion and convergence variable.
     *
     * @param num_rhs  The number of right-hand-sides in the linear systems.
     * @param max_iters  Maximum number of iterations allowed.
     * @param converged_bitset  A bit-set representing the state of convergence
     *                          of each RHS: 1 for converged and 0 otherwise.
     *                          It is initialized appropriately here, and must
     *                          be passed to the \ref check_converged function.
     * @param rhs_b_norms  The reference RHS norms.
     */
    __device__ __forceinline__ RelResidualMaxIter(
        const int num_rhs, const int max_iters, const real_type rel_res_tol,
        const real_type* const rhs_b_norms, bitset_type& converge_bitset)
        : nrhs{num_rhs},
          rel_tol{rel_res_tol},
          max_its{max_iters},
          rhs_norms{rhs_b_norms}
    {
        if (nrhs > max_nrhs) {
            printf("Batch stopping criterion: Too many right hand sides!\n");
        }
        // converge_bitset = 0 - (1 << num_rhs);
        converge_bitset = (~static_cast<bitset_type>(0) << num_rhs);
    }

    /**
     * Checks whether the different right hand sides have converged.
     *
     * @param iter  The current iteration count.
     * @param residual_norms  (Optional) current residual norm of each RHS.
     * @param residual  Current residual vectors. Unused if residual_norms
     *                  are provided.
     * @param converged  Bits representing converged (1) or not (0) for each
     *                   RHS. The 'right-most' bit corresponds to the first RHS.
     *
     * @return  True if all RHS have converged, false otherwise.
     */
    __device__ __forceinline__ bool check_converged(
        const int iter, const real_type* const residual_norms,
        const gko::batch_dense::BatchEntry<const ValueType>& residual,
        bitset_type& converged) const
    {
        if (iter >= max_its - 1) {
            return true;
        }

        if (residual_norms) {
            check_norms(residual_norms, converged);
        } else {
            __shared__ real_type norms[batch_config<ValueType>::max_num_rhs];
            compute_norm2<ValueType>(
                residual, gko::batch_dense::BatchEntry<real_type>{
                              norms,
                              static_cast<size_type>(
                                  batch_config<ValueType>::max_num_rhs),
                              1, nrhs});
            __syncthreads();
            check_norms(norms, converged);
        }

        return (converged == all_true);
    }

private:
    int nrhs;
    int max_its;
    const real_type rel_tol;
    const real_type* const rhs_norms;
    static constexpr bitset_type all_true = ~static_cast<bitset_type>(0);

    __device__ __forceinline__ void check_norms(
        const real_type* const res_norms, bitset_type& converged) const
    {
        // All warps in the thread-block do the check redundantly.
        // Only the first nrhs threads of each warp do the check.
        const int i = threadIdx.x % config::warp_size;
        const bitset_type activemask = ~(~static_cast<bitset_type>(0) << nrhs);
        if (i < nrhs) {
            const int ichk = (res_norms[i] / rhs_norms[i] < rel_tol);
            const bitset_type wchk = __ballot_sync(activemask, ichk);
            converged |= wchk;
        }
        // copy the converged value to all threads in the warp
        converged = __shfl_sync(all_true, converged, 0);
    }
};


/**
 * A stopping criterion for batch solvers that comprises a
 * maximum iteration count as well as absolute residual tolerance.
 *
 * At most `batch_config::max_num_rhs` right-hand-side vectors are supported.
 */
template <typename ValueType>
class AbsResidualMaxIter {
public:
    using real_type = remove_complex<ValueType>;
    using bitset_type = config::lane_mask_type;
    static constexpr int max_nrhs = config::warp_size;

    /**
     * Set up the stopping criterion and convergence variable.
     *
     * @param num_rhs  The number of right-hand-sides in the linear systems.
     * @param max_iters  Maximum number of iterations allowed.
     * @param converged_bitset  A bit-set representing the state of convergence
     *                          of each RHS: 1 for converged and 0 otherwise.
     *                          It is initialized appropriately here, and must
     *                          be passed to the \ref check_converged function.
     * @param rhs_b_norms  The reference RHS norms.
     */
    __device__ __forceinline__ AbsResidualMaxIter(
        const int num_rhs, const int max_iters, const real_type abs_res_tol,
        const real_type* const rhs_b_norms, bitset_type& converge_bitset)
        : nrhs{num_rhs},
          abs_tol{abs_res_tol},
          max_its{max_iters},
          rhs_norms{rhs_b_norms}
    {
        if (nrhs > max_nrhs) {
            printf("Batch stopping criterion: Too many right hand sides!\n");
        }
        // converge_bitset = 0 - (1 << num_rhs);
        converge_bitset = (~static_cast<bitset_type>(0) << num_rhs);
    }

    /**
     * Checks whether the different right hand sides have converged.
     *
     * @param iter  The current iteration count.
     * @param residual_norms  (Optional) current residual norm of each RHS.
     * @param residual  Current residual vectors. Unused if residual_norms
     *                  are provided.
     * @param converged  Bits representing converged (1) or not (0) for each
     *                   RHS. The 'right-most' bit corresponds to the first RHS.
     *
     * @return  True if all RHS have converged, false otherwise.
     */
    __device__ __forceinline__ bool check_converged(
        const int iter, const real_type* const residual_norms,
        const gko::batch_dense::BatchEntry<const ValueType>& residual,
        bitset_type& converged) const
    {
        if (iter >= max_its - 1) {
            return true;
        }

        if (residual_norms) {
            check_norms(residual_norms, converged);
        } else {
            __shared__ real_type norms[batch_config<ValueType>::max_num_rhs];
            compute_norm2<ValueType>(
                residual, gko::batch_dense::BatchEntry<real_type>{
                              norms, static_cast<size_type>(nrhs), 1, nrhs});

            __syncthreads();
            check_norms(norms, converged);
        }

        return (converged == all_true);
    }

private:
    int nrhs;
    int max_its;
    const real_type abs_tol;
    const real_type* const rhs_norms;
    static constexpr bitset_type all_true = ~static_cast<bitset_type>(0);

    __device__ __forceinline__ void check_norms(
        const real_type* const res_norms, bitset_type& converged) const
    {
        const int rhs_index = threadIdx.x % config::warp_size;
        const bitset_type activemask = ~(~static_cast<bitset_type>(0) << nrhs);

        if (rhs_index < nrhs) {
            int ichk = 0;

            if (converged &
                (1 << rhs_index)) {  // don't check for convergence again for
                                     // RHSs which have already converged.
                ichk = 1;
            }

            ichk = (res_norms[rhs_index] < abs_tol);

            const bitset_type wchk = __ballot_sync(activemask, ichk);
            converged |= wchk;
        }

        // copy the converged value to all threads in the warp
        converged = __shfl_sync(all_true, converged, 0);
        // note that every warp does all this redundantly
    }
};


template <typename ValueType>
class SimpleRelResidual {
public:
    using real_type = remove_complex<ValueType>;

    /**
     * Set up the stopping criterion and convergence variable.
     *
     * @param max_iters  Maximum number of iterations allowed.
     * @param rel_res_tol  Tolerance on relative residual norm.
     * @param rhs_b_norms  The reference RHS norms.
     */
    __device__ __forceinline__ SimpleRelResidual(
        const real_type rel_res_tol, const real_type* const rhs_b_norms)
        : rel_tol_{rel_res_tol}, rhs_norms_{rhs_b_norms}
    {}

    /**
     * Checks whether the right hand side has converged.
     *
     * @param residual_norms  Current residual norm.
     *
     * @return  True if RHS has converged, false otherwise.
     */
    __device__ __forceinline__ bool check_converged(
        const real_type* const residual_norms) const
    {
        return (residual_norms[0] / rhs_norms_[0] < rel_tol_);
    }

private:
    const real_type rel_tol_;
    const real_type* const rhs_norms_;
};

template <typename ValueType>
class SimpleAbsResidual {
public:
    using real_type = remove_complex<ValueType>;

    /**
     * Set up the stopping criterion and convergence variable.
     *
     * @param max_iters  Maximum number of iterations allowed.
     * @param tol  Tolerance on residual norm.
     */
    __device__ __forceinline__ SimpleAbsResidual(const real_type tol,
                                                 const real_type*)
        : abs_tol_{tol}
    {}

    /**
     * Checks whether the different right hand sides have converged.
     *
     * @param iter  The current iteration count.
     * @param residual_norms  (Optional) current residual norm of each RHS.
     * @param residual  Current residual vectors. Unused if residual_norms
     *                  are provided.
     * @param converged  Bits representing converged (1) or not (0) for each
     *                   RHS. The 'right-most' bit corresponds to the first RHS.
     *
     * @return  True if all RHS have converged, false otherwise.
     */
    __device__ __forceinline__ bool check_converged(
        const real_type* const residual_norms) const
    {
        return (residual_norms[0] < abs_tol_);
    }

private:
    const real_type abs_tol_;
};


}  // namespace stop
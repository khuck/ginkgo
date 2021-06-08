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

for (size_type ibatch = 0; ibatch < nbatch; ibatch++) {
    byte *const shared_space = local_space.get_data();
    ValueType *const r = reinterpret_cast<ValueType *>(shared_space);
    ValueType *const z = r + nrows * nrhs;
    ValueType *const w = z + nrows * nrhs;
    ValueType *const helper = w + nrows * nrhs;
    ValueType *const cs = helper + nrows * nrhs;
    ValueType *const sn = cs + restart * nrhs;
    ValueType *const y = sn + restart * nrhs;
    ValueType *const s = y + restart * nrhs;
    ValueType *const H = s + (restart + 1) * nrhs;  // Hessenberg matrix
    ValueType *const V =
        H + restart * (restart + 1) * nrhs;  // Krylov subspace basis vectors
    ValueType *const prec_work = V + nrows * (restart + 1) * nrhs;

    real_type *const norms_rhs = reinterpret_cast<real_type *>(
        prec_work + PrecType::dynamic_work_size(nrows, a.num_nnz));
    real_type *const norms_res = norms_rhs + nrhs;
    real_type *const norms_tmp = norms_res + nrhs;

    uint32 converged = 0;


    const gko::batch_dense::BatchEntry<const ValueType> left_entry =
        gko::batch::batch_entry(left, ibatch);

    const gko::batch_dense::BatchEntry<const ValueType> right_entry =
        gko::batch::batch_entry(right, ibatch);

    // scale the matrix and rhs
    if (left_entry.values) {
        const typename BatchMatrixType::entry_type A_entry =
            gko::batch::batch_entry(a, ibatch);
        const gko::batch_dense::BatchEntry<ValueType> b_entry =
            gko::batch::batch_entry(b, ibatch);
        batch_scale(left_entry, right_entry, A_entry);
        batch_dense::batch_scale(left_entry, b_entry);
    }

    // const typename BatchMatrixType::entry_type A_entry =
    const auto A_entry =
        gko::batch::batch_entry(gko::batch::to_const(a), ibatch);

    const gko::batch_dense::BatchEntry<const ValueType> b_entry =
        gko::batch::batch_entry(gko::batch::to_const(b), ibatch);

    const gko::batch_dense::BatchEntry<ValueType> x_entry =
        gko::batch::batch_entry(x, ibatch);


    const gko::batch_dense::BatchEntry<ValueType> r_entry{
        // storage:row-major , residual vector corresponding to each rhs is
        // stored as a col. of the matrix
        r, static_cast<size_type>(nrhs), nrows, nrhs};

    const gko::batch_dense::BatchEntry<ValueType> z_entry{
        // storage:row-major ,vector corresponding to each rhs is stored as
        // a col. of the matrix
        z, static_cast<size_type>(nrhs), nrows, nrhs};

    const gko::batch_dense::BatchEntry<ValueType> w_entry{
        // storage:row-major ,vector corresponding to each rhs is stored as
        // a col. of the matrix
        w, static_cast<size_type>(nrhs), nrows, nrhs};

    const gko::batch_dense::BatchEntry<ValueType> helper_entry{
        // storage:row-major ,vector corresponding to each rhs is stored as
        // a col. of the matrix
        helper, static_cast<size_type>(nrhs), nrows, nrhs};

    const gko::batch_dense::BatchEntry<ValueType> cs_entry{
        // storage:row-major ,vector corresponding to each rhs is stored as
        // a col. of the matrix
        cs, static_cast<size_type>(nrhs), restart, nrhs};

    const gko::batch_dense::BatchEntry<ValueType> sn_entry{
        // storage:row-major ,vector corresponding to each rhs is stored as
        // a col. of the matrix
        sn, static_cast<size_type>(nrhs), restart, nrhs};

    const gko::batch_dense::BatchEntry<ValueType> y_entry{
        // storage:row-major ,vector corresponding to each rhs is stored as
        // a col. of the matrix
        y, static_cast<size_type>(nrhs), restart, nrhs};


    const gko::batch_dense::BatchEntry<ValueType> s_entry{
        // storage:row-major ,vector corresponding to each rhs is stored as
        // a col. of the matrix
        s, static_cast<size_type>(nrhs), restart + 1, nrhs};


    const gko::batch_dense::BatchEntry<ValueType> H_entry{
        H, static_cast<size_type>(nrhs * restart), restart + 1, restart * nrhs};
    // storage:row-major ,  entry (i,j) for different RHSs are placed after
    // the other in a row - when drawn on paper, (and the same is true for
    // actual storage as the storage order is row-major) to get entry (i,j)
    // for rhs: rhs_k , H_entry.stride*i + j*nrhs  + rhs_k

    const gko::batch_dense::BatchEntry<ValueType> V_entry{
        V, static_cast<size_type>(nrhs), nrows * (restart + 1), nrhs};
    // storage:row-major order , subspace basis vectors corr. to each rhs
    // are stored in a single col. one after the other-(on paper). And to
    // get vi : that is ith basis vector for each rhs: vi_entry{  &V[i*
    // V_entry.stride * nrows], V_entry.stride , nrows, nrhs}; So if nrhs=1,
    // effectively the cols. are stored contiguously in memory one after the
    // other.


    const gko::batch_dense::BatchEntry<real_type> rhs_norms_entry{
        norms_rhs, static_cast<size_type>(nrhs), 1, nrhs};


    const gko::batch_dense::BatchEntry<real_type> res_norms_entry{
        norms_res, static_cast<size_type>(nrhs), 1, nrhs};


    const gko::batch_dense::BatchEntry<real_type> tmp_norms_entry{
        norms_tmp, static_cast<size_type>(nrhs), 1, nrhs};


    // generate preconditioner
    prec.generate(A_entry, prec_work);

    // initialization
    // compute b norms
    // r = b - A*x
    // z = precond*r
    // compute residual norms
    // initialize V,H,cs,sn with zeroes
    initialize(A_entry, b_entry, gko::batch::to_const(x_entry), r_entry, prec,
               z_entry, cs_entry, sn_entry, V_entry, H_entry, rhs_norms_entry,
               res_norms_entry);

    // stopping criterion object
    StopType stop(nrhs, opts.max_its, opts.residual_tol, rhs_norms_entry.values,
                  converged);

    int outer_iter = -1;
    bool inner_loop_break_flag = false;

    // Note: restart - inner iterations and the outer iteration:  are
    // counted as (restart + 1) number of iterations instead of one.
    while (1) {
        ++outer_iter;

        bool all_converged = stop.check_converged(outer_iter * (restart + 1),
                                                  res_norms_entry.values,
                                                  {NULL, 0, 0, 0}, converged);

        logger.log_iteration(ibatch, outer_iter * (restart + 1),
                             res_norms_entry.values, converged);

        if (all_converged) {
            break;
        }


        // KrylovBasis_0 = z/norm(z)
        // s -> fill with zeroes
        // s(0) = norm(z)
        update_v_naught_and_s(gko::batch::to_const(z_entry), V_entry, s_entry,
                              tmp_norms_entry, converged);

        for (int inner_iter = 0; inner_iter < restart; inner_iter++) {
            // w_temp = A * v_i
            // w = precond * w_temp
            // i = inner_iter
            // for k = 0 to inner_iter
            //     Hessenburg(k,i) =  w' * v_k
            //     w = w - Hessenburg(k,i) * v_k
            // end
            // Hessenburg(i+1, i) = norm(w)
            // KrylovBasis_i+1 = w / Hessenburg(i+1,i)
            arnoldi_method(A_entry, inner_iter, V_entry, H_entry, w_entry,
                           helper_entry, tmp_norms_entry, prec, converged);


            for (int k = 0; k < inner_iter; k++) {
                // temp = cs(k) * Hessenberg( k, inner_iter )  +   sn(k) *
                // Hessenberg(k + 1, inner_iter)
                // Hessenberg(k + 1, inner_iter) = -1 * conj(sn(k)) *
                // Hessenberg( k , inner_iter) + conj(cs(k)) * Hessenberg(k
                // + 1 , inner_iter) Hessenberg(k,inner_iter) = temp
                apply_plane_rotation(

                    &cs_entry.values[k * cs_entry.stride],
                    &sn_entry.values[k * sn_entry.stride], nrhs,
                    &H_entry.values[k * H_entry.stride + inner_iter * nrhs],
                    &H_entry
                         .values[(k + 1) * H_entry.stride + inner_iter * nrhs],
                    converged);
            }

            // compute sine and cos
            generate_plane_rotation(
                &H_entry
                     .values[inner_iter * H_entry.stride + inner_iter * nrhs],
                &H_entry.values[(inner_iter + 1) * H_entry.stride +
                                inner_iter * nrhs],
                nrhs, &cs_entry.values[inner_iter * cs_entry.stride],
                &sn_entry.values[inner_iter * cs_entry.stride], converged);

            // temp = cs(inner_iter) * s(inner_iter)
            // s(inner_iter + 1) = -1 * conj(sn(inner_iter)) * s(inner_iter)
            // s(inner_iter) = temp
            // Hessenberg(inner_iter , inner_iter) = cs(inner_iter) *
            // Hessenberg(inner_iter , inner_iter) + sn(inner_iter) *
            // Hessenberg(inner_iter + 1, inner_iter) Hessenberg(inner_iter
            // + 1, inner_iter) = 0
            apply_plane_rotation(
                &cs_entry.values[inner_iter * cs_entry.stride],
                &sn_entry.values[inner_iter * sn_entry.stride], nrhs,
                &s_entry.values[inner_iter * s_entry.stride],
                &s_entry.values[(inner_iter + 1) * s_entry.stride], converged);

            apply_plane_rotation(
                &cs_entry.values[inner_iter * cs_entry.stride],
                &sn_entry.values[inner_iter * sn_entry.stride], nrhs,
                &H_entry
                     .values[inner_iter * H_entry.stride + inner_iter * nrhs],
                &H_entry.values[(inner_iter + 1) * H_entry.stride +
                                inner_iter * nrhs],
                converged);

            for (int c = 0; c < nrhs; c++) {
                const uint32 conv = converged & (1 << c);

                if (conv) {
                    continue;
                }
                H_entry.values[(inner_iter + 1) * H_entry.stride +
                               inner_iter * nrhs + c] = zero<ValueType>();
            }


            // estimate of residual norms
            // residual = abs(s(inner_iter + 1))
            for (int c = 0; c < res_norms_entry.num_rhs; c++) {
                const uint32 conv = converged & (1 << c);

                if (conv) {
                    continue;
                }

                res_norms_entry.values[c] =
                    abs(s_entry.values[(inner_iter + 1) * s_entry.stride + c]);
            }

            const uint32 converged_prev = converged;

            all_converged = stop.check_converged(
                outer_iter * (restart + 1) + inner_iter + 1,
                res_norms_entry.values, {NULL, 0, 0, 0}, converged);

            const uint32 converged_recent = converged_prev ^ converged;

            // y = Hessenburg(0 : inner_iter,0 : inner_iter) \ s(0 :
            // inner_iter) x = x + KrylovBasis(:, 0 : inner_iter ) * y
            update_x(inner_iter, gko::batch::to_const(H_entry),
                     gko::batch::to_const(s_entry),
                     gko::batch::to_const(V_entry), x_entry, y_entry,
                     ~converged_recent);

            logger.log_iteration(ibatch,
                                 outer_iter * (restart + 1) + inner_iter + 1,
                                 res_norms_entry.values, converged);

            if (all_converged) {
                inner_loop_break_flag = true;
                break;
            }
        }

        if (inner_loop_break_flag == true) {
            break;
        }

        // y = Hessenburg(0:restart - 1,0:restart - 1) \ s(0:restart-1)
        // x = x + KrylovBasis(:,0 : restart - 1) * y
        update_x(restart - 1, gko::batch::to_const(H_entry),
                 gko::batch::to_const(s_entry), gko::batch::to_const(V_entry),
                 x_entry, y_entry, converged);


        // r = b
        batch_dense::copy(b_entry, r_entry, converged);

        // r = r - A*x
        advanced_spmv_kernel(static_cast<ValueType>(-1.0), A_entry,
                             gko::batch::to_const(x_entry),
                             static_cast<ValueType>(1.0), r_entry);
        batch_dense::compute_norm2<ValueType>(gko::batch::to_const(r_entry),
                                              res_norms_entry, converged);


        // z = precond * r
        prec.apply(gko::batch::to_const(r_entry), z_entry);
    }


    if (left_entry.values) {
        batch_dense::batch_scale(right_entry, x_entry);
    }
}
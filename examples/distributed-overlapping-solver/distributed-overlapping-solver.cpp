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

// @sect3{Include files}

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

// Add the C++ iostream header to output information to the console.
#include <iostream>
// Add the STL map header for the executor selection
#include <map>
// Add the string manipulation header to handle strings.
#include <string>


int main(int argc, char* argv[])
{
    // @sect3{Type Definitiions}
    // Define the needed types. In a parallel program we need to differentiate
    // beweeen global and local indices, thus we have two index types.
    using LocalIndexType = gko::int32;
    // The underlying value type.
    using ValueType = double;
    // As vector type we use the following, which implements a subset of @ref
    // gko::matrix::Dense.
    using vec = gko::matrix::Dense<ValueType>;
    using dist_vec = gko::experimental::distributed::Vector<ValueType>;
    // As matrix type we simply use the following type, which can read
    // distributed data and be applied to a distributed vector.
    using mtx = gko::matrix::Csr<ValueType, LocalIndexType>;
    // We can use here the same solver type as you would use in a
    // non-distributed program. Please note that not all solvers support
    // distributed systems at the moment.
    using solver = gko::solver::Cg<ValueType>;

    // @sect3{Initialization and User Input Handling}
    // Since this is an MPI program, we need to initialize and finalize
    // MPI at the begin and end respectively of our program. This can be easily
    // done with the following helper construct that uses RAII to automize the
    // initialization and finalization.
    const gko::experimental::mpi::environment env(argc, argv);

    // Create an MPI communicator wrapper and get the rank.
    const gko::experimental::mpi::communicator comm{MPI_COMM_WORLD};
    const auto rank = comm.rank();
    const bool on_boundary = rank == 0 || rank == comm.size() - 1;
    const int num_boundary_intersections =
        (rank == 0) + (rank == comm.size() - 1);

    // Print the ginkgo version information and help message.
    if (rank == 0) {
        std::cout << gko::version_info::get() << std::endl;
    }
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        if (rank == 0) {
            std::cerr
                << "Usage: " << argv[0]
                << " [executor] [num_grid_points_per_domain] [num_overlap] "
                   "[num_iterations] "
                << std::endl;
        }
        std::exit(-1);
    }

    ValueType t_init = gko::experimental::mpi::get_walltime();

    // User input settings:
    // - The executor, defaults to reference.
    // - The number of grid points, defaults to 100.
    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const auto overlap =
        static_cast<gko::size_type>(argc >= 4 ? std::atoi(argv[3]) : 1);
    const auto interior_grid_dim =
        static_cast<gko::size_type>(argc >= 3 ? std::atoi(argv[2]) : 50);
    const auto grid_dim =
        interior_grid_dim + (overlap - 1) * (2 - num_boundary_intersections);
    const auto num_iters =
        static_cast<gko::size_type>(argc >= 5 ? std::atoi(argv[4]) : 1000);

    // Pick the requested executor.
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [&] {
                 return gko::CudaExecutor::create(
                     gko::experimental::mpi::map_rank_to_device_id(
                         MPI_COMM_WORLD, gko::CudaExecutor::get_num_devices()),
                     gko::ReferenceExecutor::create(), false,
                     gko::allocation_mode::device);
             }},
            {"hip",
             [&] {
                 return gko::HipExecutor::create(
                     gko::experimental::mpi::map_rank_to_device_id(
                         MPI_COMM_WORLD, gko::HipExecutor::get_num_devices()),
                     gko::ReferenceExecutor::create(), true);
             }},
            {"dpcpp",
             [&] {
                 auto ref = gko::ReferenceExecutor::create();
                 if (gko::DpcppExecutor::get_num_devices("gpu") > 0) {
                     return gko::DpcppExecutor::create(
                         gko::experimental::mpi::map_rank_to_device_id(
                             MPI_COMM_WORLD,
                             gko::DpcppExecutor::get_num_devices("gpu")),
                         ref);
                 } else if (gko::DpcppExecutor::get_num_devices("cpu") > 0) {
                     return gko::DpcppExecutor::create(
                         gko::experimental::mpi::map_rank_to_device_id(
                             MPI_COMM_WORLD,
                             gko::DpcppExecutor::get_num_devices("cpu")),
                         ref);
                 } else {
                     throw std::runtime_error("No suitable DPC++ devices");
                 }
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};
    const auto exec = exec_map.at(executor_string)();

    // @sect3{Creating the Distributed Matrix and Vectors}
    // As a first step, we create a partition of the rows. The partition
    // consists of ranges of consecutive rows which are assigned a part-id.
    // These part-ids will be used for the distributed data structures to
    // determine which rows will be stored locally. In this example each rank
    // has (nearly) the same number of rows, so we can use the following
    // specialized constructor. See @ref
    // gko::experimental::distributed::Partition for other modes of creating a
    // partition.
    const auto num_rows = grid_dim;

    // Assemble the matrix using a 3-pt stencil and fill the right-hand-side
    // with a sine value. The distributed matrix supports only constructing an
    // empty matrix of zero size and filling in the values with
    // gko::experimental::distributed::Matrix::read_distributed. Only the data
    // that belongs to the rows by this rank will be assembled.
    gko::matrix_data<ValueType, LocalIndexType> A_data;
    gko::matrix_data<ValueType, LocalIndexType> b_data;
    gko::matrix_data<ValueType, LocalIndexType> x_data;
    A_data.size = {num_rows, num_rows};
    b_data.size = {num_rows, 1};
    x_data.size = {num_rows, 1};
    for (int i = 0; i < grid_dim; i++) {
        if (i > 0) {
            A_data.nonzeros.emplace_back(i, i - 1, -1);
        }
        A_data.nonzeros.emplace_back(i, i, 2);
        if (i < grid_dim - 1) {
            A_data.nonzeros.emplace_back(i, i + 1, -1);
        }
    }
    // u(0) = u(1) = 1
    // values in the interior will be overwritten during the communication
    b_data.nonzeros.emplace_back(0, 0, 1.0);
    b_data.nonzeros.emplace_back(grid_dim - 1, 0, 1.0);

    // Take timings.
    comm.synchronize();
    ValueType t_init_end = gko::experimental::mpi::get_walltime();

    // Read the matrix data, currently this is only supported on CPU executors.
    // This will also set up the communication pattern needed for the
    // distributed matrix-vector multiplication.
    auto A_host = gko::share(mtx::create(exec->get_master()));
    auto x_host = vec::create(exec->get_master());
    auto b_host = vec::create(exec->get_master());
    A_host->read(A_data);
    b_host->read(b_data);
    x_host->read(x_data);
    // After reading, the matrix and vector can be moved to the chosen executor,
    // since the distributed matrix supports SpMV also on devices.
    auto A = gko::share(mtx::create(exec));
    auto x = vec::create(exec);
    auto b = vec::create(exec);
    A->copy_from(A_host.get());
    b->copy_from(b_host.get());
    x->copy_from(x_host.get());

    // Take timings.
    comm.synchronize();
    ValueType t_read_setup_end = gko::experimental::mpi::get_walltime();

    // @sect3{Solve the Distributed System}
    // Generate the solver, this is the same as in the non-distributed case.
    auto Ainv =
        solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(100u).on(exec),
                gko::stop::ResidualNorm<ValueType>::build()
                    .with_baseline(gko::stop::mode::absolute)
                    .with_reduction_factor(1e-4)
                    .on(exec))
            .on(exec)
            ->generate(A);

    // Take timings.
    comm.synchronize();
    ValueType t_solver_generate_end = gko::experimental::mpi::get_walltime();

    auto one = gko::initialize<vec>({1}, exec);
    auto exact_solution = dist_vec ::create(
        exec, comm, vec::create(exec, gko::dim<2>{interior_grid_dim, 1}).get());
    exact_solution->fill(1.0);

    std::vector<int> send_sizes(comm.size());
    std::vector<int> send_offsets(send_sizes.size() + 1);
    std::vector<int> recv_sizes(comm.size());
    std::vector<int> recv_offsets(recv_sizes.size() + 1);
    if (comm.rank() > 0) {
        send_sizes[comm.rank() - 1] = 1;
        recv_sizes[comm.rank() - 1] = 1;
    }
    if (comm.rank() < comm.size() - 1) {
        send_sizes[comm.rank() + 1] = 1;
        recv_sizes[comm.rank() + 1] = 1;
    }
    std::partial_sum(send_sizes.begin(), send_sizes.end(),
                     send_offsets.begin() + 1);
    std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                     recv_offsets.begin() + 1);

    auto send_buffer = vec::create(
        exec, gko::dim<2>{static_cast<gko::size_type>(send_offsets.back()), 1});
    auto recv_buffer = vec::create(
        exec, gko::dim<2>{static_cast<gko::size_type>(recv_offsets.back()), 1});

    gko::array<LocalIndexType> send_idxs(exec->get_master(),
                                         send_offsets.back());
    gko::array<LocalIndexType> recv_idxs(exec->get_master(),
                                         recv_offsets.back());
    {
        int i = 0;
        if (comm.rank() > 0) {
            send_idxs.get_data()[i] = 0;
            recv_idxs.get_data()[i] = 0;
            i++;
        }
        if (comm.rank() < comm.size() - 1) {
            send_idxs.get_data()[i] = grid_dim - 1;
            recv_idxs.get_data()[i] = grid_dim - 1;
            i++;
        }
        send_idxs.set_executor(exec);
        recv_idxs.set_executor(exec);
    }

    auto local_error = vec::create(exec->get_master(), gko::dim<2>{1, 1});
    for (gko::size_type it = 0; it < num_iters; ++it) {
        // exchange boundary data
        // maybe also need to exchange whole overlap?
        x->row_gather(&send_idxs, send_buffer.get());
        comm.all_to_all_v(exec, send_buffer->get_values(), send_sizes.data(),
                          send_offsets.data(), recv_buffer->get_values(),
                          recv_sizes.data(), recv_offsets.data());
        for (gko::size_type i = 0; i < recv_idxs.get_num_elems(); ++i) {
            b->at(recv_idxs.get_data()[i]) = recv_buffer->get_values()[i];
        }

        // inner solve
        Ainv->apply(gko::lend(b), gko::lend(x));
        // depending on partition of unity, might need to combine overlapping
        // dofs

        // compute error
        // need to restrict to owned dofs
        auto interior_x = dist_vec ::create(
            exec, comm,
            x->create_submatrix(
                 {rank == 0 ? 0 : overlap - 1,
                  rank == comm.size() - 1 ? grid_dim : grid_dim - overlap + 1},
                 {0})
                .get());
        auto error = gko::clone(exact_solution);
        error->sub_scaled(one.get(), interior_x.get());
        error->compute_norm2(local_error.get());
        if (rank == 0) {
            std::cout << it << ": " << local_error->at(0) << std::endl;
        }
    }

    // Apply the distributed solver, this is the same as in the non-distributed
    // case.
    Ainv->apply(gko::lend(b), gko::lend(x));

    // Take timings.
    comm.synchronize();
    ValueType t_solver_apply_end = gko::experimental::mpi::get_walltime();

    // Compute the residual, this is done in the same way as in the
    // non-distributed case.
    x_host->copy_from(x.get());
    auto minus_one = gko::initialize<vec>({-1.0}, exec);
    A_host->apply(gko::lend(minus_one), gko::lend(x_host), gko::lend(one),
                  gko::lend(b_host));
    auto res_norm = gko::initialize<vec>({0.0}, exec->get_master());
    b_host->compute_norm2(gko::lend(res_norm));

    // Take timings.
    comm.synchronize();
    ValueType t_end = gko::experimental::mpi::get_walltime();

    // @sect3{Printing Results}
    // Print the achieved residual norm and timings on rank 0.
    if (comm.rank() == 0) {
        // clang-format off
        std::cout << "\nNum rows in matrix: " << num_rows
                  << "\nNum ranks: " << comm.size()
                  << "\nFinal Res norm: " << *res_norm->get_values()
                  << "\nInit time: " << t_init_end - t_init
                  << "\nRead time: " << t_read_setup_end - t_init
                  << "\nSolver generate time: " << t_solver_generate_end - t_read_setup_end
                  << "\nSolver apply time: " << t_solver_apply_end - t_solver_generate_end
                  << "\nTotal time: " << t_end - t_init
                  << std::endl;
        // clang-format on
    }
}

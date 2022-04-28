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


#include <ginkgo/ginkgo.hpp>

#include <iostream>
#include <set>
#include <string>

#include "benchmark/utils/general.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"


DEFINE_int64(
    target_rows, 100,
    "Target number of rows, either in total (strong_scaling == true) or per "
    "process (strong_scaling == false).");
DEFINE_int32(dim, 2, "Dimension of stencil, either 2D or 3D");
DEFINE_bool(restrict, false,
            "If true creates 5/7pt stencil, if false creates 9/27pt stencil.");
DEFINE_bool(graph_comm, false,
            "If true, the matrix will use neighborhood communication.");


/**
 * Generates matrix data for a 2D stencil matrix. If restricted is set to true,
 * creates a 5-pt stencil, if it is false creates a 9-pt stencil. If
 * strong_scaling is set to true, creates the same problem size independent of
 * the number of ranks, if it false the problem size grows with the number of
 * ranks.
 */
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> generate_2d_stencil(
    gko::mpi::communicator comm, const gko::size_type target_local_size,
    bool restricted)
{
    std::array<int, 2> dims{};
    std::array<int, 2> periods{};
    MPI_Dims_create(comm.size(), dims.size(), dims.data());

    MPI_Comm cart_comm;
    MPI_Cart_create(comm.get(), dims.size(), dims.data(), periods.data(), 0,
                    &cart_comm);

    std::array<int, 2> coords{};
    MPI_Cart_coords(cart_comm, comm.rank(), coords.size(), coords.data());
    std::cout << coords[0] << "|" << coords[1] << std::endl;

    const auto dp =
        static_cast<IndexType>(std::ceil(std::pow(target_local_size, 1. / 2.)));
    const auto global_size = static_cast<gko::size_type>(dp * dp * comm.size());
    auto A_data = gko::matrix_data<ValueType, IndexType>(
        gko::dim<2>{global_size, global_size});

    auto flat_idx = [&](const auto ix, auto iy) {
        return ix + (coords[0] * dp) + (iy + coords[1] * dp) * dims[0] * dp;
    };

    for (IndexType i = 0; i < dp; ++i) {
        for (IndexType j = 0; j < dp; ++j) {
            auto row = flat_idx(j, i);
            for (IndexType d_i : {-1, 0, 1}) {
                for (IndexType d_j : {-1, 0, 1}) {
                    if (!restricted || ((d_i == 0 && d_j == 0))) {
                        auto col = flat_idx(j + d_j, i + d_i);
                        if (col >= 0 &&
                            col < static_cast<IndexType>(global_size)) {
                            A_data.nonzeros.emplace_back(row, col,
                                                         gko::one<ValueType>());
                        }
                    }
                }
            }
        }
    }

    return A_data;
}


/**
 * Generates matrix data for a 3D stencil matrix. If restricted is set to true,
 * creates a 7-pt stencil, if it is false creates a 27-pt stencil. If
 * strong_scaling is set to true, creates the same problem size independent of
 * the number of ranks, if it false the problem size grows with the number of
 * ranks.
 */
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> generate_3d_stencil(
    gko::mpi::communicator comm, const gko::size_type target_local_size,
    bool restricted)
{
    std::array<int, 3> dims{};
    std::array<int, 3> periods{};
    MPI_Dims_create(comm.size(), dims.size(), dims.data());

    MPI_Comm cart_comm;
    MPI_Cart_create(comm.get(), dims.size(), dims.data(), periods.data(), 0,
                    &cart_comm);

    std::array<int, 3> coords{};
    MPI_Cart_coords(cart_comm, comm.rank(), coords.size(), coords.data());

    const auto dp =
        static_cast<IndexType>(std::ceil(std::pow(target_local_size, 1. / 3.)));
    const auto global_size = dp * dp * dp * comm.size();
    auto A_data = gko::matrix_data<ValueType, IndexType>(
        gko::dim<2>{global_size, global_size});

    auto flat_idx = [&](const auto ix, auto iy, auto iz) {
        return ix + (coords[0] * dp) + (iy + coords[1] * dp) * dims[0] * dp +
               (iz + coords[2] * dp) * dims[0] * dims[1] * dp * dp;
    };

    for (IndexType i = 0; i < dp; ++i) {
        for (IndexType j = 0; j < dp; ++j) {
            for (IndexType k = 0; k < dp; ++k) {
                auto row = flat_idx(k, j, i);
                for (IndexType d_i : {-1, 0, 1}) {
                    for (IndexType d_j : {-1, 0, 1}) {
                        for (IndexType d_k : {-1, 0, 1}) {
                            if (!restricted || ((d_i == 0 && d_j == 0) ||
                                                (d_i == 0 && d_k == 0) ||
                                                (d_j == 0 && d_k == 0))) {
                                auto col = flat_idx(k + d_k, j + d_j, i + d_i);
                                if (col >= 0 && col < global_size) {
                                    A_data.nonzeros.emplace_back(
                                        row, col, gko::one<ValueType>());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return A_data;
}


template <typename LocalIndexType, typename GlobalIndexType, typename ValueType>
std::unique_ptr<gko::distributed::Partition<LocalIndexType, GlobalIndexType>>
build_part_from_local_rows(
    std::shared_ptr<const gko::Executor> exec, gko::mpi::communicator comm,
    const gko::matrix_data<ValueType, GlobalIndexType>& data)
{
    std::set<GlobalIndexType> global_rows;
    for (const auto& entry : data.nonzeros) {
        global_rows.emplace(entry.row);
    }

    auto local_size = global_rows.size();
    auto global_size = local_size;
    comm.all_reduce(&global_size, 1, MPI_SUM);

    std::vector<GlobalIndexType> local_mapping(global_rows.begin(),
                                               global_rows.end());
    std::vector<GlobalIndexType> all_global_rows(global_size);

    comm.all_gather(local_mapping.data(), local_size, all_global_rows.data(),
                    local_size);

    gko::Array<gko::distributed::comm_index_type> mapping{exec->get_master(),
                                                          global_size};
    for (std::size_t i = 0; i < global_size; ++i) {
        auto row = all_global_rows[i];
        auto part = i / local_size;
        mapping.get_data()[row] = part;
    }

    comm.synchronize();
    return gko::distributed::Partition<
        LocalIndexType, GlobalIndexType>::build_from_mapping(exec, mapping,
                                                             comm.size());
}


int main(int argc, char* argv[])
{
    gko::mpi::environment mpi_env{argc, argv};

    const auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();

    using ValueType = etype;
    using GlobalIndexType = gko::int64;
    using LocalIndexType = GlobalIndexType;
    using dist_mtx =
        gko::distributed::Matrix<ValueType, LocalIndexType, GlobalIndexType>;
    using dist_vec = gko::distributed::Vector<ValueType>;
    using vec = gko::matrix::Dense<ValueType>;

    std::string header =
        "A benchmark for measuring the strong or weak scaling of Ginkgo's "
        "distributed SpMV\n";
    std::string format = "";
    initialize_argument_parsing(&argc, &argv, header, format);
    if (rank == 0) {
        print_general_information("");
    }

    if (FLAGS_repetitions == "auto") {
        if (rank == 0) {
            std::string extra_information =
                "WARNING: repetitions = 'auto' not supported for MPI "
                "benchmarks, setting repetitions to the default value.";
            print_general_information(extra_information);
        }
        FLAGS_repetitions = "10";
    }

    auto exec = executor_factory_mpi.at(FLAGS_executor)(comm);

    const auto num_target_rows = FLAGS_target_rows;
    const auto dim = FLAGS_dim;
    const bool restricted = FLAGS_restrict;

    // Generate matrix data on each rank
    if (rank == 0) {
        std::cout << "Generating stencil matrix..." << std::endl;
    }
    auto A_data = dim == 2 ? generate_2d_stencil<ValueType, GlobalIndexType>(
                                 comm, num_target_rows, restricted)
                           : generate_3d_stencil<ValueType, GlobalIndexType>(
                                 comm, num_target_rows, restricted);
    auto part = build_part_from_local_rows<LocalIndexType, GlobalIndexType>(
        exec, comm, A_data);
    auto global_size = part->get_size();

    // Build global matrix from local matrix data.
    auto h_A = dist_mtx::create(exec->get_master(), comm);
    auto A = dist_mtx::create(exec, comm);
    h_A->read_distributed(A_data, part.get(), part.get());
    A->copy_from(h_A.get());

    // Set up global vectors for the distributed SpMV
    if (rank == 0) {
        std::cout << "Setting up vectors..." << std::endl;
    }
    const auto local_size =
        static_cast<gko::size_type>(part->get_part_size(comm.rank()));
    auto x = dist_vec::create(exec, comm, gko::dim<2>{global_size, 1},
                              gko::dim<2>{local_size, 1});
    x->fill(gko::one<ValueType>());
    auto b = dist_vec::create(exec, comm, gko::dim<2>{global_size, 1},
                              gko::dim<2>{local_size, 1});
    b->fill(gko::one<ValueType>());

    auto timer = get_timer(exec, FLAGS_gpu_timer);
    IterationControl ic{timer};

    // Do a warmup run
    if (rank == 0) {
        std::cout << "Warming up..." << std::endl;
    }
    comm.synchronize();
    for (auto _ : ic.warmup_run()) {
        A->apply(lend(x), lend(b));
    }

    // Do and time the actual benchmark runs
    if (rank == 0) {
        std::cout << "Running benchmark..." << std::endl;
    }
    comm.synchronize();
    for (auto _ : ic.run()) {
        A->apply(lend(x), lend(b));
    }

    if (rank == 0) {
        std::cout << "SIZE: " << part->get_size() << std::endl;
        std::cout << "DURATION: " << ic.compute_average_time() << "s"
                  << std::endl;
        std::cout << "ITERATIONS: " << ic.get_num_repetitions() << std::endl;
    }
}

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

#include <ginkgo/ginkgo.hpp>


#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>


#include "benchmark/solver/solver_common.hpp"
#include "benchmark/utils/general_matrix.hpp"
#include "benchmark/utils/generator.hpp"


int main(int argc, char* argv[])
{
    // Set the default repetitions = 1.
    FLAGS_repetitions = "1";
    FLAGS_min_repetitions = 1;
    std::string header =
        "A benchmark for measuring performance of Ginkgo's solvers.\n";
    std::string format = solver_example_config + R"(
  "optimal":"spmv" can be one of the recognized spmv formats
)";
    std::string additional_json = R"(,"optimal":{"spmv":"csr"})";
    initialize_argument_parsing_matrix(&argc, &argv, header, format,
                                       additional_json);

    std::stringstream ss_rel_res_goal;
    ss_rel_res_goal << std::scientific << FLAGS_rel_res_goal;

    std::string extra_information =
        "Running " + FLAGS_solvers + " with " +
        std::to_string(FLAGS_max_iters) + " iterations and residual goal of " +
        ss_rel_res_goal.str() + "\nThe number of right hand sides is " +
        std::to_string(FLAGS_nrhs);
    print_general_information(extra_information);

    auto exec = get_executor(FLAGS_gpu_timer);

    json test_cases;
    if (!FLAGS_overhead) {
        test_cases = json::parse(get_input_stream());
    } else {
        // Fake test case to run once
        auto overhead_json = std::string() +
                             " [{\"filename\": \"overhead.mtx\", \"optimal\": "
                             "{ \"spmv\": \"csr\"}}]";
        test_cases = json::parse(overhead_json);
    }

    run_test_cases(SolverBenchmark<SolverGenerator>{SolverGenerator{}}, exec,
                   get_timer(exec, FLAGS_gpu_timer), test_cases);

    std::cout << std::setw(4) << test_cases << std::endl;
}

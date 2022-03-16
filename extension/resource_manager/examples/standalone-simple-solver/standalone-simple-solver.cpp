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

// @sect3{Include files}

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <resource_manager/resource_manager.hpp>
// Add the fstream header to read from data from files.
#include <fstream>
// Add the C++ iostream header to output information to the console.
#include <iostream>
// Add the STL map header for the executor selection
#include <map>
// Add the string manipulation header to handle strings.
#include <string>


int main(int argc, char *argv[])
{
    using mtx = gko::matrix::Dense<double>;
    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;

    std::ifstream solver_json("data/solver.json");
    rapidjson::IStreamWrapper solver_in(solver_json);
    rapidjson::Document f_solver;
    f_solver.ParseStream(solver_in);

    std::ifstream A_json("data/A.json");
    rapidjson::IStreamWrapper A_in(A_json);
    rapidjson::Document f_A;
    f_A.ParseStream(A_in);

    auto exec = share(gko::ReferenceExecutor::create());
    auto A = gko::extension::resource_manager::create_from_config<gko::LinOp>(
        f_A, exec);
    auto solver =
        gko::extension::resource_manager::create_from_config<gko::LinOp>(
            f_solver, exec, A);
    std::cout << exec.get() << std::endl;


    auto x = share(gko::read<mtx>(std::ifstream("data/x0.mtx"), exec));
    auto b = share(gko::read<mtx>(std::ifstream("data/b.mtx"), exec));

    std::cout << "Apply:\n";
    solver->apply(lend(b), lend(x));

    std::cout << "Solution (x):\n";
    write(std::cout, lend(x));

    return 0;
}
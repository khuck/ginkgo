// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_BENCHMARK_UTILS_GENERAL_MATRIX_HPP_
#define GKO_BENCHMARK_UTILS_GENERAL_MATRIX_HPP_


#include <ginkgo/ginkgo.hpp>


#include <gflags/gflags.h>


#include "benchmark/utils/general.hpp"


DEFINE_string(input_matrix, "",
              "Filename of a matrix to be used as the single input. Overwrites "
              "the value of the -input flag");


/**
 * @copydoc initialize_argument_parsing
 * @param additional_matrix_file_json  text to be appended to the
 *                                     `{"filename":"..."}` JSON object that
 *                                     will be used as input for the benchmark
 *                                     if the `-input_matrix` flag is used.
 */
void initialize_argument_parsing_matrix(
    int* argc, char** argv[], std::string& header, std::string& format,
    std::string additional_matrix_file_json = "", bool do_print = true)
{
    initialize_argument_parsing(argc, argv, header, format, do_print);
    std::string input_matrix_str{FLAGS_input_matrix};
    if (!input_matrix_str.empty()) {
        if (input_stream) {
            std::cerr
                << "-input and -input_matrix cannot be used simultaneously\n";
            std::exit(1);
        }
        // create JSON for the filename via nlohmann_json to ensure the string
        // is correctly escaped
        auto json_template =
            R"([{"filename":"")" + additional_matrix_file_json + "}]";
        auto doc = json::parse(json_template);
        doc[0]["filename"] = input_matrix_str;
        input_stream = std::make_unique<std::stringstream>(doc.dump());
    }
}


#endif  // GKO_BENCHMARK_UTILS_GENERAL_MATRIX_HPP_

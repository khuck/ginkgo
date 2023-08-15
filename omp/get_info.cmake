# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

ginkgo_print_module_header(${detailed_log} "OpenMP")
ginkgo_print_variable(${detailed_log} "OpenMP_CXX_FLAGS")
ginkgo_print_variable(${detailed_log} "OpenMP_CXX_LIB_NAMES")
ginkgo_print_variable(${detailed_log} "OpenMP_CXX_LIBRARIES")
ginkgo_print_module_footer(${detailed_log} "OMP variables:")
ginkgo_print_variable(${detailed_log} "GINKGO_COMPILER_FLAGS")
ginkgo_print_module_footer(${detailed_log} "OMP environment variables:")
ginkgo_print_env_variable(${detailed_log} "OMP_NUM_THREADS")
ginkgo_print_module_footer(${detailed_log} "")

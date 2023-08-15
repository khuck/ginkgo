# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

ginkgo_print_module_header(${detailed_log} "Core")
ginkgo_print_variable(${detailed_log} "BUILD_SHARED_LIBS")
ginkgo_print_variable(${detailed_log} "CMAKE_C_COMPILER")
ginkgo_print_flags(${detailed_log} "CMAKE_C_FLAGS")
ginkgo_print_variable(${detailed_log} "CMAKE_CXX_COMPILER")
ginkgo_print_flags(${detailed_log} "CMAKE_CXX_FLAGS")
ginkgo_print_variable(${detailed_log} "CMAKE_GENERATOR")
ginkgo_print_module_footer(${detailed_log} "")

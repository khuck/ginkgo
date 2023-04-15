#!/usr/bin/env python3
import test_framework
base_flags = ["blas/distributed/multi_vector_distributed"]
# check that all input modes work:
# parameter
test_framework.compare_output_distributed(base_flags + ["-input", '[{"n": 100}]'],
                                          expected_stdout="multi_vector_distributed.simple.stdout",
                                          expected_stderr="multi_vector_distributed.simple.stderr",
                                          num_procs=3)

# stdin
test_framework.compare_output_distributed(base_flags,
                                          expected_stdout="multi_vector_distributed.simple.stdout",
                                          expected_stderr="multi_vector_distributed.simple.stderr",
                                          stdin='[{"n": 100}]',
                                          num_procs=3)

# file
test_framework.compare_output_distributed(base_flags + ["-input", str(test_framework.sourcepath / "input.blas.json")],
                                          expected_stdout="multi_vector_distributed.simple.stdout",
                                          expected_stderr="multi_vector_distributed.simple.stderr",
                                          stdin='[{"n": 100}]',
                                          num_procs=3)

# profiler annotations
test_framework.compare_output_distributed(base_flags + ["-input", '[{"n": 100}]', '-profile', '-profiler_hook', 'debug'],
                                          expected_stdout="multi_vector_distributed.profile.stdout",
                                          expected_stderr="multi_vector_distributed.profile.stderr",
                                          stdin='[{"n": 100}]',
                                          num_procs=3)

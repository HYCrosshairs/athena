function(add_gpu_concepts_test TEST_NAME)
  cmake_parse_arguments(ARG "" "" "EXTRA_LIBS" ${ARGN})

  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

  athena_add_test(${TEST_NAME} EXTRA_LABELS "gpuConcepts")

  set_tests_properties(
    ${TEST_NAME}
    PROPERTIES TIMEOUT 100 ENVIRONMENT
               "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}")

  string(REPLACE "TEST_" "Test" FILE_NAME ${TEST_NAME})

  add_executable(${TEST_NAME} ${FILE_NAME}.cu)

  target_link_libraries(
    ${TEST_NAME} PRIVATE CppUTest::CppUTest CppUTest::CppUTestExt
                         ${ARG_EXTRA_LIBS})
endfunction()

add_gpu_concepts_test(TEST_LoadVsDirectCompute EXTRA_LIBS HY::CONCEPTS)

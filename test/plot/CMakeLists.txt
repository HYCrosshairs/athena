add_executable(PLOT
    main.cpp
)

target_include_directories(PLOT
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(PLOT
    PRIVATE
        gnuplot_wrapper
        matplot
        algebra_lib
)
add_library(systemDefinitions_lib INTERFACE)

target_include_directories(systemDefinitions_lib
    INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/
)

target_link_libraries(systemDefinitions_lib
    INTERFACE
)

add_library(HY::SystemDefinitions ALIAS systemDefinitions_lib)
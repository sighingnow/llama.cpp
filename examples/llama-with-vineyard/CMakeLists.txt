set(TARGET llama-with-vineyard)
add_executable(${TARGET} llama-with-vineyard.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

find_package(vineyard REQUIRED)
target_include_directories(${TARGET} PRIVATE ${VINEYARD_INCLUDE_DIRS})
target_link_libraries(${TARGET} PRIVATE ${VINEYARD_LIBRARIES})

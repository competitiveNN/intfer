# Host-only, model-free probe for intfer::DraftAsyncProducer. Deliberately a
# plain C++ executable -- NO add_sycl_to_target -- so it builds and runs wherever
# a compiler exists, with no GPU, no Level Zero and no model. The seam it tests
# is host-side, so this is the whole cost of certifying it.
add_executable(draft_async_probe main.cpp)
target_include_directories(draft_async_probe PRIVATE ${CMAKE_SOURCE_DIR}/include)
find_package(Threads REQUIRED)
target_link_libraries(draft_async_probe PRIVATE Threads::Threads)

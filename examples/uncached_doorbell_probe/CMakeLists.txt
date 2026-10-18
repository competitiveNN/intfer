add_executable(uncached_doorbell_probe main.cpp)
add_sycl_to_target(TARGET uncached_doorbell_probe SOURCES main.cpp)

#subdirs(src)

option (BUILD_TESTS "Build NLPGraphTest" ON)

if (BUILD_TESTS)

    set (CMAKE_CXX_FLAGS "-std=c++11 -fexceptions -fpermissive")

    include_directories(${NLPGraph_SOURCE_DIR}/NLPGraph)
    include_directories(${NLPGraph_SOURCE_DIR}/NLPGraph/NLPGraphTests)
    
    include_directories(${NLPGraph_SOURCE_DIR}/install/3rdparty/compute-0.4/include)
    include_directories(${NLPGraph_SOURCE_DIR}/install/3rdparty/libqxx-4.0.1/include)
    include_directories(${NLPGraph_SOURCE_DIR}/install/3rdparty/boost_1_58_0)
    
    link_directories(${NLPGraph_SOURCE_DIR}/NLPGraph/NLPGraphTests)
    
    add_executable (NLPGraphTests calc/levenstein_damerau_test.cpp dao/postgres/model_postgres_test.cpp util/opencl_test.cpp util/resource_pool_test.cpp nlpgraph_tests.cpp)
    
    target_link_libraries (NLPGraphTests NLPGraphLib)
    #target_compile_features(NLPGraphLib PRIVATE cxx_range_for)
    
endif (BUILD_TESTS)

#subdirs(src)

option (BUILD_NLPGRAPH_TESTS "Build NLPGraphTest" ON)

if (BUILD_NLPGRAPH_TESTS)

    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fexceptions -fpermissive")
    
    add_executable (NLPGraphTests calc/levenstein_damerau_test.cpp dao/postgres/model_postgres_test.cpp util/opencl_test.cpp util/resource_pool_test.cpp nlpgraph_tests.cpp)
    
    include_directories(${NLPGraph_SOURCE_DIR}/NLPGraph)
    include_directories(${NLPGraph_SOURCE_DIR}/NLPGraph/NLPGraphTests)
    
    include_directories(${NLPGraph_SOURCE_DIR}/install/3rdparty/compute-0.4/include)
    include_directories(${NLPGraph_SOURCE_DIR}/install/3rdparty/libpqxx-4.0.1/include)
    include_directories(${NLPGraph_SOURCE_DIR}/install/3rdparty/boost_1_58_0)
    
    target_link_libraries (NLPGraphTests NLPGraphLib)
    
endif (BUILD_NLPGRAPH_TESTS)

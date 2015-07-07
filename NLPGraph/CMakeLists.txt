#subdirs(src)

option (BUILD_LIB "Build NLPGraphLib" ON)

if (BUILD_NLPGRAPH)

    set (CMAKE_CXX_FLAGS "-fexceptions -fpermissive -std=c++11")

    include_directories(${NLPGraph_SOURCE_DIR}/NLPGraph)
    
    # boost opencl
    include_directories(${NLPGraph_SOURCE_DIR}/install/3rdparty/compute-0.4/include)
    
    # postgres driver
    include_directories(${NLPGraph_SOURCE_DIR}/install/3rdparty/libqxx-4.0.1/include)
    link_directories(${NLPGraph_SOURCE_DIR}/install/3rdparty/libqxx-4.0.1/lib)
    
    # boost
    include_directories(${NLPGraph_SOURCE_DIR}/install/3rdparty/boost_1_58_0)
    link_directories(${NLPGraph_SOURCE_DIR}/install/3rdparty/boost_1_58_0/stage/lib)
    
    add_library(NLPGraphLib calc/levenstein_damerau.cpp dao/postgres/model_postgres.cpp dao/postgres/symbol_provider_postgres.cpp context.cpp)
    #target_compile_features(NLPGraphLib PRIVATE cxx_range_for)

endif(BUILD_NLPGRAPH)


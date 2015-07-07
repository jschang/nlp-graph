#subdirs(src)

option (BUILD_NLPGRAPH_LIB "Build NLPGraphLib" ON)

if (BUILD_NLPGRAPH_LIB)

    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -fpermissive -std=c++11")
    
    add_library (NLPGraphLib calc/levenstein_damerau.cpp dao/postgres/model_postgres.cpp dao/postgres/symbol_provider_postgres.cpp context.cpp)
    
    include_directories (${NLPGraph_SOURCE_DIR}/NLPGraph)
    
    include_directories (${NLPGraph_SOURCE_DIR}/install/3rdparty/compute-0.4/include)
    include_directories (${NLPGraph_SOURCE_DIR}/install/3rdparty/libpqxx-4.0.1/include)
    include_directories (${NLPGraph_SOURCE_DIR}/install/3rdparty/boost_1_58_0)
    
    link_directories (/opt/intel/opencl-1.2-4.5.0.8/lib64)
    find_library (LIB_OPENCL NAMES OpenCL PATHS /opt/intel/opencl-1.2-4.5.0.8/lib64)
    
    find_library (LIB_PQ NAMES pq)
    
    find_library (LIB_PQXX 
        NAMES pqxx.a
        PATHS ${NLPGraph_SOURCE_DIR}/install/3rdparty/libpqxx-4.0.1/lib) 
        
    find_library (LIB_BOOST_LOG
        NAMES boost_log  
        PATHS ${NLPGraph_SOURCE_DIR}/install/3rdparty/boost_1_58_0/stage/lib)
        
    find_library (LIB_BOOST_LOG_SETUP
        NAMES boost_log_setup
        PATHS ${NLPGraph_SOURCE_DIR}/install/3rdparty/boost_1_58_0/stage/lib)
        
    find_library (LIB_BOOST_UNITTESTS
        NAMES libboost_unit_test_framework.a
        PATHS ${NLPGraph_SOURCE_DIR}/install/3rdparty/boost_1_58_0/stage/lib)
        
    find_library (LIB_BOOST_SYSTEM
        NAMES libboost_system.a
        PATHS ${NLPGraph_SOURCE_DIR}/install/3rdparty/boost_1_58_0/stage/lib)
        
    find_library (LIB_BOOST_THREAD
        NAMES libboost_thread.a
        PATHS ${NLPGraph_SOURCE_DIR}/install/3rdparty/boost_1_58_0/stage/lib)
    
    target_link_libraries (NLPGraphLib ${LIB_OPENCL} ${LIB_PQ} ${LIB_PQXX} ${LIB_BOOST_LOG} ${LIB_BOOST_LOG_SETUP} ${LIB_BOOST_UNITTESTS} ${LIB_BOOST_SYSTEM} ${LIB_BOOST_THREAD})

endif(BUILD_NLPGRAPH_LIB)


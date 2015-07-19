cmake_minimum_required (VERSION 2.6)                 

project (NLPGraph CXX)
subdirs (NLPGraph NLPGraphTests)

option (BUILD_NLPGRAPH_LIB "Build NLPGraphLib" ON)
option (BUILD_NLPGRAPH_TESTS "Build NLPGraphTest" ON)

set (OPENCL_LIB_PATH "/opt/intel/opencl-1.2-4.5.0.8/lib64" PATH "OpenCL library path")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -fpermissive -std=c++11")
    
include_directories (${NLPGraph_SOURCE_DIR}/NLPGraph)

include_directories (${NLPGraph_SOURCE_DIR}/install/3rdparty/compute-0.4/include)
include_directories (${NLPGraph_SOURCE_DIR}/install/3rdparty/libpqxx-4.0.1/include)
include_directories (${NLPGraph_SOURCE_DIR}/install/3rdparty/boost_1_58_0)

find_library (LIB_PTHREAD
    NAMES pthread) 

find_library (LIB_OPENCL 
    NAMES OpenCL 
    PATHS ${OPENCL_LIB_PATH})

find_library (LIB_PQ 
    NAMES pq)
    
find_library (LIB_PQXX 
    NAMES pqxx
    PATHS ${NLPGraph_SOURCE_DIR}/install/3rdparty/libpqxx-4.0.1/lib)
    
find_library (LIB_BOOST_LOG
    NAMES boost_log  
    PATHS ${NLPGraph_SOURCE_DIR}/install/3rdparty/boost_1_58_0/stage/lib)
    
find_library (LIB_BOOST_LOG_SETUP
    NAMES boost_log_setup
    PATHS ${NLPGraph_SOURCE_DIR}/install/3rdparty/boost_1_58_0/stage/lib)
    
find_library (LIB_BOOST_UNITTESTS
    NAMES boost_unit_test_framework
    PATHS ${NLPGraph_SOURCE_DIR}/install/3rdparty/boost_1_58_0/stage/lib)
    
find_library (LIB_BOOST_SYSTEM
    NAMES boost_system
    PATHS ${NLPGraph_SOURCE_DIR}/install/3rdparty/boost_1_58_0/stage/lib)
    
find_library (LIB_BOOST_THREAD
    NAMES boost_thread
    PATHS ${NLPGraph_SOURCE_DIR}/install/3rdparty/boost_1_58_0/stage/lib)

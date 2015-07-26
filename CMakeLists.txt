cmake_minimum_required (VERSION 2.6)                 

project (NLPGraph CXX)
subdirs (NLPGraph NLPGraphTests)

option (BUILD_NLPGRAPH_LIB "Build NLPGraphLib" ON)
option (BUILD_NLPGRAPH_TESTS "Build NLPGraphTest" ON)

set (OPENCL_LIB_PATH "/usr/lib/x86_64-linux-gnu/" PATH "OpenCL library path")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -fpermissive -std=c++11")
    
include_directories (${NLPGraph_SOURCE_DIR}/NLPGraph)

include_directories (${NLPGraph_SOURCE_DIR}/install/3rdparty/compute-0.4/include)
include_directories (${NLPGraph_SOURCE_DIR}/install/3rdparty/libpqxx-4.0.1/include)
include_directories (${NLPGraph_SOURCE_DIR}/install/3rdparty/boost_1_58_0)

find_library (LIB_PTHREAD
    NAMES pthread) 
MESSAGE(STATUS "LIB_PTHREAD:         " ${LIB_PTHREAD})

find_library (LIB_OPENCL 
    NAMES OpenCL 
    PATHS ${OPENCL_LIB_PATH})
MESSAGE(STATUS "LIB_OPENCL:          " ${LIB_OPENCL})

find_library (LIB_PQ 
    NAMES pq)
MESSAGE(STATUS "LIB_PQ:              " ${LIB_PQ})
    
find_library (LIB_PQXX 
    NAMES pqxx
    PATHS ${NLPGraph_SOURCE_DIR}/install/3rdparty/libpqxx-4.0.1/lib)
MESSAGE(STATUS "LIB_PQXX:            " ${LIB_PQXX})
    
find_library (LIB_BOOST_LOG
    NAMES boost_log  
    PATHS ${NLPGraph_SOURCE_DIR}/install/3rdparty/boost_1_58_0/stage/lib)
MESSAGE(STATUS "LIB_BOOST_LOG:       " ${LIB_BOOST_LOG})
    
find_library (LIB_BOOST_LOG_SETUP
    NAMES boost_log_setup
    PATHS ${NLPGraph_SOURCE_DIR}/install/3rdparty/boost_1_58_0/stage/lib)
MESSAGE(STATUS "LIB_BOOST_LOG_SETUP: " ${LIB_BOOST_LOG_SETUP})
    
find_library (LIB_BOOST_UNITTESTS
    NAMES boost_unit_test_framework
    PATHS ${NLPGraph_SOURCE_DIR}/install/3rdparty/boost_1_58_0/stage/lib)
MESSAGE(STATUS "LIB_BOOST_UNITTESTS: " ${LIB_BOOST_UNITTESTS})
    
find_library (LIB_BOOST_SYSTEM
    NAMES boost_system
    PATHS ${NLPGraph_SOURCE_DIR}/install/3rdparty/boost_1_58_0/stage/lib)
MESSAGE(STATUS "LIB_BOOST_SYSTEM:    " ${LIB_BOOST_SYSTEM})
    
find_library (LIB_BOOST_THREAD
    NAMES boost_thread
    PATHS ${NLPGraph_SOURCE_DIR}/install/3rdparty/boost_1_58_0/stage/lib)
MESSAGE(STATUS "LIB_BOOST_THREAD:    " ${LIB_BOOST_THREAD})


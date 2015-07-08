if (BUILD_NLPGRAPH_TESTS)

    project(NLPGraphTests)

    include_directories(${NLPGraph_SOURCE_DIR}/NLPGraph/NLPGraphTests)
    link_directories(${NLPGraph_SOURCE_DIR}/NLPGraph)
    
    add_executable (NLPGraphTests
        nlpgraph_tests.cpp
        calc/levenstein_damerau_test.cpp 
        dao/postgres/model_postgres_test.cpp 
        util/opencl_test.cpp util/resource_pool_test.cpp 
        )
        
    target_link_libraries (NLPGraphTests 
        ${LIB_OPENCL} 
        ${LIB_PQ} 
        ${LIB_PQXX} 
        ${LIB_BOOST_LOG} 
        ${LIB_BOOST_LOG_SETUP} 
        ${LIB_BOOST_UNITTESTS} 
        ${LIB_BOOST_SYSTEM} 
        ${LIB_BOOST_THREAD}
        NLPGraphLib.a)
    
endif (BUILD_NLPGRAPH_TESTS)

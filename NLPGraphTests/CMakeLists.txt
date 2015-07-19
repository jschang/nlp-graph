if (BUILD_NLPGRAPH_TESTS)

    project(NLPGraphTests)

    include_directories(${NLPGraph_SOURCE_DIR}/NLPGraph)
    link_directories(${NLPGraph_SOURCE_DIR}/NLPGraph)
    
    add_executable (NLPGraphTests
        ./calc/levenstein_damerau_test.cpp
        ./dao/postgres/model_postgres_test.cpp
        ./nlpgraph_tests.cpp
        ./nlpgraph_tests.h
        ./util/opencl_test.cpp
        ./util/resource_pool_test.cpp
    )
    
    source_group(src FILES 
        ./nlpgraph_tests.cpp
        ./nlpgraph_tests.h
    )
    source_group(src\\calc FILES 
        ./calc/levenstein_damerau_test.cpp
    )
    source_group(src\\calc FILES 
        ./calc/levenstein_damerau_test.cpp
    )
    source_group(src\\dao\\postgres FILES 
        ./dao/postgres/model_postgres_test.cpp
    )
    source_group(src\\util FILES 
        ./util/opencl_test.cpp
        ./util/resource_pool_test.cpp
    )
        
    target_link_libraries (NLPGraphTests 
        ${LIB_PTHREAD}
        ${LIB_OPENCL}
        ${LIB_PQXX}
        ${LIB_PQ}  
        ${LIB_BOOST_LOG} 
        ${LIB_BOOST_LOG_SETUP} 
        ${LIB_BOOST_UNITTESTS} 
        ${LIB_BOOST_SYSTEM} 
        ${LIB_BOOST_THREAD}
        NLPGraphLib.a
    )
    
endif (BUILD_NLPGRAPH_TESTS)

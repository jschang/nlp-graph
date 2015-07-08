if (BUILD_NLPGRAPH_LIB)
    
    project(NLPGraphLib)

    add_library (NLPGraphLib
        util/time_helper.cpp
        util/file.cpp
        util/opencl.cpp
        calc/levenstein_damerau.cpp 
        dao/postgres/model_postgres.cpp 
        dao/postgres/symbol_provider_postgres.cpp
        context.cpp)
        
    target_link_libraries (NLPGraphLib
        ${LIB_PTHREAD}
        ${LIB_OPENCL}
        ${LIB_PQXX}
        ${LIB_PQ}  
        ${LIB_BOOST_LOG} 
        ${LIB_BOOST_LOG_SETUP} 
        ${LIB_BOOST_UNITTESTS} 
        ${LIB_BOOST_SYSTEM} 
        ${LIB_BOOST_THREAD})

endif(BUILD_NLPGRAPH_LIB)


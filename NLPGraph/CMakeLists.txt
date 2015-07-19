if (BUILD_NLPGRAPH_LIB)
    
    project(NLPGraphLib)

    add_library (NLPGraphLib
        ./calc/levenstein_damerau.cpp
        ./calc/levenstein_damerau.h
        ./context.cpp
        ./context.h
        ./dao/model.h
        ./dao/model_exception.h
        ./dao/postgres/model_postgres.cpp
        ./dao/postgres/model_postgres.h
        ./dao/postgres/symbol_provider_postgres.cpp
        ./dao/postgres/symbol_provider_postgres.h
        ./dao/symbol_provider.h
        ./dto/input_channel.h
        ./dto/recollection.h
        ./dto/recollection_exception.h
        ./dto/symbol.h
        ./nlpgraph.h
        ./util/file.cpp
        ./util/file.h
        ./util/logger.h
        ./util/opencl.cpp
        ./util/opencl.h
        ./util/resource_pool.h
        ./util/time_helper.cpp
        ./util/time_helper.h
    )
    
    source_group(src FILES 
        ./context.cpp
        ./context.h
        ./nlpgraph.h
    )
    source_group(src\\calc FILES 
        ./calc/levenstein_damerau.cpp
        ./calc/levenstein_damerau.h
    )
    source_group(src\\dao FILES
        ./dao/model.h
        ./dao/model_exception.h
        ./dao/symbol_provider.h
    )
    source_group(src\\dao\\postgres FILES
        ./dao/postgres/model_postgres.cpp
        ./dao/postgres/model_postgres.h
        ./dao/postgres/symbol_provider_postgres.cpp
        ./dao/postgres/symbol_provider_postgres.h
    )
    source_group(src\\dto FILES    
        ./dto/input_channel.h
        ./dto/recollection.h
        ./dto/recollection_exception.h
        ./dto/symbol.h
    )
    source_group(src\\util FILES
        ./util/file.cpp
        ./util/file.h
        ./util/logger.h
        ./util/opencl.cpp
        ./util/opencl.h
        ./util/resource_pool.h
        ./util/time_helper.cpp
        ./util/time_helper.h
    )
        
    target_link_libraries (NLPGraphLib
        ${LIB_PTHREAD}
        ${LIB_OPENCL}
        ${LIB_PQXX}
        ${LIB_PQ}  
        ${LIB_BOOST_LOG} 
        ${LIB_BOOST_LOG_SETUP} 
        ${LIB_BOOST_UNITTESTS} 
        ${LIB_BOOST_SYSTEM} 
        ${LIB_BOOST_THREAD}
    )

endif(BUILD_NLPGRAPH_LIB)


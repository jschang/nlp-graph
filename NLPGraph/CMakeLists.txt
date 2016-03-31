if (BUILD_NLPGRAPH_LIB)
    
    project(NLPGraphLib)

    add_library (NLPGraphLib
    
        ./context.cpp
        ./context.h
        ./nlpgraph.h
        
        ./math/generator.h
        ./math/function_1d.h
        
        ./math/function/tanh.h
        ./math/function/tanh.cpp
        ./math/function/logistic.h
        ./math/function/logistic.cpp

        ./calc/kohonen_som.cpp
        ./calc/kohonen_som.h
        ./calc/kohonen_som/kohonen_som_data.cpp
        ./calc/kohonen_som/kohonen_som_data.h
        ./calc/kohonen_som/kohonen_som_sample_data.cpp
        ./calc/kohonen_som/kohonen_som_sample_data.h
        ./calc/kohonen_som/kohonen_som_result.cpp
        ./calc/kohonen_som/kohonen_som_result.h
        
        ./calc/levenstein_damerau.cpp
        ./calc/levenstein_damerau.h
        ./calc/levenstein_damerau/levenstein_damerau_data.cpp
        ./calc/levenstein_damerau/levenstein_damerau_data.h
        ./calc/levenstein_damerau/levenstein_damerau_reconstruct_data.cpp
        ./calc/levenstein_damerau/levenstein_damerau_reconstruct_data.h

        ./calc/smith_waterman.cpp
        ./calc/smith_waterman.h
        ./calc/smith_waterman/smith_waterman_data.cpp
        ./calc/smith_waterman/smith_waterman_data.h
        ./calc/smith_waterman/smith_waterman_util.cl.h
        ./calc/smith_waterman/smith_waterman_cost.cl.h
        ./calc/smith_waterman/smith_waterman_matrices.cl.h
        ./calc/smith_waterman/smith_waterman_dists.cl.h

        ./neural/neuron.cpp
        ./neural/neuron.h
        ./neural/synapse.cpp
        ./neural/synapse.h
        ./neural/network.cpp
        ./neural/network.h
        ./neural/network_runner.cpp
        ./neural/network_runner.h
        ./neural/network_run_data.cpp
        ./neural/network_run_data.h
        ./neural/network_trainer.cpp
        ./neural/network_trainer.h
        ./neural/network_training_data.cpp
        ./neural/network_training_data.h
        ./neural/neural.h
        ./neural/constant.h

        ./dao/model.h
        ./dao/model_exception.h
        ./dao/symbol_provider.h
        
        ./dao/postgres/model_postgres.cpp
        ./dao/postgres/model_postgres.h
        ./dao/postgres/symbol_provider_postgres.cpp
        ./dao/postgres/symbol_provider_postgres.h
        
        ./dto/input_channel.h
        ./dto/recollection.h
        ./dto/recollection_exception.h
        ./dto/symbol.h
        
        ./util/math.cpp
        ./util/math.h
        ./util/string.cpp
        ./util/string.h
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
    source_group(src\\math FILES
        ./math/generator.h
        ./math/function_1d.h
    )
    source_group(src\\math\\function FILES
        ./math/function/tanh.h
        ./math/function/tanh.cpp
        ./math/function/logistic.h
        ./math/function/logistic.cpp
    )
    source_group(src\\calc FILES
        ./calc/kohonen_som.cpp
        ./calc/kohonen_som.h
        ./calc/levenstein_damerau.cpp
        ./calc/levenstein_damerau.h
        ./calc/smith_waterman.cpp
        ./calc/smith_waterman.h
    )
    source_group(src\\calc\\kohonen_som FILES
        ./calc/kohonen_som/kohonen_som_data.cpp
        ./calc/kohonen_som/kohonen_som_data.h
        ./calc/kohonen_som/kohonen_som_sample_data.cpp
        ./calc/kohonen_som/kohonen_som_sample_data.h
        ./calc/kohonen_som/kohonen_som_result.cpp
        ./calc/kohonen_som/kohonen_som_result.h
    )
    source_group(src\\calc\\levenstein_damerau FILES
        ./calc/levenstein_damerau/levenstein_damerau_data.cpp
        ./calc/levenstein_damerau/levenstein_damerau_data.h
        ./calc/levenstein_damerau/levenstein_damerau_reconstruct_data.cpp
        ./calc/levenstein_damerau/levenstein_damerau_reconstruct_data.h
    )
    source_group(src\\calc\\smith_waterman FILES
        ./calc/smith_waterman/smith_waterman_data.cpp
        ./calc/smith_waterman/smith_waterman_data.h
        ./calc/smith_waterman/smith_waterman_util.cl.h
        ./calc/smith_waterman/smith_waterman_cost.cl.h
        ./calc/smith_waterman/smith_waterman_matrices.cl.h
        ./calc/smith_waterman/smith_waterman_dists.cl.h
    )
    source_group(src\\neural FILES
        ./neural/neuron.cpp
        ./neural/neuron.h
        ./neural/synapse.cpp
        ./neural/synapse.h
        ./neural/network.cpp
        ./neural/network.h
        ./neural/constant.h
        ./neural/network_runner.cpp
        ./neural/network_runner.h
        ./neural/network_run_data.cpp
        ./neural/network_run_data.h
        ./neural/network_trainer.cpp
        ./neural/network_trainer.h
        ./neural/network_training_data.cpp
        ./neural/network_training_data.h
        ./neural/neural.h
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
    	./util/math.cpp
        ./util/math.h
    	./util/string.cpp
        ./util/string.h
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


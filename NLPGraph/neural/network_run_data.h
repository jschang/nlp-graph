//
//  network_run_data.hpp
//  NLPGraph
//
//  Created by Jonathan Schang on 11/24/15.
//
//

#ifndef network_run_data_hpp
#define network_run_data_hpp

#import <numeric>
#include "../nlpgraph.h"
#include "../util/logger.h"
#include "../util/opencl.h"

namespace NLPGraph {
    namespace Neural {

class NetworkRunData {
private:
    cl_mem _activations;
};

}}

#endif /* network_run_data_hpp */

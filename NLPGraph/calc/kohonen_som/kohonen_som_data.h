//
//  kohonen_som_data.hpp
//  NLPGraph
//
//  Created by Jonathan Schang on 11/24/15.
//
//

#ifndef kohonen_som_data_hpp
#define kohonen_som_data_hpp

#import <numeric>
#include "../../nlpgraph.h"
#include "../../util/logger.h"
#include "../../util/opencl.h"

namespace NLPGraph {
    namespace Calc {
        
class KohonenSOMData {
private:
    // the vector length of each node's weights
    int _nodeWidth;
    
    // OpenCL data
    cl_mem _clNodeWeights = 0;
    uint64_t _nodeCount = 0;
    cl_mem _clMapDimensions = 0;
    boost::shared_ptr< std::vector<uint32_t> > _mapDimensions;
public:
    KohonenSOMData(const boost::compute::context &context, 
            const std::vector<double> &nodeWeights, // product(mapDimensions) * nodeWidth
            const std::vector<uint32_t> &mapDimensions, 
            const int nodeWidth);
    ~KohonenSOMData();
    void fromClMem(const boost::compute::command_queue &commandQueue, std::vector<double> &weights);
    const int nodeWidth();
    const uint64_t nodeCount();
    const std::vector<uint32_t>* mapDimensions();
    const cl_mem clNodeWeights();
    const cl_mem clMapDimensions();
};

}}

#endif /* kohonen_som_data_hpp */

//
//  kohonen_som_sample_data.hpp
//  NLPGraph
//
//  Created by Jonathan Schang on 11/24/15.
//
//

#ifndef kohonen_som_sample_data_hpp
#define kohonen_som_sample_data_hpp

#import <numeric>
#include "../../nlpgraph.h"
#include "../../util/logger.h"
#include "../../util/opencl.h"

namespace NLPGraph {
    namespace Calc {

class KohonenSOMSampleData {
private:
    cl_mem _clData = 0;
    uint32_t _width = 0;
    uint32_t _count = 0;
public:
    KohonenSOMSampleData(const boost::compute::context &context, 
            const std::vector<double> &sampleData, 
            const uint32_t sampleWidth);
    ~KohonenSOMSampleData();
    cl_mem clData();
    uint width();
    uint count();
};

}}

#endif /* kohonen_som_sample_data_hpp */

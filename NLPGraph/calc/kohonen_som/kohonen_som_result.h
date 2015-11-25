//
//  kohonen_som_result.hpp
//  NLPGraph
//
//  Created by Jonathan Schang on 11/24/15.
//
//

#ifndef kohonen_som_result_hpp
#define kohonen_som_result_hpp

#import <numeric>
#include "../../nlpgraph.h"
#include "../../util/logger.h"
#include "../../util/opencl.h"

namespace NLPGraph {
    namespace Calc {

class KohonenSOMResult {
private:
    boost::shared_ptr<std::vector<std::vector<uint32_t>>> _indexes;
    boost::shared_ptr<std::vector<float>> _distances;
    cl_mem _clDistances = 0;
    cl_mem _clIndexes = 0;
public:
    KohonenSOMResult(const boost::compute::context &context, const KohonenSOMSampleDataPtr &data);
    ~KohonenSOMResult();
    void freeClMem();
    void toClMem(const boost::compute::context &context);
    std::vector<std::vector<uint32_t>>* indexes();
    std::vector<float>* distances();
    cl_mem clDistances();
    cl_mem clIndexes();
};

}}

#endif /* kohonen_som_result_hpp */

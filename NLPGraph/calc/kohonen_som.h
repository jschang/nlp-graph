//
//  kohonen_som.h
//  NLPGraph
//
//  Created by Jonathan Schang on 10/24/15.
//
//

#ifndef kohonen_som_h
#define kohonen_som_h


#include "../nlpgraph.h"
#include "../util/logger.h"
#include "../util/opencl.h"

namespace NLPGraph {
    namespace Calc {
        
class KohonenSOMData {
private:
    // the weights for each node in the map
    std::vector<std::vector<double>> nodeWeights;
    // the final map dimensions
    std::vector<int> mapDimensions;
    // the vector length of each node's weights
    int nodeDimensions;
public:
    KohonenSOMData(std::vector<int> mapDimensions, int nodeDimensions) {
        this->mapDimensions = mapDimensions;
        this->nodeDimensions = nodeDimensions;
    }
    ~KohonenSOMData() {
    }
};

class KohonenSOM {
private:
    boost::compute::context       m_context;
    boost::compute::kernel        m_kernel;
    boost::compute::program       m_program;
    boost::compute::command_queue m_commandQueue;
    Util::LoggerType              m_logger;
public:
    bool clLogOn;
    bool clLogErrorOnly;
public:
    KohonenSOM(boost::compute::context &context);
    ~KohonenSOM();
    int train(KohonenSOMDataPtr data, const std::vector<double> &sampleData);
    std::vector<int> map(KohonenSOMDataPtr data, const std::vector<double> &sample);
};

}}

#endif /* kohonen_som_h */

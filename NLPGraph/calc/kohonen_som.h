//
//  kohonen_som.h
//  NLPGraph
//
//  Created by Jonathan Schang on 10/24/15.
//
//

#ifndef kohonen_som_h
#define kohonen_som_h

#import <numeric>
#include "../nlpgraph.h"
#include "../util/logger.h"
#include "../util/opencl.h"
#include "kohonen_som/kohonen_som_data.h"
#include "kohonen_som/kohonen_som_sample_data.h"
#include "kohonen_som/kohonen_som_result.h"

namespace NLPGraph {
    namespace Calc {

class KohonenSOM {
private:
    boost::compute::context       m_context;
    boost::compute::kernel        m_mappingKernel;
    boost::compute::kernel        m_weightUpdateKernel;
    boost::compute::program       m_program;
    boost::compute::command_queue m_commandQueue;
    Util::LoggerType              m_logger;
public:
    bool clLogOn;
    bool clLogErrorOnly;
public:
    KohonenSOM(const boost::compute::context &context);
    ~KohonenSOM();
    const boost::compute::command_queue& commandQueue() {
        return m_commandQueue;
    }
    /**
     * Convenience wrapper that simply iterates over map and updateWeights.
     */
    void train(const KohonenSOMDataPtr &data, const KohonenSOMSampleDataPtr &sampleData);
    /**
     * @return For each sample, in order, a map of node indices, up to max nodes, and the distance from the weights at that nodes index.
     */
    KohonenSOMResultPtr map(const KohonenSOMDataPtr &data, const KohonenSOMSampleDataPtr &sampleData);
    /**
     * Updates map node weights using the result passed in
     */
    void updateWeights(const KohonenSOMDataPtr &data, const KohonenSOMSampleDataPtr &sampleData, const KohonenSOMResultPtr &result, double radius, double learningRate);
};

}}

#endif /* kohonen_som_h */

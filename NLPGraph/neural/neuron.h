//
//  neuron.h
//  NLPGraph
//
//  Created by Jonathan Schang on 11/18/15.
//
//

#ifndef neuron_h
#define neuron_h

#import <numeric>
#include "../nlpgraph.h"
#include "constant.h"
#include "../util/logger.h"
#include "../util/opencl.h"

namespace NLPGraph {
    namespace Neural {
    
class Neuron {
private:
    uint64_t _id;
    double _threshold;
    std::vector<SynapsePtr> _outputSynapses;
    std::vector<SynapsePtr> _inputSynapses;
public:
    uint64_t id(uint64_t id = 0);
    double threshold(double threshold = -99999.999);
    std::vector<SynapsePtr> outputSynapses(const std::vector<SynapsePtr>& outputSynapses = kNullSynapses);
    std::vector<SynapsePtr> inputSynapses(const std::vector<SynapsePtr>& inputSynapses = kNullSynapses);
};

}}

#endif /* neuron_h */
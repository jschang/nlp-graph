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
#include "../util/logger.h"
#include "../util/opencl.h"

namespace NLPGraph {
    namespace Neural {
    
class Neuron {
private:
    double _threshold;
    std::vector<SynapsePtr> _outputSynapses;
    std::vector<SynapsePtr> _inputSynapses;
public:
    double threshold() {
        return _threshold;
    }
    std::vector<SynapsePtr>& outputSynapses() {
        return _outputSynapses;
    }
    std::vector<SynapsePtr>& inputSynapses() {
        return _inputSynapses;
    }
};

}}

#endif /* neuron_h */
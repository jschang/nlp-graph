//
//  synapse.h
//  NLPGraph
//
//  Created by Jonathan Schang on 11/18/15.
//
//

#ifndef synapse_h
#define synapse_h

#import <numeric>
#include "../nlpgraph.h"
#include "../util/logger.h"
#include "../util/opencl.h"

namespace NLPGraph {
    namespace Neural {
    
class Synapse {
private:
    double _weight;
    NeuronPtr _inNeuron;
    NeuronPtr _outNeuron;
public:
    double weight() {
        return _weight;
    }
    NeuronPtr inNeuron() {
        return _inNeuron;
    }
    NeuronPtr outNeuron() {
        return _outNeuron;
    }
};

}}

#endif /* synapse_h */
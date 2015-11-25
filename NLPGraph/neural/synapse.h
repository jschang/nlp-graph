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
    uint64_t _id;
    double _weight;
    NeuronPtr _inNeuron;
    NeuronPtr _outNeuron;
public:
    uint64_t id(uint64_t id);
    double weight(double weight);
    NeuronPtr inNeuron(const NeuronPtr &inNeuron);
    NeuronPtr outNeuron(const NeuronPtr &outNeuron);
};

const SynapsePtr kNullSynapse(nullptr);
const std::vector<SynapsePtr> kNullSynapses = std::vector<SynapsePtr>();

}}

#endif /* synapse_h */
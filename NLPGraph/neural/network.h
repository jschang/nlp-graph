//
//  network.h
//  NLPGraph
//
//  Created by Jonathan Schang on 11/18/15.
//
//

#ifndef network_h
#define network_h

#import <numeric>
#include "../nlpgraph.h"
#include "../util/logger.h"
#include "../util/opencl.h"

namespace NLPGraph {
    namespace Neural {
    
class Network {
private:
    std::map<uint64_t,NeuronPtr> _neurons;
    std::map<uint64_t,SynapsePtr> _synapses;
public:
    NeuronPtr& neuron(uint64_t id, const NeuronPtr& neuron);
    SynapsePtr& synapse(uint64_t id, const SynapsePtr& synapse);
};

}}

#endif /* network_h */

#import <numeric>
#include "../nlpgraph.h"
#include "neural.h"
#include "../util/math.h"
#include "../util/logger.h"
#include "../util/opencl.h"

using namespace NLPGraph;

namespace NLPGraph {
    namespace Neural {

extern const SynapsePtr kNullSynapse(nullptr);
extern const std::vector<SynapsePtr> kNullSynapses = std::vector<SynapsePtr>();
    
uint64_t Synapse::id(uint64_t id) {
    if(id!=0) {
        _id = id;
    }
    return _id;
}
double Synapse::weight(double weight) {
    if(!Util::Math::isEqual(weight,-99999.999)) {
        _weight = weight;
    }
    return _weight;
}
NeuronPtr Synapse::inNeuron(const NeuronPtr &inNeuron) {
    if(inNeuron.get()!=nullptr) {
        _inNeuron = inNeuron;
    }
    return _inNeuron;
}
NeuronPtr Synapse::outNeuron(const NeuronPtr &outNeuron) {
    if(outNeuron.get()!=nullptr) {
        _outNeuron = outNeuron;
    }
    return _outNeuron;
}

}}
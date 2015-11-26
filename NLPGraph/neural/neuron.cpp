#include "../util/math.h"
#include "neural.h"

using namespace NLPGraph;

namespace NLPGraph {
    namespace Neural {
    
extern const NeuronPtr kNullNeuron(nullptr);

uint64_t Neuron::id(uint64_t id) {
    if(id!=0) {
        _id = id;
    }
    return _id;
}
double Neuron::threshold(double threshold) {
    if(!Util::Math::isEqual(threshold,-99999.999)) {
        _threshold = threshold;
    }
    return _threshold;
}
std::vector<SynapsePtr> Neuron::outputSynapses(const std::vector<SynapsePtr>& outputSynapses) {
    if(outputSynapses.size()!=0) {
        _outputSynapses = outputSynapses;
    }
    return _outputSynapses;
}
std::vector<SynapsePtr> Neuron::inputSynapses(const std::vector<SynapsePtr>& inputSynapses) {
    if(inputSynapses.size()!=0) {
        _inputSynapses = inputSynapses;
    }
    return _inputSynapses;
}

}}
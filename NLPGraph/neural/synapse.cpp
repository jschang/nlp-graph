#import <numeric>
#include "../nlpgraph.h"
#include "neural.h"
#include "../util/math.h"
#include "../util/logger.h"
#include "../util/opencl.h"

using namespace NLPGraph;

namespace NLPGraph {
    namespace Neural {
    
uint64_t Synapse::id(uint64_t id = 0) {
    if(id!=0) {
        _id = id;
    }
    return _id;
}
double Synapse::weight(double weight = -99999.999) {
    if(!Util::Math::isEqual(weight,-99999.999)) {
        _weight = weight;
    }
    return _weight;
}
NeuronPtr Synapse::inNeuron(const NeuronPtr &inNeuron = kNullNeuron) {
    if(inNeuron.get()!=nullptr) {
        _inNeuron = inNeuron;
    }
    return _inNeuron;
}
NeuronPtr Synapse::outNeuron(const NeuronPtr &outNeuron = kNullNeuron) {
    if(outNeuron.get()!=nullptr) {
        _outNeuron = outNeuron;
    }
    return _outNeuron;
}

}}
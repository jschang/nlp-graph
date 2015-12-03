#include "../nlpgraph.h"
#include "neural.h"

namespace NLPGraph {
    namespace Neural {
    
NeuronPtr Network::neuron(uint64_t id, const NeuronPtr& neuronIn = kNullNeuron) {

    if(neuronIn.get()!=nullptr) {
        _neurons[id] = neuronIn;
    }
    return _neurons[id];
}

SynapsePtr Network::synapse(uint64_t id, const SynapsePtr& synapseIn = kNullSynapse) {

    if(synapseIn.get()!=nullptr) {
        _synapses[id] = synapseIn;
    }
    return _synapses[id];
}

std::vector<uint64_t> Network::layerAfterIds(std::vector<uint64_t> layer) {

    std::map<uint64_t,NeuronPtr> neuronsById = std::map<uint64_t,NeuronPtr>();
    std::vector<uint64_t> layerAfterIds = std::vector<uint64_t>();
    for(auto iter = layer.begin(); iter!=layer.end(); iter++) {
        NeuronPtr layerNeuron = _neurons[*iter];
        std::vector<SynapsePtr> synapses = layerNeuron->outputSynapses();
        for(auto iterSyn = synapses.begin(); iterSyn!=synapses.end(); iterSyn++) {
            neuronsById[(*iterSyn)->inNeuron()->id()] = (*iterSyn)->inNeuron();
        }
    }
    return layerAfterIds;
}

NetworkPtr Network::newNetworkFullyConnected(const uint32_t dims[], const uint32_t dimCount) {
    if(dimCount<2) {
        NLPGraphExceptionType exc;
        exc.msg = "Requires at least 2 layers";
        throw exc;
    }
    std::vector<std::vector<NeuronPtr>> neuronsByLayer;
    for(int dimIdx=0; dimIdx<dimCount; dimIdx++) {
        neuronsByLayer.push_back(std::vector<NeuronPtr>());
        for(int neuronIdx=0; neuronIdx<dims[dimIdx]; neuronIdx++) {
            NeuronPtr newNeuron(new Neuron());
            neuronsByLayer[dimIdx].push_back(newNeuron);
            
        }
    }
}

}}
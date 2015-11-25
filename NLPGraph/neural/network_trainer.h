//
//  network_trainer.h
//  NLPGraph
//
//  Created by Jonathan Schang on 11/18/15.
//
//

#ifndef network_trainer_h
#define network_trainer_h

#import <numeric>
#include "../nlpgraph.h"
#include "../util/logger.h"
#include "../util/opencl.h"

namespace NLPGraph {
    namespace Neural {
    
class NetworkTrainer {
private:
    std::vector<NetworkPtr> _networks;
public:
    /**
     * Adds a network for training.  All networks
     * must have the same number of outputs and inputs,
     * or an exception will be thrown.
     */
    void addNetwork(NetworkPtr network);
};

}}

#endif /* network_trainer_h */
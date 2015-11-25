//
//  network_runner.h
//  NLPGraph
//
//  Created by Jonathan Schang on 11/18/15.
//
//

#ifndef network_runner_h
#define network_runner_h

#import <numeric>
#include "../nlpgraph.h"
#include "../util/logger.h"
#include "../util/opencl.h"

namespace NLPGraph {
    namespace Neural {

class NetworkRunner {
private:
    std::vector<NetworkPtr> _networks;
public:
    /**
     * Adds a network for running.  All networks
     * must have the same number of outputs and inputs,
     * or an exception will be thrown.
     */
    void addNetwork(NetworkPtr network);
};

}}

#endif /* network_runner_h */

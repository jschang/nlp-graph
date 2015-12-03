//
//  nlpgraph.h
//  NLPGraph
//
//  Created by Jonathan Schang on 6/16/15.
//  Copyright (c) 2015 Jonathan Schang. All rights reserved.
//

#ifndef __NLPGraph__nlpgraph__
#define __NLPGraph__nlpgraph__

#include <boost/shared_ptr.hpp>
#include <boost/log/common.hpp>

namespace NLPGraph {
namespace Neural {
    class Neuron;
    class Synapse;
    class Network;
    typedef boost::shared_ptr<Neuron> NeuronPtr;
    typedef boost::shared_ptr<Synapse> SynapsePtr;
    typedef boost::shared_ptr<Network> NetworkPtr;
};
namespace Math {
    class Function1D;
    typedef boost::shared_ptr<Function1D> Function1DPtr;
    namespace Function {
        class TanH;
        class Logistic;
    }
}
namespace Calc {
    class LevensteinDamerauData;
    typedef boost::shared_ptr<LevensteinDamerauData> LevensteinDamerauDataPtr;
    class KohonenSOMData;
    class KohonenSOMSampleData;
    class KohonenSOMResult;
    typedef boost::shared_ptr<KohonenSOMData> KohonenSOMDataPtr;
    typedef boost::shared_ptr<KohonenSOMSampleData> KohonenSOMSampleDataPtr;
    typedef boost::shared_ptr<KohonenSOMResult> KohonenSOMResultPtr;
};
namespace Util {
    class OpenCL;
    template <class T> class ResourcePool;
    class Logger;
    class File;
    class TimeHelper;
};
namespace Dao {
    class Model;
    struct ModelException;
    typedef boost::shared_ptr<Model> ModelPtr;
}
namespace Dto {
    class Recollection;
    typedef boost::shared_ptr<Recollection> RecollectionPtr;
    class Symbol;
    typedef boost::shared_ptr<Symbol> SymbolPtr;
    class InputChannel;
    typedef boost::shared_ptr<InputChannel> InputChannelPtr;
    class RecollectionException;
    typedef boost::shared_ptr<RecollectionException> RecollectionExceptionPtr;
}
typedef struct NLPGraphException : boost::exception, std::exception {
    std::string msg;
    const char *what() const noexcept { return msg.c_str(); };
} NLPGraphExceptionType;
};

#endif /* defined(__NLPGraph__nlpgraph__) */

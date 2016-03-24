//
//  smith_waterman.hpp
//  NLPGraph
//
//  Created by Jonathan Schang on 2/29/16.
//
//

#ifndef smith_waterman_hpp
#define smith_waterman_hpp

#include "../nlpgraph.h"
#include "../util/logger.h"
#include "../util/opencl.h"
#include "smith_waterman/smith_waterman_data.h"

namespace NLPGraph {
namespace Calc {
        
class SmithWaterman {
public:
    bool clLogOn;
    bool clLogErrorOnly;
private:
    boost::shared_ptr<boost::compute::context>       m_context;
    boost::shared_ptr<boost::compute::kernel>        m_kernelCalc;
    boost::shared_ptr<boost::compute::program>       m_program;
    boost::shared_ptr<boost::compute::command_queue> m_commandQueue;
    boost::shared_ptr<Util::LoggerType>              m_logger;
public:
    SmithWaterman(const boost::compute::context &context);
    ~SmithWaterman();
    int calculate(SmithWatermanDataPtr data);
};
    
}}

#endif /* smith_waterman_hpp */

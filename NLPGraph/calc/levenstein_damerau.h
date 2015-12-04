//
//  levenstein_damerau.h
//  NLPGraph
//
//  Created by Jonathan Schang on 6/25/15.
//  Copyright (c) 2015 Jonathan Schang. All rights reserved.
//

#ifndef __NLPGraph__levenstein_damerau__
#define __NLPGraph__levenstein_damerau__

#include "../nlpgraph.h"
#include "../util/logger.h"
#include "../util/opencl.h"
#include "levenstein_damerau/levenstein_damerau_data.h"

namespace NLPGraph {
namespace Calc {

class LevensteinDamerau {
private:
    boost::shared_ptr<boost::compute::context>       m_context;
    boost::shared_ptr<boost::compute::kernel>        m_kernel;
    boost::shared_ptr<boost::compute::program>       m_program;
    boost::shared_ptr<boost::compute::command_queue> m_commandQueue;
    boost::shared_ptr<Util::LoggerType>              m_logger;
public:
    bool clLogOn;
    bool clLogErrorOnly;
public:
    LevensteinDamerau(const boost::compute::context &context);
    int calculate(LevensteinDamerauDataPtr data);
    int reconstruct(LevensteinDamerauDataPtr data);
};

}}

#endif /* defined(__NLPGraph__levenstein_damerau__) */

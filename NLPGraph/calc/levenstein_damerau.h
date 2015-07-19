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
#include <boost/compute.hpp>

namespace NLPGraph {
namespace Calc {

extern const char *kLevensteinDamerauOpenCLSource;

class LevensteinDamerau {
private:
    boost::compute::context       m_context;
    boost::compute::program       m_program;
    boost::compute::kernel        m_kernel;
    boost::compute::command_queue m_commandQueue;
    uint64_t*                     m_needle;
    uint64_t**                    m_haystack;
    unsigned int                  m_haystackSize;
    Util::LoggerType              m_logger;
public:
    LevensteinDamerau(boost::compute::context &context);
    ~LevensteinDamerau();
    int calculate(uint width, uint haystackSize, uint64_t* needle, uint64_t* haystack, uint64_t *distancesOut, uint64_t *operationsOut);
};

}}

#endif /* defined(__NLPGraph__levenstein_damerau__) */

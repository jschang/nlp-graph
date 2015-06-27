//
//  levenstein_damerau.cpp
//  NLPGraph
//
//  Created by Jonathan Schang on 6/25/15.
//  Copyright (c) 2015 Jonathan Schang. All rights reserved.
//

#define BOOST_LOG_DYN_LINK
#include "levenstein_damerau.h"
#include "../util/opencl.h"
#include <boost/compute.hpp>

#define LOG_E BOOST_LOG_SEV(m_logger,severity_level::critical) << __PRETTY_FUNCTION__ << " "
#define LOG_I BOOST_LOG_SEV(m_logger,severity_level::normal) << __PRETTY_FUNCTION__ << " "

using namespace boost::compute;
using namespace NLPGraph::Util;

namespace NLPGraph {
namespace Calc {

LevensteinDamerau::LevensteinDamerau(context &context) 
        : m_logger(boost::log::keywords::channel="NLPGraph::Calc::LevensteinDamerau") {
    m_context = context;
    device dev(m_context.get_device());
    m_commandQueue = command_queue(m_context, dev);
    m_program = OpenCL::createAndBuildProgram(BOOST_COMPUTE_STRINGIZE_SOURCE(    
        __kernel void calc_levenstein_damerau(
                const ulong widthIn,      // needle and each in haystack width
                const ulong lengthIn,     // length of the haystack 
                const __local ulong *needleIn,    // needle uint64_t's 
                const __local ulong *haystackIn,  // haystack uint64_t's 
                __global ulong *distancesOut      // results 
        ) { 
         
            int gid = get_global_id(0);
            distancesOut[gid] = gid;
        }
    ),m_context);
    m_kernel = kernel(m_program, "calc_levenstein_damerau");
}
LevensteinDamerau::~LevensteinDamerau() {
}
std::vector<ulong8_> LevensteinDamerau::calculate(uint16_t width, uint16_t haystackSize, uint64_t* needle, uint64_t** haystack) {

    vector<ulong8_> device_needle(width,m_context);
    std::vector<uint64_t> host_needle(needle,needle+width);
    copy(host_needle.begin(),host_needle.end(), device_needle.begin(), m_commandQueue);
    
    vector<ulong8_> device_haystack(width,m_context);
    std::vector<uint64_t> host_haystack(haystackSize*width);
    std::vector<uint64_t>::iterator host_haystack_iter = host_haystack.end();
    std::vector<uint64_t*> t(haystack,haystack+haystackSize);
    for(std::vector<uint64_t*>::iterator iter = t.begin(); iter != t.end(); iter++) {
        host_haystack.insert(host_haystack_iter,width,(**iter));
        host_haystack_iter = host_haystack.end();
    }
    copy(host_haystack.begin(), host_haystack.end(), device_haystack.begin(), m_commandQueue);
    
    vector<ulong8_> device_distances(haystackSize,m_context);
    
    m_kernel.set_arg(0,width);
    m_kernel.set_arg(1,haystackSize);
    m_kernel.set_arg(2,device_needle);
    m_kernel.set_arg(3,device_haystack);
    m_kernel.set_arg(4,device_distances);
    m_commandQueue.enqueue_1d_range_kernel(m_kernel, 0, haystackSize, 1);

    std::vector<ulong8_> result(haystackSize);
    copy(device_distances.begin(),device_distances.end(),result.begin(),m_commandQueue);

    return result;
}

}}
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

const uint CL_LOG_ON = 0b00000001;

LevensteinDamerau::LevensteinDamerau(context &context) 
        : m_logger(boost::log::keywords::channel="NLPGraph::Calc::LevensteinDamerau") {
    m_context = context;
    clLogOn = false;
    device dev(m_context.get_device());
    m_commandQueue = command_queue(m_context, dev);
    m_program = OpenCL::createAndBuildProgram(BOOST_COMPUTE_STRINGIZE_SOURCE(
    
        __constant uint CL_LOG_ON = 0b00000001; 
    
        inline uint append(const uint flags, __global char *log, uint logPos, __constant char* string) {
            if ( flags & CL_LOG_ON ) {
                uint idx=0;
                while(string[idx]!=0) {
                    log[logPos++] = string[idx++];
                }
                log[logPos]=0;
                return logPos;
            }
            return logPos;
        }
            
        __kernel void calc_levenstein_damerau(
                const uint flags,
                const uint widthIn,               // needle and each in haystack width
                __constant ulong *needleIn,       // needle uint64_t's 
                __global ulong *haystackIn,       // haystack uint64_t's 
                __global ulong *distancesOut,     // results 
                __global ulong *operationsOut,    // the operations to transform the haystack element into the needle
                __global char *logOut
        ) { 
            __private uint haystackRowIdx = get_global_id(0);
            
            __private uint logPos = haystackRowIdx * 2048;
            
            __private uint needleIdx = 0;
            __private uint haystackIdx = 0;
            __private ulong needleLast = 0;
            __private ulong haystackLast = 0;
            
            __private ulong distanceTotal = 0;
            
            while(needleIdx<widthIn  && (logPos=append(flags,logOut,logPos,"Iteration Start\n"))) {
            
                do {
                    if(haystackIdx>=widthIn) {
                        logPos=append(flags,logOut,logPos,"haystack ended...incrementing dist\n");
                        distanceTotal++;
                        needleIdx++;
                        break;
                    }
                    if(needleIdx>=widthIn) {
                        logPos=append(flags,logOut,logPos,"needle ended...incrementing dist\n");
                        distanceTotal++;
                        haystackIdx++;
                        break;
                    }

                    ulong needleCur = needleIn[needleIdx];
                    ulong haystackCur = haystackIn[(widthIn*haystackRowIdx)+haystackIdx];
                    
                    if(needleCur == haystackCur) {
                        if(haystackCur == needleLast) {
                            if(needleCur == haystackLast) {
                                if(needleLast == haystackLast) { 
                                    // 1111 - error: never been here before
                                    logPos=append(flags,logOut,logPos,"1111 - error: never been here before\n");
                                } else { 
                                    // 1110 - error: never been here before
                                    logPos=append(flags,logOut,logPos,"1110 - error: never been here before\n");
                                }
                            } else {
                                if(needleLast == haystackLast) { 
                                    // 1101 - error: never been here before
                                    logPos=append(flags,logOut,logPos,"1101 - error: never been here before\n");
                                } else { 
                                    // 1100 - error: never been here before
                                    logPos=append(flags,logOut,logPos,"1100 - error: never been here before\n");
                                }
                            }
                        } else {
                            if(needleCur == haystackLast) { 
                                if(needleLast == haystackLast) { 
                                    // 1011 - error: never been here before
                                    logPos=append(flags,logOut,logPos,"1011 - error: never been here before\n");
                                } else { 
                                    // 1010 - possible insertion
                                    logPos=append(flags,logOut,logPos,"1010 - possible insertion\n");
                                    distanceTotal++;
                                }
                            } else { 
                                if(needleLast == haystackLast) {
                                    // 1001 - continuing match
                                    logPos=append(flags,logOut,logPos,"1001 - continuing match\n");
                                } else {
                                    // 1000 - match restored
                                    logPos=append(flags,logOut,logPos,"1000 - match restored\n");
                                }
                            }
                        }
                    } else {
                        if(haystackCur == needleLast) {
                            if(needleCur == haystackLast) {
                                if(needleLast == haystackLast) {
                                    // 0111 - error: never been here before
                                    logPos=append(flags,logOut,logPos,"0111 - error: never been here before\n");
                                } else {
                                    // 0110 - transposition
                                    logPos=append(flags,logOut,logPos,"0110 - transposition\n");
                                    distanceTotal++;
                                }
                            } else {
                                if(needleLast == haystackLast) {
                                    // 0101 - repeat haystack
                                    logPos=append(flags,logOut,logPos,"0101 - repeat haystack\n");
                                    distanceTotal++;
                                } else {
                                    // 0100 - may be restoration of match
                                    logPos=append(flags,logOut,logPos,"0100 - may be restoration of match\n");
                                    needleIdx--;
                                    break;
                                }
                            }
                        } else {
                            if(needleCur == haystackLast) {
                                if(needleLast == haystackLast) {
                                    // 0011 - replacement or deletion
                                    logPos=append(flags,logOut,logPos,"0011 - replacement or deletion\n");
                                    //distanceTotal++;
                                } else {
                                    // 0010 - restore match, last was actually ommission in haystack
                                    logPos=append(flags,logOut,logPos,"0010 - restore match, last was actually ommission in haystack\n");
                                    haystackIdx--;
                                    distanceTotal--;
                                    break;
                                }
                            } else {
                                if(needleLast == haystackLast) {
                                    // 0001 - break match, replacement
                                    logPos=append(flags,logOut,logPos,"0001 - break match, replacement\n");
                                    distanceTotal++;
                                } else {
                                    // 0000 - continue broken match, replacement
                                    logPos=append(flags,logOut,logPos,"0000 - continue broken match, replacement\n");
                                    distanceTotal++;
                                }
                            }
                        }
                    }
                    
                    needleIdx++;
                    haystackIdx++;
                    needleLast = needleCur;
                    haystackLast = haystackCur;
                    
                } while(false);
                logPos=append(flags,logOut,logPos,(needleIdx<widthIn?"needleIdx<widthIn = true; ":"needleIdx<widthIn = false; "));
                logPos=append(flags,logOut,logPos,(haystackIdx<widthIn?"haystackIdx<widthIn = true\n":"haystackIdx<widthIn = false\n"));
            }
            distancesOut[haystackRowIdx] = distanceTotal;
        }
    ),m_context);
    m_kernel = kernel(m_program, "calc_levenstein_damerau");
}
LevensteinDamerau::~LevensteinDamerau() {
}
int LevensteinDamerau::calculate(uint width, uint haystackSize, uint64_t* needle, uint64_t* haystack, uint64_t *distancesOut, uint64_t *operationsOut) {

    int result = 0;

    vector<uint64_t> device_needle(width,m_context);
    std::vector<uint64_t> host_needle(needle, needle+width);
    copy(host_needle.begin(),host_needle.end(), device_needle.begin(), m_commandQueue);
    
    vector<uint64_t> device_haystack(haystackSize*width,m_context);
    std::vector<uint64_t> host_haystack(haystack, haystack+(haystackSize*width));
    copy(host_haystack.begin(), host_haystack.end(), device_haystack.begin(), m_commandQueue);
    
    vector<uint64_t> device_distances(haystackSize,m_context);
    vector<uint64_t> device_operations(haystackSize*(width*2),m_context);
    vector<char> device_log(50000,m_context);
    device_log[0] = 0;
    
    uint flags = (clLogOn ? CL_LOG_ON : 0);
    m_kernel.set_arg(0,flags);
    m_kernel.set_arg(1,width);
    m_kernel.set_arg(2,device_needle);
    m_kernel.set_arg(3,device_haystack);
    m_kernel.set_arg(4,device_distances);
    m_kernel.set_arg(5,device_operations);
    m_kernel.set_arg(6,device_log);
    m_commandQueue.enqueue_1d_range_kernel(m_kernel, 0, haystackSize, 1);

    char log[50000];
    copy(device_log.begin(),device_log.end(),(char*)&log,m_commandQueue);
    if(clLogOn) {
        LOG_I << "Run log:" << std::string(log);
    }
    
    copy(device_distances.begin(),device_distances.end(),distancesOut,m_commandQueue);
    copy(device_operations.begin(),device_operations.end(),operationsOut,m_commandQueue);
    
    m_kernel.

    return result;
}

}}
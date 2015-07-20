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
const uint CL_LOG_ERROR_ONLY = 0b00000010;

LevensteinDamerau::LevensteinDamerau(context &context) 
        : m_logger(boost::log::keywords::channel="NLPGraph::Calc::LevensteinDamerau") {
    m_context = context;
    clLogOn = false;
    clLogErrorOnly = 0;
    device dev(m_context.get_device());
    m_commandQueue = command_queue(m_context, dev);
    m_program = OpenCL::createAndBuildProgram(BOOST_COMPUTE_STRINGIZE_SOURCE(
    
        __constant uint CL_LOG_ON = 0b00000001; 
        __constant uint CL_LOG_ERROR_ONLY = 0b00000010;
        
        __constant uint CL_LOG_TYPE_ERROR = 0b00000001; 
    
        typedef struct {
            __private uint flags;
            __private uint widthIn;           // needle and each in haystack width
            __constant ulong *needleIn;       // needle uint64_t's 
            __global ulong *haystackIn;       // haystack uint64_t's 
            __global ulong *distancesOut;     // results 
            __global ulong *operationsOut;    // the operations to transform the haystack element into the needle
            __global char *logOut;
            __private uint haystackRowIdx;
            __private uint strLen;
            __local char *str;
            __local uint logPos;
            __private uint needleIdx;
            __private uint haystackIdx;
            __private ulong needleLast;
            __private ulong haystackLast;
            __private ulong distanceTotal;
            __local uint needleCur;
            __local uint haystackCur;
        } self_type;
        
        uint append_preamble(self_type self);
        uint append(self_type self, __local char* string, __private uint descr);
        void al(self_type self, __private uint descr, __local char* str);
        void ac(self_type self, __private uint descr, __constant char* str);
        __local char * z(__local char *in, __private int len);
        __local char * s(__local char *strOut, __constant char *strIn);
        __local char* itoa(self_type self, __private int inNum, __private int base);
        
        inline void al(self_type self, __private uint descr, __local char* str) {
            self.logPos = append_preamble(self);
            self.logPos = append(self,str,descr);
        }
        inline void ac(self_type self, __private uint descr, __constant char* str) {
            self.logPos = append_preamble(self);
            self.logPos = append(self,s(z(self.str,self.strLen),str),descr);
        }
        
        inline __local char * s(__local char *strOut, __constant char *strIn) {
            int i=0;
            while(strIn[i]!='\0') {
                strOut[i] = strIn[i];
                i++;
            }
            return strOut;
        }
        
        inline __local char * z(__local char *in, __private int len) {
            for(int i=0; i<len; i++) {
                in[i] = 0;
            }
            return in;
        }
        
        inline uint append_preamble(self_type self) {
            self.logPos = append(self,s(z(self.str,self.strLen),"global_id:"),0);
            self.logPos = append(self,itoa(self,self.haystackRowIdx,10),0);
            self.logPos = append(self,s(z(self.str,self.strLen),",needle:"),0);
            self.logPos = append(self,itoa(self,self.needleIdx,10),0);
            self.logPos = append(self,s(z(self.str,self.strLen),",haystack:"),0);
            self.logPos = append(self,itoa(self,self.haystackIdx,10),0);
            return self.logPos;
        }
    
        inline uint append(self_type self, __local char* string, __private uint descr) {
            if ( self.flags & CL_LOG_ON ) {
                if( ((self.flags & CL_LOG_ERROR_ONLY) && (descr & CL_LOG_TYPE_ERROR))
                        || !(self.flags & CL_LOG_ERROR_ONLY) ) {
                    uint idx=0;
                    while(string[idx]!=0) {
                        self.logOut[self.logPos++] = string[idx++];
                    }
                    self.logOut[self.logPos]=0;
                    return self.logPos;
                }
            }
            return 1;
        }
        
        inline void reverse(__local char *str, __private int len) {
        }
        
        inline __local char* itoa(self_type self, __private int inNum, __private int base) {
            int num = inNum;
            z(self.str,self.strLen);
            int i = 0;
            bool isNegative = false;
            if (num == 0) {
                self.str[i++] = '0';
                self.str[i] = '\0';
                return self.str;
            }
            if (num < 0 && base == 10) {
                isNegative = true;
                num = -num;
            }
            while (num != 0) {
                int rem = num % base;
                self.str[i++] = (rem > 9)? (rem-10) + 'a' : rem + '0';
                num = num/base;
            }
            if (isNegative) {
                self.str[i++] = '-';
            }
            self.str[i] = '\0';
            reverse(self.str, i);
            return self.str;
        }
            
        __kernel void calc_levenstein_damerau(
                __private uint flags,
                __private uint widthIn,           // needle and each in haystack width
                __constant ulong *needleIn,       // needle uint64_t's 
                __global ulong *haystackIn,       // haystack uint64_t's 
                __global ulong *distancesOut,     // results 
                __global ulong *operationsOut,    // the operations to transform the haystack element into the needle
                __global char *logOut
        ) { 
            self_type self;
        
            __local char strAr[255];  // didn't have another way to allocate it, 
                                      // and the param has to retain addr space
            __local char *str;
            __private int strLen;
            strLen = 255;
            str = (__local char*)&strAr;
            z(strAr,strLen);
            self.str = str;
            self.strLen = strLen;
            self.flags = flags;
            self.widthIn = widthIn;
            self.needleIn = needleIn;
            
            self.haystackRowIdx = get_global_id(0);

            self.logOut = logOut;            
            self.logOut[0] = 0x13;
            self.logPos = self.haystackRowIdx * 2048;
            
            self.logPos = 0;
            self.needleIdx = 0;
            self.haystackIdx = 0;
            self.needleLast = 0;
            self.haystackLast = 0;
            self.distanceTotal = 0;
            
            ac(self,0,"test\n");
            while(self.needleIdx<widthIn  && (self.logPos=append(self,s(z(self.str,self.strLen),"Iteration Start\n"),0))) {
            
                do {
                    if(self.haystackIdx>=self.widthIn) {
                        ac(self,0,"haystack ended...incrementing dist\n");
                        self.distanceTotal++;
                        self.needleIdx++;
                        break;
                    }
                    if(self.needleIdx>=self.widthIn) {
                        ac(self,0,"needle ended...incrementing dist\n");
                        self.distanceTotal++;
                        self.haystackIdx++;
                        break;
                    }

                    self.needleCur = self.needleIn[self.needleIdx];
                    self.haystackCur = self.haystackIn[ ( self.widthIn * self.haystackRowIdx ) + self.haystackIdx ];
                    
                    if(self.needleCur == self.haystackCur) {
                        if(self.haystackCur == self.needleLast) {
                            if(self.needleCur == self.haystackLast) {
                                if(self.needleLast == self.haystackLast) { 
                                    // 1111 - error: never been here before
                                    ac(self,CL_LOG_TYPE_ERROR," - 1111 - error: never been here before\n");
                                } else { 
                                    // 1110 - error: never been here before
                                    ac(self,CL_LOG_TYPE_ERROR," - 1110 - error: never been here before\n");
                                }
                            } else {
                                if(self.needleLast == self.haystackLast) { 
                                    // 1101 - error: never been here before
                                    ac(self,CL_LOG_TYPE_ERROR," - 1101 - error: never been here before\n");
                                } else { 
                                    // 1100 - error: never been here before
                                    ac(self,CL_LOG_TYPE_ERROR," - 1100 - error: never been here before\n");
                                }
                            }
                        } else {
                            if(self.needleCur == self.haystackLast) { 
                                if(self.needleLast == self.haystackLast) { 
                                    // 1011 - error: never been here before
                                    ac(self,CL_LOG_TYPE_ERROR," - 1011 - error: never been here before\n");
                                } else { 
                                    // 1010 - possible insertion
                                    ac(self,0," - 1010 - possible insertion\n");
                                    self.distanceTotal++;
                                }
                            } else { 
                                if(self.needleLast == self.haystackLast) {
                                    // 1001 - continuing match
                                    ac(self,0," - 1001 - continuing match\n");
                                } else {
                                    // 1000 - match restored
                                    ac(self,0," - 1000 - match restored\n");
                                }
                            }
                        }
                    } else {
                        if(self.haystackCur == self.needleLast) {
                            if(self.needleCur == self.haystackLast) {
                                if(self.needleLast == self.haystackLast) {
                                    // 0111 - error: never been here before
                                    ac(self,CL_LOG_TYPE_ERROR," - 0111 - error: never been here before\n");
                                } else {
                                    // 0110 - transposition
                                    ac(self,0," - 0110 - transposition\n");
                                    self.distanceTotal++;
                                }
                            } else {
                                if(self.needleLast == self.haystackLast) {
                                    // 0101 - repeat haystack
                                    ac(self,0," - 0101 - repeat haystack\n");
                                    self.distanceTotal++;
                                } else {
                                    // 0100 - may be restoration of match
                                    ac(self,0," - 0100 - may be restoration of match\n");
                                    self.needleIdx--;
                                    break;
                                }
                            }
                        } else {
                            if(self.needleCur == self.haystackLast) {
                                if(self.needleLast == self.haystackLast) {
                                    // 0011 - replacement or deletion
                                    ac(self,0," - 0011 - replacement or deletion\n");
                                    //distanceTotal++;
                                } else {
                                    // 0010 - restore match, last was actually ommission in haystack
                                    ac(self,0," - 0010 - restore match, last was actually ommission in haystack\n");
                                    self.haystackIdx--;
                                    self.distanceTotal--;
                                    break;
                                }
                            } else {
                                if(self.needleLast == self.haystackLast) {
                                    // 0001 - break match, replacement
                                    //logPos=log(haystackRowIdx,needleIdx,haystackIdx,str,strLen,)
                                    ac(self,0," - 0001 - break match, replacement\n");
                                    self.distanceTotal++;
                                } else {
                                    // 0000 - continue broken match, replacement
                                    ac(self,0," - 0000 - continue broken match, replacement\n");
                                    self.distanceTotal++;
                                }
                            }
                        }
                    }
                    
                    self.needleIdx++;
                    self.haystackIdx++;
                    self.needleLast = self.needleCur;
                    self.haystackLast = self.haystackCur;
                    
                } while(false);
                ac(self,0,(self.needleIdx<self.widthIn?"needleIdx<widthIn = true; ":"needleIdx<widthIn = false; "));
                ac(self,0,(self.haystackIdx<self.widthIn?"haystackIdx<widthIn = true\n":"haystackIdx<widthIn = false\n"));
            }
            self.distancesOut[self.haystackRowIdx] = self.distanceTotal;
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
    
    uint flags = (clLogOn ? CL_LOG_ON : 0) 
        | (clLogErrorOnly ? CL_LOG_ERROR_ONLY : 0);
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

    return result;
}

}}
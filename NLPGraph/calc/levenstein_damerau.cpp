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

using namespace NLPGraph::Util;

namespace NLPGraph {
namespace Calc {

/* ************* */
/* HEADER SOURCE */
/* ************* */
const char *kLevensteinDamerauOpenCLHeader = BOOST_COMPUTE_STRINGIZE_SOURCE(

__constant uint CL_LOG_ON         = 0b00000001; 
__constant uint CL_LOG_ERROR_ONLY = 0b00000010;

__constant uint CL_LOG_TYPE_ERROR = 0b00000001;  

typedef struct {
    uint flags;
    uint widthIn;                     // needle and each in haystack width
    __constant ulong *needleIn;       // needle uint64_t's 
    __global ulong *haystackIn;       // haystack uint64_t's 
    __global ulong *distancesOut;     // results 
    __global ulong *operationsOut;    // the operations to transform the haystack element into the needle
    __global char *logOut;
    uint logLength;
    uint haystackSize;
    uint haystackRowIdx;
    uint strLen;
    char *str;
    uint logPos;
    uint needleIdx;
    uint haystackIdx;
    ulong needleLast;
    ulong haystackLast;
    ulong distanceTotal;
    uint needleCur;
    uint haystackCur;
} levenstein_damerau_type;

uint append_preamble(levenstein_damerau_type *self, uint descr);
uint append(levenstein_damerau_type *self, char* string, uint descr);
void al(levenstein_damerau_type *self, uint descr, char* str);
void ac(levenstein_damerau_type *self, uint descr, __constant char* str);
void aci(levenstein_damerau_type *self, uint descr, __constant char* str, ulong num);
char * z(char *in, int len);
__global char * zg(__global char *in, int len);
char * s(char *strOut, __constant char *strIn);
char * itoa(levenstein_damerau_type *self, ulong inNum, int base);

);

/* ****************** */
/* SUPPORT LIB SOURCE */
/* ****************** */
const char *kLevensteinDamerauOpenCLSupprtSource = BOOST_COMPUTE_STRINGIZE_SOURCE(

inline void al(levenstein_damerau_type *self, uint descr, char* str) {
    self->logPos = append_preamble(self,descr);
    self->logPos = append(self,str,descr);
}
inline void ac(levenstein_damerau_type *self, uint descr, __constant char* str) {
    self->logPos = append_preamble(self,descr);
    self->logPos = append(self,s(z(self->str,self->strLen),str),descr);
}
inline void aci(levenstein_damerau_type *self, uint descr, __constant char* str, ulong num) {
    self->logPos = append_preamble(self,descr);
    self->logPos = append(self,s(z(self->str,self->strLen),str),descr);
    self->logPos = append(self,itoa(self,num,10),descr);
    self->logPos = append(self,s(z(self->str,self->strLen),"\n"),descr);
}

/**
 * copy a __constant string to a __private one, returning
 * the pointer to the __private one.
 */
inline char * s(char *strOut, __constant char *strIn) {
    int i=0;
    while(strIn[i]!='\0') {
        strOut[i] = strIn[i];
        i++;
    }
    return strOut;
}

/**
 * Zero out a __private string of a given length.
 */
inline char * z(char *in, int len) {
    for(int i=0; i<len; i++) {
        in[i] = 0;
    }
    return in;
}

/**
 * Zero out a __private string of a given length.
 */
inline __global char * zg(__global char *in, int len) {
    for(int i=0; i<len; i++) {
        in[i] = 0;
    }
    return in;
}

inline uint append_preamble(levenstein_damerau_type *self, uint descr) {

    self->logPos = append(self,s(z(self->str,self->strLen),"global_id:"),descr);
        self->logPos = append(self,itoa(self,self->haystackRowIdx,10),descr);
        
    self->logPos = append(self,s(z(self->str,self->strLen),",needleIdx:"),descr);
        self->logPos = append(self,itoa(self,self->needleIdx,10),descr);
        
    self->logPos = append(self,s(z(self->str,self->strLen),",haystackIdx:"),descr);
        self->logPos = append(self,itoa(self,self->haystackIdx,10),descr);
        
    self->logPos = append(self,s(z(self->str,self->strLen),",needleCur:"),descr);
        self->logPos = append(self,itoa(self,self->needleCur,10),descr);
        
    self->logPos = append(self,s(z(self->str,self->strLen),",haystackCur:"),descr);
        self->logPos = append(self,itoa(self,self->haystackCur,10),descr);
        
    self->logPos = append(self,s(z(self->str,self->strLen)," - "),descr);
    return self->logPos;
}

/**
 * Core logging function
 */
inline uint append(levenstein_damerau_type *self, char* string, uint descr) {
    if ( self->flags & CL_LOG_ON ) {
        if( ((self->flags & CL_LOG_ERROR_ONLY) && (descr & CL_LOG_TYPE_ERROR))
                || !(self->flags & CL_LOG_ERROR_ONLY) ) {
            uint idx=0;
            while(string[idx]!=0) {
                self->logOut[self->logPos++] = string[idx++];
            }
            self->logOut[self->logPos]=0;
            return self->logPos;
        }
    }
    return 1;
}

/**
 * Will eventually reverse a string...used in itoa
 */
inline void reverse(char *str, int len) {
}

inline char* itoa(levenstein_damerau_type *self, ulong inNum, int base) {
    ulong num = inNum;
    z(self->str,self->strLen);
    int i = 0;
    //bool isNegative = false;
    if (num == 0) {
        self->str[i++] = '0';
        self->str[i] = '\0';
        return self->str;
    }
    /*if (num < 0 && base == 10) {
        isNegative = true;
        num = -num;
    }*/
    while (num != 0) {
        int rem = num % base;
        self->str[i++] = (rem > 9)? (rem-10) + 'a' : rem + '0';
        num = num/base;
    }
    /*if (isNegative) {
        self->str[i++] = '-';
    }*/
    self->str[i] = '\0';
    reverse(self->str, i);
    return self->str;
}

);

/* ************* */
/* KERNEL SOURCE */
/* ************* */
const char *kLevensteinDamerauOpenCLSource = BOOST_COMPUTE_STRINGIZE_SOURCE(
    
__kernel void calc_levenstein_damerau(
        uint flags,
        uint widthIn,           // needle and each in haystack width
        __constant ulong *needleIn,       // needle uint64_t's 
        __global ulong *haystackIn,       // haystack uint64_t's 
        __global ulong *distancesOut,     // results 
        __global ulong *operationsOut,    // the operations to transform the haystack element into the needle
        __global char  *logOut,
        uint logLength,
        uint haystackSize
) { 
    levenstein_damerau_type self;

    char strAr[255];  // didn't have another way to allocate it, 
                      // and the param has to retain addr space
    char *str;
    int strLen;
    
    strLen = 255;
    str = (char*)&strAr;
    z(strAr,strLen);
    
    self.str = str;
    self.strLen = strLen;
    self.flags = flags;
    self.widthIn = widthIn;
    self.needleIn = needleIn;
    
    self.haystackSize = haystackSize;
    self.haystackRowIdx = get_global_id(0);

    self.logLength = logLength;
    self.logOut = logOut;
    self.logPos = self.haystackRowIdx * 2048;
    self.logOut[self.logPos] = 0x13;
    zg(self.logOut+self.logPos,2048);
    
    self.needleIdx = 0;
    self.haystackIdx = 0;
    self.needleLast = 0;
    self.haystackLast = 0;
    self.distanceTotal = 0;
    self.distancesOut = distancesOut;
    
    aci(&self,0,"strLen:         ",self.strLen);
    aci(&self,0,"flags:          ",self.flags);
    aci(&self,0,"widthIn:        ",self.widthIn);

    aci(&self,0,"haystackSize:   ",self.haystackSize);
    aci(&self,0,"haystackRowIdx: ",self.haystackRowIdx);

    aci(&self,0,"logLength:      ",self.logLength);            
    aci(&self,0,"logPos:         ",self.logPos);

    while( self.needleIdx < self.widthIn ) {

        do {
            if(self.haystackIdx>=self.widthIn) {
                ac(&self,0,"haystack ended...incrementing dist\n");
                self.distanceTotal++;
                self.needleIdx++;
                break;
            }
            if(self.needleIdx>=self.widthIn) {
                ac(&self,0,"needle ended...incrementing dist\n");
                self.distanceTotal++;
                self.haystackIdx++;
                break;
            }
            self.needleCur = self.needleIn[ self.needleIdx ];
            self.haystackCur = self.haystackIn[ ( self.widthIn * self.haystackRowIdx ) + self.haystackIdx ];
 /*          } while(false);
    }*/
            
            if(self.needleCur == self.haystackCur) {
                if(self.haystackCur == self.needleLast) {
                    if(self.needleCur == self.haystackLast) {
                        if(self.needleLast == self.haystackLast) { 
                            // 1111 - error: never been here before
                            ac(&self,CL_LOG_TYPE_ERROR,"1111 - error: never been here before\n");
                        } else { 
                            // 1110 - error: never been here before
                            ac(&self,CL_LOG_TYPE_ERROR,"1110 - error: never been here before\n");
                        }
                    } else {
                        if(self.needleLast == self.haystackLast) { 
                            // 1101 - error: never been here before
                            ac(&self,CL_LOG_TYPE_ERROR,"1101 - error: never been here before\n");
                        } else { 
                            // 1100 - error: never been here before
                            ac(&self,CL_LOG_TYPE_ERROR,"1100 - error: never been here before\n");
                        }
                    }
                } else {
                    if(self.needleCur == self.haystackLast) { 
                        if(self.needleLast == self.haystackLast) { 
                            // 1011 - error: never been here before
                            ac(&self,CL_LOG_TYPE_ERROR,"1011 - error: never been here before\n");
                        } else { 
                            // 1010 - possible insertion
                            ac(&self,0,"1010 - possible insertion\n");
                            self.distanceTotal++;
                        }
                    } else { 
                        if(self.needleLast == self.haystackLast) {
                            // 1001 - continuing match
                            ac(&self,0,"1001 - continuing match\n");
                        } else {
                            // 1000 - match restored
                            ac(&self,0,"1000 - match restored\n");
                        }
                    }
                }
            } else {
                if(self.haystackCur == self.needleLast) {
                    if(self.needleCur == self.haystackLast) {
                        if(self.needleLast == self.haystackLast) {
                            // 0111 - error: never been here before
                            ac(&self,CL_LOG_TYPE_ERROR,"0111 - error: never been here before\n");
                        } else {
                            // 0110 - transposition
                            ac(&self,0,"0110 - transposition\n");
                            self.distanceTotal++;
                        }
                    } else {
                        if(self.needleLast == self.haystackLast) {
                            // 0101 - repeat haystack
                            ac(&self,0,"0101 - repeat haystack\n");
                            self.distanceTotal++;
                        } else {
                            // 0100 - may be restoration of match
                            ac(&self,0,"0100 - may be restoration of match\n");
                            self.needleIdx--;
                            break;
                        }
                    }
                } else {
                    if(self.needleCur == self.haystackLast) {
                        if(self.needleLast == self.haystackLast) {
                            // 0011 - replacement or deletion
                            ac(&self,0,"0011 - replacement or deletion\n");
                            //distanceTotal++;
                        } else {
                            // 0010 - restore match, last was actually ommission in haystack
                            ac(&self,0,"0010 - restore match, last was actually ommission in haystack\n");
                            self.haystackIdx--;
                            self.distanceTotal--;
                            break;
                        }
                    } else {
                        if(self.needleLast == self.haystackLast) {
                            // 0001 - break match, replacement
                            //logPos=log(haystackRowIdx,needleIdx,haystackIdx,str,strLen,)
                            ac(&self,0,"0001 - break match, replacement\n");
                            self.distanceTotal++;
                        } else {
                            // 0000 - continue broken match, replacement
                            ac(&self,0,"0000 - continue broken match, replacement\n");
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
        ac(&self,0,self.needleIdx<self.widthIn?"needleIdx<widthIn = true; \n":"needleIdx<widthIn = false; \n");
        ac(&self,0,self.haystackIdx<self.widthIn?"haystackIdx<widthIn = true\n":"haystackIdx<widthIn = false\n");
    }
    self.distancesOut[self.haystackRowIdx] = self.distanceTotal;
}

);

const uint CL_LOG_ON = 0b00000001;
const uint CL_LOG_ERROR_ONLY = 0b00000010;

LevensteinDamerau::LevensteinDamerau(boost::compute::context &context) 
        : m_logger(boost::log::keywords::channel="NLPGraph::Calc::LevensteinDamerau") {
    m_context = context;
    clLogOn = false;
    clLogErrorOnly = 0;
    boost::compute::device dev(m_context.get_device());
    m_commandQueue = boost::compute::command_queue(m_context, dev);
    
    if(false) {
        /*std::vector<boost::compute::program> programs;
        
        int headerSize = sizeof(char)*strlen(kLevensteinDamerauOpenCLHeader);
        int sourceSize = sizeof(char)*strlen(kLevensteinDamerauOpenCLSource);
        int supportSize = sizeof(char)*strlen(kLevensteinDamerauOpenCLSupprtSource);
        
        char * source = (char *)malloc(headerSize+sourceSize+1);
        char * support = (char *)malloc(headerSize+supportSize+1);
        
        memset(source,0,headerSize+sourceSize+1);
        memset(support,0,headerSize+supportSize+1);
        
        memcpy(support, kLevensteinDamerauOpenCLHeader, headerSize);
        memcpy(source, kLevensteinDamerauOpenCLHeader, headerSize);
        
        memcpy(support+headerSize, kLevensteinDamerauOpenCLSupprtSource, supportSize);
        memcpy(source+headerSize, kLevensteinDamerauOpenCLSource, sourceSize);
        
        LOG_I << "Support:\n" << support;
        LOG_I << "Source:\n" << source;
        
        boost::compute::program pSupport = OpenCL::createAndBuildProgram(support,m_context);
        boost::compute::program pProgram = OpenCL::createAndBuildProgram(source,m_context);
        
        programs.push_back(pSupport);
        programs.push_back(pProgram);
        m_kernel = boost::compute::kernel(boost::compute::program::link(programs, m_context), "calc_levenstein_damerau");
        */
    } else {
    
        std::vector<boost::compute::program> programs;
        
        int headerSize = sizeof(char)*strlen(kLevensteinDamerauOpenCLHeader);
        int sourceSize = sizeof(char)*strlen(kLevensteinDamerauOpenCLSource);
        int supportSize = sizeof(char)*strlen(kLevensteinDamerauOpenCLSupprtSource);
        
        char * source = (char *)malloc(headerSize+supportSize+sourceSize+1);
        
        memset(source,0,headerSize+sourceSize+supportSize+1);
        
        memcpy(source, kLevensteinDamerauOpenCLHeader, headerSize);
        memcpy(source+headerSize, kLevensteinDamerauOpenCLSupprtSource, supportSize);
        memcpy(source+headerSize+supportSize, kLevensteinDamerauOpenCLSource, sourceSize);
        
        LOG_I << "Source:\n" << source;
        
        boost::compute::program pProgram = OpenCL::createAndBuildProgram(source,m_context);
        m_kernel = boost::compute::kernel(pProgram, "calc_levenstein_damerau");
    }
}
LevensteinDamerau::~LevensteinDamerau() {
}
int LevensteinDamerau::calculate(uint width, uint haystackSize, uint64_t* needle, uint64_t* haystack, uint64_t *distancesOut, uint64_t *operationsOut) {

    int result = 0;

    boost::compute::vector<uint64_t> device_needle(width,m_context);
    std::vector<uint64_t> host_needle(needle, needle+width);
    copy(host_needle.begin(),host_needle.end(), device_needle.begin(), m_commandQueue);
    
    boost::compute::vector<uint64_t> device_haystack(haystackSize*width,m_context);
    std::vector<uint64_t> host_haystack(haystack, haystack+(haystackSize*width));
    boost::compute::copy(host_haystack.begin(), host_haystack.end(), device_haystack.begin(), m_commandQueue);
    
    uint64_t zero = 0;
    boost::compute::vector<uint64_t> device_distances(haystackSize,(const uint64_t)&zero,m_commandQueue);
    
    int count = (haystackSize*(width*2));
    boost::compute::vector<uint64_t> device_operations(count,(const uint64_t)&zero,m_commandQueue);
    
    uint logLength = 50000;
    boost::compute::vector<char> device_log(logLength,(const uint64_t)&zero,m_commandQueue);
    
    uint flags = (clLogOn ? CL_LOG_ON : 0) 
        | (clLogErrorOnly ? CL_LOG_ERROR_ONLY : 0);
    m_kernel.set_arg(0,flags);
    m_kernel.set_arg(1,width);
    m_kernel.set_arg(2,device_needle);
    m_kernel.set_arg(3,device_haystack);
    m_kernel.set_arg(4,device_distances);
    m_kernel.set_arg(5,device_operations);
    m_kernel.set_arg(6,device_log);
    m_kernel.set_arg(7,logLength);
    m_kernel.set_arg(8,haystackSize);
    m_commandQueue.enqueue_1d_range_kernel(m_kernel, 0, haystackSize, 1);

    std::vector<char> host_log(logLength);
    copy(device_log.begin(),device_log.end(),host_log.begin(),m_commandQueue);
    if(clLogOn) {
        LOG_I << "Run log:\n" << &host_log;
    }
    
    //boost::compute::copy(device_distances.begin(), device_distances.end(), host_distances_vec.begin(), m_commandQueue);
    //std::copy(host_distances_vec.begin(), host_distances_vec.end(), distancesOut);
    
    //boost::compute::copy(device_operations.begin(), device_operations.end(), host_operations_vec.begin(), m_commandQueue);
    //std::copy(host_operations_vec.begin(), host_operations_vec.end(), operationsOut);

    return result;
}

}}
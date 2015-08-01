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
#include "../util/string.h"
#include <boost/compute.hpp>
#include <exception>

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
    __global long *distancesOut;     // results 
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
    long currentLocation;
} levenstein_damerau_type;

uint a(levenstein_damerau_type *self, char* string, uint descr);

void al(levenstein_damerau_type *self, uint descr, char* str);
void ac(levenstein_damerau_type *self, uint descr, __constant char* str);
void aci(levenstein_damerau_type *self, uint descr, __constant char* str, ulong num);

char * z(char *in, int len);
__global char * zg(__global char *in, int len);
char * s(char *strOut, __constant char *strIn);
char * utoa(levenstein_damerau_type *self, ulong inNum, int base);
char * itoa(levenstein_damerau_type *self, long inNum, int base);

uint append_state(levenstein_damerau_type *self, uint descr);

);

/* ****************** */
/* SUPPORT LIB SOURCE */
/* ****************** */
const char *kLevensteinDamerauOpenCLSupprtSource = BOOST_COMPUTE_STRINGIZE_SOURCE(

inline void al(levenstein_damerau_type *self, uint descr, char* str) {
    a(self, str, descr);
}
inline void ac(levenstein_damerau_type *self, uint descr, __constant char* str) {
    a(self, s(z(self->str, self->strLen), str), descr);
}
inline void aci(levenstein_damerau_type *self, uint descr, __constant char* str, ulong num) {
    a(self, s(z(self->str, self->strLen), str), descr);
    a(self, utoa(self, num, 10), descr);
    a(self, s(z(self->str, self->strLen), "\n"),descr);
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
 * Zero out a __global string of a given length.
 */
inline __global char * zg(__global char *in, int len) {
    for(int i=0; i<len; i++) {
        in[i] = 0;
    }
    return in;
}

inline uint append_state(levenstein_damerau_type *self, uint descr) {

    a(self,s(z(self->str,self->strLen),"gid:"),descr);
    a(self,utoa(self,self->haystackRowIdx,10),descr);
    
    z(self->str,self->strLen);
    s(self->str,",dT:");
    a(self,self->str,descr);
    utoa(self,self->distanceTotal,10);
    a(self,self->str,descr);
        
    z(self->str,self->strLen);
    s(self->str,",nI:");
    a(self,self->str,descr);
    utoa(self,self->needleIdx,10);
    a(self,self->str,descr);
        
    z(self->str,self->strLen);  
    s(self->str,",hI:");
    a(self,self->str,descr);
    utoa(self,self->haystackIdx,10);
    a(self,self->str,descr);
    
    z(self->str,self->strLen);   
    s(self->str,",nL:");
    a(self,self->str,descr);
    utoa(self,self->needleLast,10);
    a(self,self->str,descr);
        
    z(self->str,self->strLen);   
    s(self->str,",hL:");
    a(self,self->str,descr);
    utoa(self,self->haystackLast,10);
    a(self,self->str,descr);
    
    z(self->str,self->strLen);   
    s(self->str,",nC:");
    a(self,self->str,descr);
    utoa(self,self->needleCur,10);
    a(self,self->str,descr);
        
    z(self->str,self->strLen);   
    s(self->str,",hC:");
    a(self,self->str,descr);
    utoa(self,self->haystackCur,10);
    a(self,self->str,descr);
    
    z(self->str,self->strLen);   
    s(self->str,",cL:");
    a(self,self->str,descr);
    itoa(self,self->currentLocation,2);
    a(self,self->str,descr);

    return self->logPos;
}

/**
 * Core logging function
 */
inline uint a(levenstein_damerau_type *self, char* string, uint descr) {
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
 * Will eventually reverse a string...used in utoa
 */
inline void reverse(char *str, int len) {
    if(!len || len==1) {
        return;
    }
    int i=0, j=len-1;
    
    while(j>(len/2)) {
        char left = str[i];
        str[i] = str[j];
        str[j] = left;
        j--; 
        i++;
    }
}

inline char* utoa(levenstein_damerau_type *self, ulong inNum, int base) {
    ulong num = inNum;
    z(self->str,self->strLen);
    int i = 0;
    if (num == 0) {
        self->str[i++] = '0';
        self->str[i] = '\0';
        return self->str;
    }
    while (num != 0) {
        int rem = num % base;
        self->str[i++] = (rem > 9) ? (rem-10) + 'a' : rem + '0';
        num = num/base;
    }
    self->str[i] = '\0';
    reverse(self->str, i);
    return self->str;
}

inline char* itoa(levenstein_damerau_type *self, long inNum, int base) {
    long num = inNum;
    z(self->str,self->strLen);
    int i = 0;
    bool isNegative = false;
    if (num == 0) {
        self->str[i++] = '0';
        self->str[i] = '\0';
        return self->str;
    }
    if (num < 0 && base == 10) {
        isNegative = true;
        num = -num;
    }
    while (num != 0) {
        int rem = num % base;
        self->str[i++] = (rem > 9) ? (rem-10) + 'a' : rem + '0';
        num = num/base;
    }
    if (isNegative) {
        self->str[i++] = '-';
    }
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
        __global long *distancesOut,      // results 
        __global ulong *operationsOut,    // the operations to transform the haystack element into the needle
        __global char  *logOut,
        uint logLength,
        uint haystackSize
) { 
    levenstein_damerau_type self;
    
    int strLen = 255;
    char strAr[255];  // didn't have another way to allocate it, 
                      // and the param has to retain addr space
    char *str = (char*)&strAr;
    z(strAr,strLen);
    
    self.flags = flags;
    self.widthIn = widthIn;
    self.needleIn = needleIn;
    
    self.haystackIn = haystackIn;
    self.haystackSize = haystackSize;
    self.haystackRowIdx = get_global_id(0);

    self.str = str;
    self.strLen = strLen;
    self.logLength = logLength;
    self.logOut = logOut;
    self.logPos = self.haystackRowIdx * 2048;
    self.logOut [ self.logPos ] = 0x13;
    zg(self.logOut + self.logPos, 2048);
    
    self.needleIdx = 0;
    self.haystackIdx = 0;
    self.needleLast = 0;
    self.haystackLast = 0;
    self.distanceTotal = 0;
    self.needleCur = 0;
    self.haystackCur = 0;
    self.distancesOut = distancesOut;
    self.operationsOut = operationsOut;
    
    self.currentLocation = 0;
    
    aci(&self,0,"strLen:         ",self.strLen);
    aci(&self,0,"flags:          ",self.flags);
    aci(&self,0,"widthIn:        ",self.widthIn);

    aci(&self,0,"haystackSize:   ",self.haystackSize);
    aci(&self,0,"haystackRowIdx: ",self.haystackRowIdx);

    aci(&self,0,"logLength:      ",self.logLength);            
    aci(&self,0,"logPos:         ",self.logPos);
    ac(&self,0,"\n");

    int iterationCount = 0;
    while( self.needleIdx < self.widthIn ) {
        aci(&self,0,"iteration :",iterationCount);
        do {
            self.currentLocation = 0;
            if(self.haystackIdx>=self.widthIn) {
                ac(&self,0,"haystack ended...incrementing dist\n\n");
                self.distanceTotal++;
                self.needleIdx++;
                break;
            }
            if(self.needleIdx>=self.widthIn) {
                ac(&self,0,"needle ended...incrementing dist\n\n");
                self.distanceTotal++;
                self.haystackIdx++;
                break;
            }
            self.needleCur = self.needleIn[ self.needleIdx ];
            self.haystackCur = self.haystackIn[ ( self.widthIn * self.haystackRowIdx ) + self.haystackIdx ];
            
            append_state(&self,0);
            if(self.needleCur == self.haystackCur) {
                self.currentLocation = self.currentLocation | 1 << 3;
                if(self.haystackCur == self.needleLast) {
                    self.currentLocation = self.currentLocation | 1 << 2;
                    if(self.needleCur == self.haystackLast) {
                        self.currentLocation = self.currentLocation | 1 << 1;
                        if(self.needleLast == self.haystackLast) { 
                            self.currentLocation = self.currentLocation | 1;
                            // 1111 - error: never been here before
                            ac(&self,CL_LOG_TYPE_ERROR," - 1111 - continuing match\n");
                            /* eg. n:AA, h:AA */
                        } else { 
                            // 1110 - error: never been here before
                            ac(&self,CL_LOG_TYPE_ERROR," - 1110 - error: never been here before\n");
                            self.distancesOut[self.haystackRowIdx] = -1 * self.currentLocation;
                            return;
                        }
                    } else {
                        if(self.needleLast == self.haystackLast) { 
                            self.currentLocation = self.currentLocation | 1;
                            // 1101 - error: never been here before
                            ac(&self,CL_LOG_TYPE_ERROR," - 1101 - error: never been here before\n");
                            self.distancesOut[self.haystackRowIdx] = -1 * self.currentLocation;
                            return;
                        } else { 
                            // 1100 - error: never been here before
                            ac(&self,CL_LOG_TYPE_ERROR," - 1100 - continuing match\n");
                            /* eg. n:AA, h:0A */
                        }
                    }
                } else {
                    if(self.needleCur == self.haystackLast) { 
                        self.currentLocation = self.currentLocation | 1 << 1;
                        if(self.needleLast == self.haystackLast) { 
                            self.currentLocation = self.currentLocation | 1;
                            // 1011 - error: never been here before
                            ac(&self,CL_LOG_TYPE_ERROR," - 1011 - error: never been here before\n");
                            self.distancesOut[self.haystackRowIdx] = -1 * self.currentLocation;
                            return;
                        } else { 
                            // 1010 - possible insertion
                            ac(&self,0," - 1010 - possible insertion\n");
                            self.distanceTotal++;
                        }
                    } else { 
                        if(self.needleLast == self.haystackLast) {
                            self.currentLocation = self.currentLocation | 1;
                            // 1001 - continuing match
                            ac(&self,0," - 1001 - continuing match\n");
                        } else {
                            // 1000 - match restored
                            ac(&self,0," - 1000 - match restored\n");
                        }
                    }
                }
            } else {
                if(self.haystackCur == self.needleLast) {
                    self.currentLocation = self.currentLocation | 1 << 2;
                    if(self.needleCur == self.haystackLast) {
                        self.currentLocation = self.currentLocation | 1 << 1;
                        if(self.needleLast == self.haystackLast) {
                            self.currentLocation = self.currentLocation | 1;
                            // 0111 - error: never been here before
                            ac(&self,CL_LOG_TYPE_ERROR," - 0111 - error: never been here before\n");
                            self.distancesOut[self.haystackRowIdx] = -1 * self.currentLocation;
                            return;
                        } else {
                            // 0110 - transposition
                            ac(&self,0," - 0110 - transposition\n");
                            self.distanceTotal++;
                        }
                    } else {
                        if(self.needleLast == self.haystackLast) {
                            self.currentLocation = self.currentLocation | 1;
                            // 0101 - repeat haystack
                            ac(&self,0," - 0101 - repeat haystack\n");
                            self.distanceTotal++;
                        } else {
                            // 0100 - may be restoration of match
                            ac(&self,0," - 0100 - may be restoration of match\n");
                            self.needleIdx--;
                            break;
                        }
                    }
                } else {
                    if(self.needleCur == self.haystackLast) {
                        self.currentLocation = self.currentLocation | 1 << 1;
                        if(self.needleLast == self.haystackLast) {
                            self.currentLocation = self.currentLocation | 1;
                            // 0011 - replacement or deletion
                            ac(&self,0," - 0011 - replacement or deletion; i need to see this before i'll trust that i can get away with not incrementing distanceTotal\n");
                            self.distancesOut[self.haystackRowIdx] = -1 * self.currentLocation;
                            return;
                        } else {
                            // 0010 - restore match, last was actually ommission in haystack
                            ac(&self,0," - 0010 - restore match, last was actually ommission in haystack\n");
                            self.haystackIdx--;   // need to rewind the haystack
                            self.distanceTotal--; // and our distance wouldn't be accurate now either
                            break;
                        }
                    } else {
                        if(self.needleLast == self.haystackLast) {
                            self.currentLocation = self.currentLocation | 1;
                            // 0001 - break match, replacement
                            //logPos=log(haystackRowIdx,needleIdx,haystackIdx,str,strLen,)
                            ac(&self,0," - 0001 - break match, replacement\n");
                            self.distanceTotal++;
                        } else {
                            // 0000 - continue broken match, replacement
                            ac(&self,0," - 0000 - continue broken match, replacement\n");
                            self.distanceTotal++;
                        }
                    }
                }
            }
            
            self.needleIdx++;
            self.haystackIdx++;
            self.needleLast = self.needleCur;
            self.haystackLast = self.haystackCur;
            
        } while(false); // intended to provide a means of conveniently skipping last and idx updates just above.
        ac(&self,0,self.needleIdx<self.widthIn?"needleIdx<widthIn = true; \n":"needleIdx<widthIn = false\n");
        ac(&self,0,self.haystackIdx<self.widthIn?"haystackIdx<widthIn = true\n":"haystackIdx<widthIn = false\n");
        append_state(&self,0);
        ac(&self,0,"\n\n");
        iterationCount++;
    }
    aci(&self,0,"distanceTotal:  ",self.distanceTotal);
    self.distancesOut[self.haystackRowIdx] = self.distanceTotal;
}
);

const uint CL_LOG_ON         = 0b00000001;
const uint CL_LOG_ERROR_ONLY = 0b00000010;

LevensteinDamerau::LevensteinDamerau(boost::compute::context &context) 
        : m_logger(boost::log::keywords::channel="NLPGraph::Calc::LevensteinDamerau") {
    m_context = context;
    clLogOn = false;
    clLogErrorOnly = 0;
    boost::compute::device dev(m_context.get_device());
    m_commandQueue = boost::compute::command_queue(m_context, dev);
    
    int headerSize = sizeof(char)*strlen(kLevensteinDamerauOpenCLHeader);
    int sourceSize = sizeof(char)*strlen(kLevensteinDamerauOpenCLSource);
    int supportSize = sizeof(char)*strlen(kLevensteinDamerauOpenCLSupprtSource);
    
    char * source = (char *)malloc(headerSize+supportSize+sourceSize+1);
    
    memset(source,0,headerSize+sourceSize+supportSize+1);
    
    memcpy(source, kLevensteinDamerauOpenCLHeader, headerSize);
    memcpy(source+headerSize, kLevensteinDamerauOpenCLSupprtSource, supportSize);
    memcpy(source+headerSize+supportSize, kLevensteinDamerauOpenCLSource, sourceSize);
    
    LOG_I << "Source:\n" << source;
    
    // I would have used link, but NVIDIA doesn't support OpenCL 1.2
    // and this will prolly end up running on AWS hardware a bunch
    m_program = OpenCL::createAndBuildProgram(source,m_context);
    m_kernel = boost::compute::kernel(m_program, "calc_levenstein_damerau");
}
LevensteinDamerau::~LevensteinDamerau() {
}
int LevensteinDamerau::calculate(uint width, uint haystackSize, uint64_t* needle, uint64_t* haystack, int64_t *distancesOut, uint64_t *operationsOut) {

    int result = 0;
    uint64_t zero = 0;
    cl_int errcode = 0;
    
    LOG_I << "width         : " << width;
    LOG_I << "haystackSize  : " << haystackSize;
    LOG_I << "needle        : " << NLPGraph::Util::String::str(needle,width);
    LOG_I << "haystack:     : " << NLPGraph::Util::String::str(haystack,width*haystackSize);

    boost::compute::vector<uint64_t> device_needle(width, m_context);
    std::vector<uint64_t> host_needle(needle, needle+width);
    copy(host_needle.begin(), host_needle.end(), device_needle.begin(), m_commandQueue);
    
    boost::compute::vector<uint64_t> device_haystack(haystackSize*width ,m_context);
    std::vector<uint64_t> host_haystack(haystack, haystack+(haystackSize*width));
    boost::compute::copy(host_haystack.begin(), host_haystack.end(), device_haystack.begin(), m_commandQueue);
    
    boost::compute::vector<uint64_t> device_distances(haystackSize, (const uint64_t)&zero, m_commandQueue);
    int64_t host_distances[haystackSize*sizeof(int64_t)];
    memset(host_distances,0,haystackSize*sizeof(int64_t));
    cl_mem deviceDistancesBuf = clCreateBuffer(m_context,CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(int64_t)*haystackSize,&host_distances,&errcode);
    
    int count = (haystackSize*(width*2));
    uint64_t host_operations[count*sizeof(uint64_t)];
    memset(&host_operations,0,count*sizeof(uint64_t));
    cl_mem deviceOperationsBuf = clCreateBuffer(m_context,CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(uint64_t)*count,&host_operations,&errcode);
    
    uint logLength = 50000;
    char log[logLength];
    memset(&log,0,sizeof(char)*logLength);
    cl_mem logBuf = clCreateBuffer(m_context,CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(char)*logLength,&log,&errcode);
    
    uint flags =   (clLogOn        ? CL_LOG_ON         : 0) 
    	         | (clLogErrorOnly ? CL_LOG_ERROR_ONLY : 0);
    	
    m_kernel.set_arg(0,flags);
    m_kernel.set_arg(1,width);
    m_kernel.set_arg(2,device_needle);
    m_kernel.set_arg(3,device_haystack);
    m_kernel.set_arg(4,deviceDistancesBuf);
    m_kernel.set_arg(5,deviceOperationsBuf);
    m_kernel.set_arg(6,logBuf);
    m_kernel.set_arg(7,logLength);
    m_kernel.set_arg(8,haystackSize);
    
    m_commandQueue.enqueue_1d_range_kernel(m_kernel, 0, haystackSize, 1);

    errcode = clEnqueueReadBuffer(m_commandQueue, logBuf, true, 0, sizeof(char)*logLength, log, 0, NULL, NULL);
    if(errcode!=CL_SUCCESS) {
        OpenCLExceptionType except;
        except.msg = "Unable to read logBuf";
        throw except;
    }
    if(clLogOn) {
        LOG_I << "Run log:\n" << log;
    }
    
    clEnqueueReadBuffer(m_commandQueue, deviceDistancesBuf, true, 0, sizeof(uint64_t)*haystackSize, distancesOut, 0, NULL, NULL);
    
    clReleaseMemObject(logBuf);
    clReleaseMemObject(deviceOperationsBuf);
    clReleaseMemObject(deviceDistancesBuf);
    
    return result;
}

}}
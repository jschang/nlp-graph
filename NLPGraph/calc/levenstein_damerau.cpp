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

#define LOG_E BOOST_LOG_SEV((*m_logger),severity_level::critical) << __PRETTY_FUNCTION__ << " "
#define LOG_I BOOST_LOG_SEV((*m_logger),severity_level::normal) << __PRETTY_FUNCTION__ << " "

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

__constant ulong OP_INSERT    = 1;
__constant ulong OP_DELETE    = 2;
__constant ulong OP_REPEAT    = 3;
__constant ulong OP_TRANSPOSE = 4;
__constant ulong OP_REPLACE   = 5;

typedef struct {
    uint flags;
    int widthIn;                     // needle and each in haystack width
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
    int reconsIdx;
    int needleIdx;
    int haystackIdx;
    int lastIdxDiff;
    ulong needleLast;
    ulong haystackLast;
    ulong distanceTotal;
    uint needleCur;
    uint haystackCur;
    long currentLocation;
    ulong operationsOutIdx;
    ulong operationsOutEndIdx;
    ulong operationsOutStartIdx;
    uint lastWasLegitInsert;
    uint flagsModified;
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

inline uint is_last_op(levenstein_damerau_type *self, ulong op) {
    aci(self,0,"\tlastOp: ",self->operationsOut[self->operationsOutIdx-3]);            
    aci(self,0,"\tcmpOp : ",op);
    return self->operationsOut[self->operationsOutIdx-3] == op;
}

inline void add_op(levenstein_damerau_type *self, ulong op, ulong haystackCur) {
    if(op==OP_INSERT) {
        ac(self,0,"\tadd_op:      INSERT\n");
    } else if(op==OP_DELETE) {
        ac(self,0,"\tadd_op:      DELETE\n");
    } else if(op==OP_REPEAT) {
        ac(self,0,"\tadd_op:      REPEAT\n");
    } else if(op==OP_TRANSPOSE) {
        ac(self,0,"\tadd_op:      TRANSPOSE\n");
    } else if(op==OP_REPLACE) {
        ac(self,0,"\tadd_op:      REPLACE\n");
    }
    aci(self,0,"\treconsIdx:   ",self->reconsIdx);            
    aci(self,0,"\thaystackCur: ",haystackCur);
    self->distanceTotal++;
    self->operationsOut[self->operationsOutIdx++] = op;
    self->operationsOut[self->operationsOutIdx++] = self->reconsIdx;
    self->operationsOut[self->operationsOutIdx++] = haystackCur;
}

inline void replace_op(levenstein_damerau_type *self, ulong op, ulong haystackCur) {
    if(op==OP_INSERT) {
        ac(self,0,"\treplace_op:  INSERT\n");
    } else if(op==OP_DELETE) {
        ac(self,0,"\treplace_op:  DELETE\n");
    } else if(op==OP_REPEAT) {
        ac(self,0,"\treplace_op:  REPEAT\n");
    } else if(op==OP_TRANSPOSE) {
        ac(self,0,"\treplace_op:  TRANSPOSE\n");
    } else if(op==OP_REPLACE) {
        ac(self,0,"\treplace_op:  REPLACE\n");
    }
    self->operationsOutIdx -= 3;
    self->operationsOut[self->operationsOutIdx++] = op;
    aci(self,0,"\treconsIdx:   ",self->operationsOut[self->operationsOutIdx]);
    aci(self,0,"\thaystackCur: ",haystackCur);
    self->operationsOut[self->operationsOutIdx++] = self->operationsOut[self->operationsOutIdx];
    self->operationsOut[self->operationsOutIdx++] = haystackCur;
}

inline void forward_needle(levenstein_damerau_type *self) {
    ac(self,0,"\tforwarding needle\n");
    self->needleIdx++;
    self->needleCur = self->needleIdx >= 0 ? self->needleIn[ self->needleIdx ] : 0;
    self->needleLast = self->needleIdx >= 1 ? self->needleIn[ self->needleIdx - 1] : 0;
}

inline void rewind_needle(levenstein_damerau_type *self) {
    ac(self,0,"\trewinding needle\n");
    self->needleIdx--;
    self->needleCur = self->needleIdx >= 0 ? self->needleIn[ self->needleIdx ] : 0;
    self->needleLast = self->needleIdx >= 1 ? self->needleIn[ self->needleIdx - 1] : 0;
}

inline void rewind_haystack(levenstein_damerau_type *self) {
    ac(self,0,"\trewinding haystack\n");
    self->haystackIdx--;   // need to rewind the haystack
    self->haystackCur = self->haystackIdx >= 0 ? self->haystackIn[ ( self->widthIn * self->haystackRowIdx ) + self->haystackIdx ] : 0;
    self->haystackLast = self->haystackIdx >= 1 ? self->haystackIn[ ( self->widthIn * self->haystackRowIdx ) + self->haystackIdx - 1] : 0;
}


__kernel void calc_levenstein_damerau(
        uint flags,
        uint widthIn,                  // needle and each in haystack width
        __constant ulong *needleIn,    // needle uint64_t's 
        __global ulong *haystackIn,    // haystack uint64_t's 
        __global long *distancesOut,   // results 
        __global ulong *operationsOut, // the operations to transform the haystack element into the needle
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
    
    self.lastWasLegitInsert = 0;
    self.flagsModified = 0;
    
    self.reconsIdx = 0;
    self.needleIdx = 0;
    self.haystackIdx = 0;
    self.lastIdxDiff = 0;
    self.needleLast = 0;
    self.haystackLast = 0;
    self.distanceTotal = 0;
    self.needleCur = 0;
    self.haystackCur = 0;
    self.distancesOut = distancesOut;
    
    self.operationsOut = operationsOut;
    self.operationsOutStartIdx = self.haystackRowIdx * (self.widthIn * 3);
    self.operationsOutEndIdx = self.operationsOutStartIdx + (self.widthIn * 3);
    self.operationsOutIdx = self.operationsOutStartIdx;
    
    self.currentLocation = 0;
    
    aci(&self,0,"strLen:         ",self.strLen);
    aci(&self,0,"flags:          ",self.flags);
    aci(&self,0,"widthIn:        ",self.widthIn);

    aci(&self,0,"haystackSize:   ",self.haystackSize);
    aci(&self,0,"haystackRowIdx: ",self.haystackRowIdx);

    aci(&self,0,"logLength:      ",self.logLength);            
    aci(&self,0,"logPos:         ",self.logPos);
    ac(&self,0,"\n");

    /*
    __constant ulong OP_INSERT    = 1;
    __constant ulong OP_DELETE    = 2;
    __constant ulong OP_REPEAT    = 3;
    __constant ulong OP_TRANSPOSE = 4;
    __constant ulong OP_REPLACE   = 5;
    */

    int iterationCount = 0;
    while( self.needleIdx < self.widthIn ) {
        aci(&self,0,"iteration :",iterationCount);
        do {
            self.currentLocation = 0;
            if(self.haystackIdx>=self.widthIn) {
                ac(&self,0,"haystack ended...incrementing dist\n\n");
                add_op(&self, OP_DELETE, 0);
                self.distanceTotal++;
                self.needleIdx++;
                self.reconsIdx++;
                break;
            }
            if(self.needleIdx>=self.widthIn) {
                ac(&self,0,"needle ended...incrementing dist\n\n");
                add_op(&self, OP_INSERT, self.haystackIn[self.haystackIdx]);
                self.distanceTotal++;
                self.haystackIdx++;
                self.reconsIdx++;
                break;
            }
            self.needleCur = self.needleIn[ self.needleIdx ];
            self.haystackCur = self.haystackIn[ ( self.widthIn * self.haystackRowIdx ) + self.haystackIdx ];
            if(self.needleIdx>0) self.needleLast = self.needleIn[ self.needleIdx - 1 ];
            else self.needleLast = self.needleCur;
            if(self.haystackIdx>0) self.haystackLast = self.haystackIn[ ( self.widthIn * self.haystackRowIdx ) + self.haystackIdx - 1 ];
            else self.haystackLast = self.haystackCur;
            
            append_state(&self,0);
            if(self.needleCur == self.haystackCur) { // n:?A, h:?A
                // these are all current matches
                self.currentLocation = self.currentLocation | 1 << 3;
                if(self.haystackCur == self.needleLast) { // n:AA, h:?A
                    self.currentLocation = self.currentLocation | 1 << 2;
                    if(self.needleCur == self.haystackLast) { // n:AA, h:AA
                        self.currentLocation = self.currentLocation | 1 << 1;
                        if(self.needleLast == self.haystackLast) { 
                            self.currentLocation = self.currentLocation | 1;
                            ac(&self,CL_LOG_TYPE_ERROR," - cmp_1111 - continuing match w/ repetition; n:AA, h:AA\n");
                        } else {
                            ac(&self,CL_LOG_TYPE_ERROR," - cmp_1110 - error: not logically possible\n");
                            self.distancesOut[self.haystackRowIdx] = -1 * self.currentLocation;
                            return;
                        }
                    } else { // n:AA, h:AA
                        if(self.needleLast == self.haystackLast) { 
                            self.currentLocation = self.currentLocation | 1;
                            ac(&self,CL_LOG_TYPE_ERROR," - cmp_1101 - error: not logically possible\n");
                            self.distancesOut[self.haystackRowIdx] = -1 * self.currentLocation;
                            return;
                        } else {
                            ac(&self,CL_LOG_TYPE_ERROR," - cmp_1100 - repetition in needle, rejoining match; n:AA, h:BA\n");
                            // last would have incremented distanceTotal
                        }
                    }
                } else { // n:BA, h:?A
                    if(self.needleCur == self.haystackLast) { // n:BA, h:AA
                        self.currentLocation = self.currentLocation | 1 << 1;
                        if(self.needleLast == self.haystackLast) { 
                            self.currentLocation = self.currentLocation | 1;
                            ac(&self,CL_LOG_TYPE_ERROR," - cmp_1011 - error: not logically possible\n");
                            self.distancesOut[self.haystackRowIdx] = -1 * self.currentLocation;
                            return;
                        } else { // n:BA, h:AA
                            // 1010 - possible insertion
                            ac(&self,0," - cmp_1010 - previous was replacement, match restored; n:BA, h:AA\n");
                            if(is_last_op(&self,OP_INSERT) || is_last_op(&self,OP_REPEAT)) {
                                self.lastWasLegitInsert = 1;
                                self.flagsModified = 1;
                                replace_op(&self, OP_REPLACE, self.haystackLast);
                                forward_needle(&self);
                            }
                        }
                    } else { 
                        if(self.needleLast == self.haystackLast) {
                            self.currentLocation = self.currentLocation | 1;
                            ac(&self,0," - cmp_1001 - continuing match; n:BA, h:BA\n");
                        } else {
                            // when the last is a legitimate insert, this breaks the operations
                            ac(&self,0," - cmp_1000 - match restored, last is likely a replacement; n:BA, h:CA\n");
                            if(!is_last_op(&self,OP_TRANSPOSE) && !is_last_op(&self,OP_DELETE) && !self.lastWasLegitInsert) {
                                replace_op(&self, OP_REPLACE, self.haystackLast);
                            }
                        }
                    }
                }
            } else { // n:?A, h:?B
                // these are all current match failures
                if(self.haystackCur == self.needleLast) { // n:BA, h:?B
                    self.currentLocation = self.currentLocation | 1 << 2;
                    if(self.needleCur == self.haystackLast) { 
                        self.currentLocation = self.currentLocation | 1 << 1;
                        if(self.needleLast == self.haystackLast) {
                            self.currentLocation = self.currentLocation | 1;
                            ac(&self,CL_LOG_TYPE_ERROR," - cmp_0111 - error: not logically possible\n");
                            self.distancesOut[self.haystackRowIdx] = -1 * self.currentLocation;
                            return;
                        } else {
                            ac(&self,0," - cmp_0110 - recognizing transposition; n:AB, h:BA\n");
                            // and change the last operation to a transpose    
                            replace_op(&self, OP_TRANSPOSE, 0);
                        }
                    } else {
                        if(self.needleLast == self.haystackLast) {
                            self.currentLocation = self.currentLocation | 1;
                            ac(&self,0," - cmp_0101 - current is repetition in haystack; n:AB, h:AA\n");
                            rewind_needle(&self);
                            // repetition does not replace the last operation...they next will just have the same idx
                            add_op(&self, OP_REPEAT, 0);
                            break;
                        } else {
                            ac(&self,0," - cmp_0100 - last was insertion in haystack; n:AB, h:CA\n");
                            self.lastWasLegitInsert = 1;
                            self.flagsModified = 1;
                            rewind_needle(&self);
                            break;
                        }
                    }
                } else {
                    if(self.needleCur == self.haystackLast) {
                        self.currentLocation = self.currentLocation | 1 << 1;
                        if(self.needleLast == self.haystackLast) {
                            self.currentLocation = self.currentLocation | 1;
                            ac(&self,0," - cmp_0011 - replacement or deletion; n:AA, h:AB\n");
                            add_op(&self,OP_INSERT,self.haystackCur);
                        } else {
                            // 0010 - restore match, last was actually ommission in haystack
                            ac(&self,0," - cmp_0010 - last was actually ommission in haystack; n:CA, h:AB\n");
                            rewind_haystack(&self);
                            // and change the last operation to a delete    
                            replace_op(&self,OP_DELETE,0);
                            break;
                        }
                    } else {
                        if(self.needleLast == self.haystackLast) {
                            self.currentLocation = self.currentLocation | 1;
                            ac(&self,0," - cmp_0001 - break match, replacement;  n:AC, h:AB\n");
                            add_op(&self,OP_REPLACE,self.haystackCur);
                        } else {
                            ac(&self,0," - cmp_0000 - continue broken match, replacement; n:AB, h:CD\n");
                            add_op(&self,OP_REPLACE,self.haystackCur);
                        }
                    }
                }
            }
            
            if(!self.flagsModified) {
                self.lastWasLegitInsert = 0;
            }
            self.flagsModified = 0;
            self.needleIdx++;
            self.reconsIdx++;
            self.haystackIdx++;
            self.lastIdxDiff = abs(self.needleIdx - self.haystackIdx);
            
        } while(false); // intended to provide a means of conveniently skipping last and idx updates just above.
        ac(&self, 0, self.needleIdx < self.widthIn ? "needleIdx<widthIn = true; \n" : "needleIdx<widthIn = false\n");
        ac(&self, 0, self.haystackIdx < self.widthIn ? "haystackIdx<widthIn = true\n" : "haystackIdx<widthIn = false\n");
        append_state(&self,0);
        ac(&self, 0, "\n\n");
        iterationCount++;
    }
    aci(&self,0,"distanceTotal:  ",self.distanceTotal);
    self.distancesOut[self.haystackRowIdx] = self.distanceTotal;
}

__kernel void recons_levenstein_damerau(
        uint flags,
        uint widthIn,                 // the width of each haystack sequence
        uint operationsWidth,         // the width of each operations sequence
        __global ulong *haystackIn,   // source ids that the operations will transform
        __global ulong *operationsIn, // the operations to transform the haystack element into the needle
        __global ulong *resultOut     // reconstructed sequences using operations  
) {
    ulong opIdxStart  = operationsWidth * get_global_id(0);
    ulong haystackIdx = widthIn         * get_global_id(0);
    ulong curOpIdx    = opIdxStart;
    ulong curHayIdx   = haystackIdx;
    ulong endHayIdx   = haystackIdx     + widthIn;
    ulong curResIdx   = curHayIdx;
    while(curHayIdx<endHayIdx) {
        ulong curOp        = operationsIn [ curOpIdx + opIdxStart     ];
        ulong curOpHayIdx  = operationsIn [ curOpIdx + opIdxStart + 1 ];
        ulong curOptInsId  = operationsIn [ curOpIdx + opIdxStart + 2 ];
        if(curOp && curOpHayIdx==curHayIdx) {
            if(curOp==OP_INSERT) {
                resultOut[curResIdx++] = curOptInsId;
            } else if(curOp==OP_REPLACE) {
                resultOut[curResIdx++] = curOptInsId;
                curHayIdx ++;
            } else if(curOp==OP_DELETE) {
                curHayIdx ++;
            } else if(curOp==OP_REPEAT) {
                resultOut[curResIdx++] = haystackIn [ curOpHayIdx   ];
            } else if(curOp==OP_TRANSPOSE) {
                resultOut[curResIdx++] = haystackIn [ curHayIdx + 1 ];
                resultOut[curResIdx++] = haystackIn [ curHayIdx     ];
                curHayIdx += 2;
            } 
            curOpIdx += 3;
        } else {
            resultOut[curResIdx++]     = haystackIn [ curHayIdx ++  ];
        }
    }
}

);

const uint CL_LOG_ON         = 0b00000001;
const uint CL_LOG_ERROR_ONLY = 0b00000010;

LevensteinDamerau::LevensteinDamerau(const boost::compute::context &context) {

    m_logger.reset(new Util::LoggerType(boost::log::keywords::channel="NLPGraph::Calc::LevensteinDamerau"));
    
    m_context.reset(new boost::compute::context(context));
    
    clLogOn = false;
    clLogErrorOnly = 0;
    boost::compute::device dev = boost::compute::device(m_context->get_device());
    
    m_commandQueue.reset(new boost::compute::command_queue(*m_context, dev));
    
    int headerSize = sizeof(char)*strlen(kLevensteinDamerauOpenCLHeader);
    int sourceSize = sizeof(char)*strlen(kLevensteinDamerauOpenCLSource);
    int supportSize = sizeof(char)*strlen(kLevensteinDamerauOpenCLSupprtSource);
    
    char * source = 0;
    try {
        source = (char *)malloc(headerSize+supportSize+sourceSize+1);
    
        memset(source,0,headerSize+sourceSize+supportSize+1);
        
        memcpy(source, kLevensteinDamerauOpenCLHeader, headerSize);
        memcpy(source+headerSize, kLevensteinDamerauOpenCLSupprtSource, supportSize);
        memcpy(source+headerSize+supportSize, kLevensteinDamerauOpenCLSource, sourceSize);
        
        // LOG_I << "Source:\n" << source;
        
        // I would have used link, but NVIDIA doesn't support OpenCL 1.2
        // and this will prolly end up running on AWS hardware a bunch
        boost::compute::program p = OpenCL::createAndBuildProgram(source,*m_context);
        m_program.reset(new boost::compute::program(p));
        m_kernelCalc.reset(new boost::compute::kernel(*m_program, "calc_levenstein_damerau"));
        m_kernelRecons.reset(new boost::compute::kernel(*m_program, "recons_levenstein_damerau"));
        
        delete source;
    } catch(...) {
        if(source!=0) delete source;
        throw;
    }
}
int LevensteinDamerau::reconstruct(LevensteinDamerauReconstructDataPtr data) {

    data->zeroResult(*m_commandQueue);
    
    int      result          = 0;
    
    OpenCLExceptionType except;
    except.msg = "";
        
    uint flags =   (clLogOn        ? CL_LOG_ON         : 0) 
                 | (clLogErrorOnly ? CL_LOG_ERROR_ONLY : 0);
                 
    m_kernelRecons->set_arg(0,flags);
    m_kernelRecons->set_arg(1,data->getNeedleWidth());
    m_kernelRecons->set_arg(2,data->getOperationWidth());
    m_kernelRecons->set_arg(3,data->clHaystack());
    m_kernelRecons->set_arg(4,data->clOperations());
    m_kernelRecons->set_arg(5,data->clResult());
    
    m_commandQueue->enqueue_1d_range_kernel(*m_kernelRecons, 0, data->getHaystackCount(), 1);
    
    data->read(*m_commandQueue);
    
    if(clLogOn) {
        LOG_I << "Run log:\n" << log;
    }
    
    if(except.msg.length()>0) {
        throw except;
    }  
    return result;
}
int LevensteinDamerau::calculate(LevensteinDamerauDataPtr data) {
    
    data->zeroOperations(*m_commandQueue);
    data->zeroDistances(*m_commandQueue);
    
    uint logLength      = kLevensteinLogLength;

    int result = 0;
    
    OpenCLExceptionType except;
    except.msg = "";
    
    char * log                 = 0;
    cl_mem logBuf              = 0;
    
    LOG_I << "needleWidth   : " << data->needleWidth();
    LOG_I << "haystackSize  : " << data->haystackCount();

    try {
        OpenCL::alloc<char>(*m_context, logLength, (char **)&log, (cl_mem*)&logBuf, (int)CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR);
        
        uint flags =   (clLogOn        ? CL_LOG_ON         : 0) 
                     | (clLogErrorOnly ? CL_LOG_ERROR_ONLY : 0);
            
        m_kernelCalc->set_arg(0,flags);
        m_kernelCalc->set_arg(1,data->needleWidth());
        m_kernelCalc->set_arg(2,data->clNeedle());
        m_kernelCalc->set_arg(3,data->clHaystack());
        m_kernelCalc->set_arg(4,data->clDistances());
        m_kernelCalc->set_arg(5,data->clOperations());
        m_kernelCalc->set_arg(6,logBuf);
        m_kernelCalc->set_arg(7,logLength);
        m_kernelCalc->set_arg(8,data->haystackCount());
        
        size_t gwo[2] = {0,0};
        size_t gws[2] = {data->haystackCount(),data->needleCount()};
        size_t lws[2] = {1,1};
        m_commandQueue->enqueue_nd_range_kernel(*m_kernelCalc, 2, (size_t*)&gwo, (size_t*)&gws, (size_t*)&lws, boost::compute::wait_list());
        
        OpenCL::read<char>(*m_commandQueue, logLength, log, logBuf);
        data->read(*m_commandQueue);
        
        if(clLogOn) {
            LOG_I << "Run log:\n" << log;
        }
        
        if(except.msg.length()>0) {
            throw except;
        }
        
        delete log;
        clReleaseMemObject (logBuf);

    } catch(...) {
    
        if(!log)                 delete log;
        if(!logBuf)              clReleaseMemObject (logBuf);
        
        throw;
    }    
    return result;
}

}}
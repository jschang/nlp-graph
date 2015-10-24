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

namespace NLPGraph {
namespace Calc {

extern const char *kLevensteinDamerauOpenCLSource;

class LevensteinDamerauData {
private:
    uint m_needleWidth = 0;
    uint m_haystackSize = 0;
    uint64_t* m_needle = 0;
    uint64_t* m_haystack = 0;
    uint64_t* m_operations = 0;
    uint m_operationsSize = 0;
    int64_t* m_distances = 0;
private:
    void alloc(uint needleWidth, uint haystackSize) {
    
        m_needleWidth = needleWidth;
        m_haystackSize = haystackSize;
        m_operationsSize = haystackSize * this->getOperationWidth();
        
        m_needle = new uint64_t[needleWidth];
        this->zeroNeedle();
        
        m_haystack = new uint64_t[needleWidth*haystackSize];
        this->zeroHaystack();
        
        m_distances = new int64_t[haystackSize];
        this->zeroDistances();
        
        m_operations = new uint64_t[m_operationsSize];
        this->zeroOperations();
    }
public:
    LevensteinDamerauData(uint needleWidth, uint haystackSize) {
        this->alloc(needleWidth,haystackSize);
    }
    LevensteinDamerauData(uint needleWidth, uint haystackSize, uint64_t* needle, uint64_t* haystack) {
        this->alloc(needleWidth, haystackSize);
        memcpy(m_needle,needle,sizeof(uint64_t)*needleWidth);
        memcpy(m_haystack,haystack,sizeof(uint64_t)*needleWidth*haystackSize);
    }
    ~LevensteinDamerauData() {
        delete m_needle;
        delete m_haystack;
        delete m_distances;
        delete m_operations;
    }
    void zeroNeedle() {
        memset(m_needle,0,sizeof(uint64_t)*m_needleWidth);
    }
    void zeroHaystack() {
        memset(m_haystack,0,sizeof(uint64_t)*m_needleWidth*m_haystackSize);
    }
    void zeroOperations() {
        memset(m_operations,0,sizeof(uint64_t)*m_operationsSize);
    }
    void zeroDistances() {
        memset(m_distances,0,sizeof(int64_t)*m_haystackSize);
    }
    uint getNeedleWidth() {
        return m_needleWidth;
    }
    uint getHaystackSize() {
        return m_haystackSize;
    }
    uint64_t* getNeedle() {
        return m_needle;
    }
    uint64_t* getHaystack() {
        return m_haystack;
    }
    int64_t* getDistances() {
        return m_distances;
    }
    uint getOperationWidth() {
        return (3*m_needleWidth);
    }
    uint getOperationsSize() {
        return m_operationsSize;
    }
    uint64_t* getOperations() {
        return m_operations;
    }
};

class LevensteinDamerau {
private:
    boost::compute::context       m_context;
    boost::compute::kernel        m_kernel;
    boost::compute::program       m_program;
    boost::compute::command_queue m_commandQueue;
    uint64_t*                     m_needle;
    uint64_t**                    m_haystack;
    unsigned int                  m_haystackSize;
    Util::LoggerType              m_logger;
public:
    bool clLogOn;
    bool clLogErrorOnly;
public:
    LevensteinDamerau(boost::compute::context &context);
    ~LevensteinDamerau();
    int calculate(LevensteinDamerauDataPtr data);
    int reconstruct(LevensteinDamerauDataPtr data);
};

}}

#endif /* defined(__NLPGraph__levenstein_damerau__) */

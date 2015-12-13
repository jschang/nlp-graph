//
//  levenstein_damerau_reconstruct_data.cpp
//  NLPGraph
//
//  Created by Jonathan Schang on 11/24/15.
//
//

#include "levenstein_damerau_reconstruct_data.h"
#include "../levenstein_damerau.h"

namespace NLPGraph {
namespace Calc {

LevensteinDamerauReconstructData::LevensteinDamerauReconstructData(uint needleWidth, uint haystackSize) {
    this->alloc(needleWidth, haystackSize);
}
LevensteinDamerauReconstructData::LevensteinDamerauReconstructData(uint needleWidth, uint haystackSize, uint64_t* operations, uint64_t* haystack) {
    this->alloc(needleWidth, haystackSize);
    memcpy( m_operations, operations, sizeof(uint64_t) * m_operationsSize );
    memcpy( m_haystack,   haystack,   sizeof(uint64_t) * needleWidth * haystackSize );
}
LevensteinDamerauReconstructData::~LevensteinDamerauReconstructData() {
    delete m_haystack;
    delete m_operations;
    delete m_result;
}
void LevensteinDamerauReconstructData::alloc(uint needleWidth, uint haystackCount) {

    m_needleWidth    = needleWidth;
    m_haystackCount  = haystackCount;
    m_operationsSize 
        = haystackCount 
            * kLevensteinOperationsWidth 
            * this->getOperationWidth();
    
    m_result     = new uint64_t [ m_needleWidth * m_haystackCount ];
    m_haystack   = new uint64_t [ m_needleWidth * m_haystackCount ];
    m_operations = new uint64_t [ m_operationsSize ];

    this->zeroResult();
    this->zeroHaystack();
    this->zeroOperations();
}

void LevensteinDamerauReconstructData::zeroResult() {
    memset(m_result,     0, sizeof(uint64_t) * m_needleWidth * m_haystackCount );
}
void LevensteinDamerauReconstructData::zeroHaystack() {
    memset(m_haystack,   0, sizeof(uint64_t) * m_needleWidth * m_haystackCount );
}
void LevensteinDamerauReconstructData::zeroOperations() {
    memset(m_operations, 0, sizeof(uint64_t) * m_needleWidth * m_haystackCount );
}

uint LevensteinDamerauReconstructData::getNeedleWidth() {
    return m_needleWidth;
}
uint LevensteinDamerauReconstructData::getHaystackCount() {
    return m_haystackCount;
}
uint LevensteinDamerauReconstructData::getHaystackSize() {
    return m_haystackCount * m_needleWidth;
}
uint LevensteinDamerauReconstructData::getOperationsSize() {
    return m_operationsSize;
}
uint LevensteinDamerauReconstructData::getOperationWidth() {
    return kLevensteinOperationsWidthMultiplier * m_needleWidth;
}

uint64_t* LevensteinDamerauReconstructData::getResult() {
    return m_result;
}
uint64_t* LevensteinDamerauReconstructData::getHaystack() {
    return m_haystack;
}
uint64_t* LevensteinDamerauReconstructData::getOperations() {
    return m_operations;
}

}}
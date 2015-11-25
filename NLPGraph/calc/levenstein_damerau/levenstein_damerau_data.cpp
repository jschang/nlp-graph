//
//  levenstein_damerau_data.cpp
//  NLPGraph
//
//  Created by Jonathan Schang on 11/24/15.
//
//

#include "levenstein_damerau_data.h"

namespace NLPGraph {
namespace Calc {

LevensteinDamerauData::LevensteinDamerauData(uint needleWidth, uint haystackSize) {
    this->alloc(needleWidth,haystackSize);
}
LevensteinDamerauData::LevensteinDamerauData(uint needleWidth, uint haystackSize, uint64_t* needle, uint64_t* haystack) {
    this->alloc(needleWidth, haystackSize);
    memcpy(m_needle,needle,sizeof(uint64_t)*needleWidth);
    memcpy(m_haystack,haystack,sizeof(uint64_t)*needleWidth*haystackSize);
}
LevensteinDamerauData::~LevensteinDamerauData() {
    delete m_needle;
    delete m_haystack;
    delete m_distances;
    delete m_operations;
}
void LevensteinDamerauData::zeroNeedle() {
    memset(m_needle,0,sizeof(uint64_t)*m_needleWidth);
}
void LevensteinDamerauData::zeroHaystack() {
    memset(m_haystack,0,sizeof(uint64_t)*m_needleWidth*m_haystackSize);
}
void LevensteinDamerauData::zeroOperations() {
    memset(m_operations,0,sizeof(uint64_t)*m_operationsSize);
}
void LevensteinDamerauData::zeroDistances() {
    memset(m_distances,0,sizeof(int64_t)*m_haystackSize);
}
uint LevensteinDamerauData::getNeedleWidth() {
    return m_needleWidth;
}
uint LevensteinDamerauData::getHaystackSize() {
    return m_haystackSize;
}
uint64_t* LevensteinDamerauData::getNeedle() {
    return m_needle;
}
uint64_t* LevensteinDamerauData::getHaystack() {
    return m_haystack;
}
int64_t* LevensteinDamerauData::getDistances() {
    return m_distances;
}
uint LevensteinDamerauData::getOperationWidth() {
    return (3*m_needleWidth);
}
uint LevensteinDamerauData::getOperationsSize() {
    return m_operationsSize;
}
uint64_t* LevensteinDamerauData::getOperations() {
    return m_operations;
}
void LevensteinDamerauData::alloc(uint needleWidth, uint haystackSize) {

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

}}
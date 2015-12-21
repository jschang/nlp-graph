//
//  levenstein_damerau_data.cpp
//  NLPGraph
//
//  Created by Jonathan Schang on 11/24/15.
//
//

#include "levenstein_damerau_data.h"
#include "../levenstein_damerau.h"

namespace NLPGraph {
namespace Calc {

LevensteinDamerauData::LevensteinDamerauData(cl_context clContext, uint needleWidth, uint needleCount, uint haystackCount, uint64_t* needle, uint64_t* haystack) {
    try {
        m_needleWidth     = needleWidth;
        m_haystackCount   = haystackCount;
        m_operationsCount = haystackCount * needleCount;
        m_needleCount     = needleCount;
        Util::OpenCL::alloc<uint64_t>(clContext, this->haystackCount() * this->needleWidth(), &haystack, &_clHaystack, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR);
        Util::OpenCL::alloc<uint64_t>(clContext, this->needleCount() * this->needleWidth(), &needle, &_clNeedle, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR);
        Util::OpenCL::alloc<uint64_t>(clContext, this->operationsCount() * this->operationWidth(), &_operations, &_clOperations, (int)CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR);
        Util::OpenCL::alloc <int64_t>(clContext, this->haystackCount(), &_distances, &_clDistances, (int)CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR);
    } catch(...) {
        this->free();
        throw;
    }
}
LevensteinDamerauData::~LevensteinDamerauData() {
    this->free();
}
void LevensteinDamerauData::free() {
    if(_clHaystack!=0) {
        clReleaseMemObject(_clHaystack);
    }
    if(_clNeedle!=0) {
        clReleaseMemObject(_clNeedle);
    }
    if(_clOperations!=0) {
        clReleaseMemObject(_clOperations);
    }
    if(_operations!=0) {
        delete _operations;
    }
    if(_clDistances!=0) {
        clReleaseMemObject(_clDistances);
    }
    if(_distances!=0) {
        delete _distances;
    }
}

uint LevensteinDamerauData::needleWidth() {
    return m_needleWidth;
}
uint LevensteinDamerauData::operationWidth() {
    return ( kLevensteinOperationsWidthMultiplier * m_needleWidth );
}
uint LevensteinDamerauData::operationsCount() {
    return m_operationsCount;
}
uint LevensteinDamerauData::haystackCount() {
    return m_haystackCount;
}
uint LevensteinDamerauData::needleCount() {
    return m_needleCount;
}

void LevensteinDamerauData::zeroDistances(cl_command_queue commandQueue) {
    memset(_distances,0,sizeof(int64_t) * haystackCount());
    Util::OpenCL::write<int64_t>(commandQueue, haystackCount(), _distances, _clDistances);
}
void LevensteinDamerauData::zeroOperations(cl_command_queue commandQueue) {
    memset(_operations,0,sizeof(uint64_t) * operationsCount() * operationWidth());
    Util::OpenCL::write<uint64_t>(commandQueue, operationsCount() * operationWidth(), _operations, _clOperations);
}

void LevensteinDamerauData::read(cl_command_queue commandQueue) {
    Util::OpenCL::read<int64_t>(commandQueue, haystackCount(), _distances,  _clDistances);
    Util::OpenCL::read<uint64_t>(commandQueue, operationsCount() * operationWidth(), _operations, _clOperations);
}

uint64_t* LevensteinDamerauData::operations() { return _operations; }
int64_t* LevensteinDamerauData::distances() { return _distances; }

cl_mem LevensteinDamerauData::clNeedle() {
    return _clNeedle;
}
cl_mem LevensteinDamerauData::clHaystack() {
    return _clHaystack;
}
cl_mem LevensteinDamerauData::clDistances() {
    return _clDistances;
}
cl_mem LevensteinDamerauData::clOperations() {
    return _clOperations;
}

}}
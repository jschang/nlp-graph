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

LevensteinDamerauReconstructData::LevensteinDamerauReconstructData(cl_context context, uint needleWidth, uint haystackSize) {
    this->alloc(context, needleWidth, haystackSize);
}
LevensteinDamerauReconstructData::LevensteinDamerauReconstructData(cl_context context, uint needleWidth, uint haystackSize, uint64_t* operations, uint64_t* haystack) {
    this->alloc(context, needleWidth, haystackSize, haystack, operations);
}
LevensteinDamerauReconstructData::~LevensteinDamerauReconstructData() {
    this->free();
}
void LevensteinDamerauReconstructData::free() {
    if(m_haystack!=0) {
        delete m_haystack;
    }
    if(m_operations!=0) {
        delete m_operations;
    }
    if(m_result!=0) {
        delete m_result;
    }
    if(_clResult!=0) {
        clReleaseMemObject(_clResult);
    }
    if(_clHaystack!=0) {
        clReleaseMemObject(_clHaystack);
    }
    if(_clOperations!=0) {
        clReleaseMemObject(_clOperations);
    }
}
void LevensteinDamerauReconstructData::alloc(cl_context context, uint needleWidth, uint haystackCount, uint64_t* haystack, uint64_t* operations) {

    m_needleWidth    = needleWidth;
    m_haystackCount  = haystackCount;
    m_operationsSize 
        = haystackCount 
            * kLevensteinOperationsWidth 
            * this->getOperationWidth();
    
    m_result     = new uint64_t [ m_needleWidth * m_haystackCount ];
    memset(m_result,     0, sizeof(uint64_t) * m_needleWidth * m_haystackCount );
    m_haystack   = new uint64_t [ m_needleWidth * m_haystackCount ];
    if(haystack) {
        memcpy(m_haystack, haystack, sizeof(uint64_t) * m_needleWidth * m_haystackCount);
    } else {
        memset(m_haystack, 0, sizeof(uint64_t) * m_needleWidth * m_haystackCount );
    }
    m_operations = new uint64_t [ m_operationsSize ];
    if(operations) {
        memcpy(m_operations, operations, sizeof(uint64_t) * m_needleWidth * m_haystackCount);
    } else {
        memset(m_operations, 0, sizeof(uint64_t) * m_needleWidth * m_haystackCount );
    }
    
    Util::OpenCL::alloc<uint64_t>(context, getHaystackSize(), &m_haystack,   
            (cl_mem*)&_clHaystack,  (int)CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR);
    Util::OpenCL::alloc<uint64_t>(context, getOperationsSize(), &m_operations, 
            (cl_mem*)&_clOperations,(int)CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR);
    Util::OpenCL::alloc<uint64_t>(context, getHaystackSize(), &m_result,   
            (cl_mem*)&_clResult,   (int)CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR);

}

void LevensteinDamerauReconstructData::read(cl_command_queue commandQueue) {
    Util::OpenCL::read<uint64_t>(commandQueue, m_needleWidth * m_haystackCount, m_result, _clResult);
}

void LevensteinDamerauReconstructData::zeroResult(cl_command_queue commandQueue) {
    memset(m_result, 0, sizeof(uint64_t) * m_needleWidth * m_haystackCount );
    Util::OpenCL::write<uint64_t>(commandQueue, m_needleWidth * m_haystackCount, m_result, _clResult);
}
void LevensteinDamerauReconstructData::zeroHaystack(cl_command_queue commandQueue) {
    memset(m_haystack,   0, sizeof(uint64_t) * m_needleWidth * m_haystackCount );
    Util::OpenCL::write<uint64_t>(commandQueue, m_needleWidth * m_haystackCount, m_haystack, _clHaystack);
}
void LevensteinDamerauReconstructData::zeroOperations(cl_command_queue commandQueue) {
    memset(m_operations, 0, sizeof(uint64_t) * m_needleWidth * m_haystackCount );
    Util::OpenCL::write<uint64_t>(commandQueue, m_needleWidth * m_haystackCount, m_operations, _clOperations);
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

cl_mem LevensteinDamerauReconstructData::clResult() {
    return _clResult;
}
cl_mem LevensteinDamerauReconstructData::clHaystack() {
    return _clHaystack;
}
cl_mem LevensteinDamerauReconstructData::clOperations() {
    return _clOperations;
}

}}
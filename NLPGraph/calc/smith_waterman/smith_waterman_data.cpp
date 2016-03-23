#include "smith_waterman_data.h"

namespace NLPGraph {
namespace Calc {

SmithWatermanData::SmithWatermanData(cl_context context) {
    m_context = context;
}

SmithWatermanData::~SmithWatermanData() {
}

void SmithWatermanData::alloc() {

    uint64_t uniqueCount = 100;
    
    Util::OpenCL::alloc<uint64_t>(m_context, m_candidatesCount * m_referenceWidth,
        &_candidates, &_clCandidates, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR);
    
    Util::OpenCL::alloc<uint64_t>(m_context, m_referenceWidth,
        &_reference, &_clReference, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR);
    
    Util::OpenCL::alloc<uint64_t>(m_context, (m_operationsWidth+1) * m_candidatesCount,
        &_distsAndOps, &_clDistsAndOps, (int)CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR);
    
    Util::OpenCL::alloc <int64_t>(m_context, uniqueCount,
        &_costMatrix, &_clCostMatrix, (int)CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR);
    
    Util::OpenCL::alloc <int64_t>(m_context, m_candidatesCount * m_referenceWidth * m_referenceWidth,
        &_matrices, &_clMatrices, (int)CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR);
}
void SmithWatermanData::free() {}

void SmithWatermanData::read(cl_command_queue queue) {}

void SmithWatermanData::zeroReference(cl_command_queue commandQueue) {
}
void SmithWatermanData::zeroCandidates(cl_command_queue commandQueue) {
}
void SmithWatermanData::zeroCostMatrix(cl_command_queue commandQueue) {
}
void SmithWatermanData::zeroMatrices(cl_command_queue commandQueue) {
}
void SmithWatermanData::zeroDistsAndOps(cl_command_queue commandQueue) {
}

uint SmithWatermanData::referenceWidth() {
    return m_referenceWidth;
}
uint SmithWatermanData::operationsWidth() {
    return m_operationsWidth;
}
uint SmithWatermanData::candidatesCount() {
    return m_candidatesCount;
}

uint64_t* SmithWatermanData::costMatrix(uint64_t* costMatrix = nullptr) {
    return nullptr;
}
uint64_t* SmithWatermanData::matrices(uint64_t* matrices = nullptr) {
    return nullptr;
}
uint64_t* SmithWatermanData::distsAndOps(uint64_t* distsAndOps = nullptr) {
    return nullptr;
}

cl_mem SmithWatermanData::clReference() {
    return 0;
}
cl_mem SmithWatermanData::clCandidates() {
    return 0;
}
cl_mem SmithWatermanData::clCostMatrix() {
    return 0;
}
cl_mem SmithWatermanData::clMatrices() {
    return 0;
}
cl_mem SmithWatermanData::clDistsAndOps() {
    return 0;
}
    
}}
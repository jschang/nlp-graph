#include "smith_waterman_data.h"

namespace NLPGraph {
namespace Calc {

SmithWatermanData::SmithWatermanData(cl_context context) {
    m_context = context;
}

SmithWatermanData::~SmithWatermanData() {
}

void SmithWatermanData::alloc(uint referenceWidth, uint operationsWidth, uint candidatesCount) {
    
    m_candidatesCount = candidatesCount;
    m_operationsWidth = operationsWidth;
    m_referenceWidth = referenceWidth;
    
    Util::OpenCL::alloc<uint64_t>(m_context, m_candidatesCount * m_referenceWidth,
        &_candidates, &_clCandidates, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR);
    
    Util::OpenCL::alloc<uint64_t>(m_context, m_referenceWidth,
        &_reference, &_clReference, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR);
    
    Util::OpenCL::alloc<uint64_t>(m_context, (m_operationsWidth+1) * m_candidatesCount,
        &_distsAndOps, &_clDistsAndOps, (int)CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR);
    
    Util::OpenCL::alloc <uint64_t>(m_context, m_uniqueCount * m_uniqueCount,
        &_costMatrix, &_clCostMatrix, (int)CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR);
    
    Util::OpenCL::alloc <uint64_t>(m_context, m_candidatesCount * m_referenceWidth * m_referenceWidth,
        &_matrices, &_clMatrices, (int)CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR);
}
void SmithWatermanData::free() {
    if(_candidates!=0) { delete _candidates; _candidates = nullptr; }
    if(_reference!=0) { delete _reference; _reference = nullptr; }
    if(_distsAndOps!=0) { delete _distsAndOps; _distsAndOps = nullptr; }
    if(_costMatrix!=0) { delete _costMatrix; _costMatrix = nullptr; }
    if(_matrices!=0) { delete _matrices; _matrices = nullptr; }
    if(_clCandidates!=0) { clReleaseMemObject(_clCandidates); _clCandidates = 0; }
    if(_clReference!=0) { clReleaseMemObject(_clReference); _clReference = 0; }
    if(_clDistsAndOps!=0) { clReleaseMemObject(_clDistsAndOps); _clDistsAndOps = 0; }
    if(_clCostMatrix!=0) { clReleaseMemObject(_clCostMatrix); _clCostMatrix = 0; }
    if(_clMatrices!=0) { clReleaseMemObject(_clMatrices); _clMatrices = 0; }
}

void SmithWatermanData::read(cl_command_queue queue) {
}
void SmithWatermanData::write(cl_command_queue queue) {
}

void SmithWatermanData::zeroReference(cl_command_queue commandQueue) {
    memset(_reference,0,sizeof(uint64_t) * m_referenceWidth);
    Util::OpenCL::write<uint64_t>(commandQueue, m_referenceWidth, 
        _reference, _clReference);
}
void SmithWatermanData::zeroCandidates(cl_command_queue commandQueue) {
    memset(_candidates,0,sizeof(uint64_t) * m_candidatesCount * m_referenceWidth);
    Util::OpenCL::write<uint64_t>(commandQueue, m_candidatesCount * m_referenceWidth, 
        _candidates, _clCandidates);
}
void SmithWatermanData::zeroCostMatrix(cl_command_queue commandQueue) {
    memset(_costMatrix,0,sizeof(uint64_t) * m_uniqueCount * m_uniqueCount);
    Util::OpenCL::write<uint64_t>(commandQueue, m_uniqueCount * m_uniqueCount, 
        _costMatrix, _clCostMatrix);
}
void SmithWatermanData::zeroMatrices(cl_command_queue commandQueue) {
    memset(_matrices,0,sizeof(uint64_t) * m_candidatesCount * m_referenceWidth * m_referenceWidth);
    Util::OpenCL::write<uint64_t>(commandQueue, m_candidatesCount * m_referenceWidth * m_referenceWidth, 
        _matrices, _clMatrices);
}
void SmithWatermanData::zeroDistsAndOps(cl_command_queue commandQueue) {
    memset(_distsAndOps,0,sizeof(uint64_t) * (m_operationsWidth+1) * m_candidatesCount);
    Util::OpenCL::write<uint64_t>(commandQueue, (m_operationsWidth+1) * m_candidatesCount, 
        _distsAndOps, _clDistsAndOps);
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
uint SmithWatermanData::uniqueCount() {
    return m_uniqueCount;
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
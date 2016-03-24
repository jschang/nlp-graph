#include "smith_waterman_data.h"

namespace NLPGraph {
namespace Calc {

const cl_command_queue null_command_queue();

SmithWatermanData::SmithWatermanData(cl_context context) {
    m_context = context;
}

SmithWatermanData::~SmithWatermanData() {
}

void SmithWatermanData::free() {
    if(_candidates!=0) { delete _candidates; _candidates = nullptr; }
    if(_reference!=0) { delete _reference; _reference = nullptr; }
    if(_clCandidates!=0) { clReleaseMemObject(_clCandidates); _clCandidates = 0; }
    if(_clReference!=0) { clReleaseMemObject(_clReference); _clReference = 0; }
    if(_clDistsAndOps!=0) { clReleaseMemObject(_clDistsAndOps); _clDistsAndOps = 0; }
    if(_clMatrices!=0) { clReleaseMemObject(_clMatrices); _clMatrices = 0; }
}

void SmithWatermanData::write(const cl_command_queue &queue) {
}

void SmithWatermanData::zeroReference(const cl_command_queue &commandQueue) {
    memset(_reference,0,sizeof(uint64_t) * m_referenceWidth);
    Util::OpenCL::write<uint64_t>(commandQueue, m_referenceWidth, 
        _reference, _clReference);
}
void SmithWatermanData::zeroCandidates(const cl_command_queue &commandQueue) {
    memset(_candidates,0,sizeof(uint64_t) * m_candidatesCount * m_referenceWidth);
    Util::OpenCL::write<uint64_t>(commandQueue, m_candidatesCount * m_referenceWidth, 
        _candidates, _clCandidates);
}
void SmithWatermanData::zeroMatrices(const cl_command_queue &commandQueue) {
    uint64_t zero = 0;
    Util::OpenCL::fill(commandQueue, 1, 0, m_candidatesCount * m_referenceWidth * m_referenceWidth, &zero, _clMatrices);
}
void SmithWatermanData::zeroDistsAndOps(const cl_command_queue &commandQueue) {
    uint64_t zero = 0;
    Util::OpenCL::fill(commandQueue, 1, 0, (m_operationsWidth+1) * m_candidatesCount, &zero, _clDistsAndOps);
}

uint SmithWatermanData::referenceWidth() {
    return m_referenceWidth;
}
uint SmithWatermanData::candidatesCount() {
    return m_candidatesCount;
}
uint SmithWatermanData::operationsWidth() {
    return m_operationsWidth;
}
uint SmithWatermanData::uniqueCount() {
    return m_uniqueCount;
}

void SmithWatermanData::reference(const cl_command_queue &commandQueue, const uint64_t *in, const size_t width) {
    m_referenceWidth = width;
    m_operationsWidth = m_referenceWidth * 2;
    Util::OpenCL::alloc<uint64_t>(m_context, m_referenceWidth,
        &_reference, &_clReference, (int)CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR);
}
void SmithWatermanData::candidates(const cl_command_queue &commandQueue, const uint64_t *in, const size_t count) {
    m_candidatesCount = count;
    Util::OpenCL::alloc<uint64_t>(m_context, m_candidatesCount * m_referenceWidth,
        &_candidates, &_clCandidates, (int)CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR);
    Util::OpenCL::alloc <uint64_t>(m_context, m_candidatesCount * m_referenceWidth * m_referenceWidth,
        0, &_clMatrices, (int)CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR);
    Util::OpenCL::alloc<uint64_t>(m_context, (m_operationsWidth+1) * m_candidatesCount,
        0, &_clDistsAndOps, (int)CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR);
}

void SmithWatermanData::matrices(const cl_command_queue &commandQueue, uint64_t **out) {
    *out = (uint64_t*) new uint64_t[m_candidatesCount * m_referenceWidth * m_referenceWidth];
    Util::OpenCL::read<uint64_t>(commandQueue, m_candidatesCount * m_referenceWidth * m_referenceWidth, *out, _clMatrices);
}
void SmithWatermanData::distsAndOps(const cl_command_queue &commandQueue, uint64_t **out) {
    *out = (uint64_t*) new uint64_t[(m_operationsWidth+1) * m_candidatesCount];
    Util::OpenCL::read<uint64_t>(commandQueue, (m_operationsWidth+1) * m_candidatesCount, *out, _clDistsAndOps);
}

cl_mem SmithWatermanData::clReference() {
    return 0;
}
cl_mem SmithWatermanData::clCandidates() {
    return 0;
}
cl_mem SmithWatermanData::clMatrices() {
    return 0;
}
cl_mem SmithWatermanData::clDistsAndOps() {
    return 0;
}
cl_mem SmithWatermanData::clCostMatrix() {
    return 0;
}
    
}}
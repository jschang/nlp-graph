#ifndef smith_waterman_data_hpp
#define smith_waterman_data_hpp

#include "../../nlpgraph.h"
#include "../../util/logger.h"
#include "../../util/opencl.h"

namespace NLPGraph {
namespace Calc {

class SmithWatermanData {
    
private:

    cl_context m_context;
    
    uint   m_referenceWidth  = 0;
    uint   m_operationsWidth = 0;
    uint   m_candidatesCount = 0;
    uint   m_uniqueCount     = 0;
    
    uint64_t* _reference   = 0;
    uint64_t* _candidates  = 0;
    uint64_t* _costMatrix  = 0;
    uint64_t* _matrices    = 0;
    uint64_t* _distsAndOps = 0;
    
    cl_mem _clReference   = 0;
    cl_mem _clCandidates  = 0;
    cl_mem _clCostMatrix  = 0;
    cl_mem _clMatrices    = 0;
    cl_mem _clDistsAndOps = 0;
    
public:
    
    SmithWatermanData(cl_context context);
    ~SmithWatermanData();
    
    void alloc(uint referenceWidth, uint operationsWidth, uint candidatesCount);
    void free();
    
    void read(cl_command_queue queue);
    void write(cl_command_queue queue);
    
    void zeroReference(cl_command_queue commandQueue);
    void zeroCandidates(cl_command_queue commandQueue);
    void zeroCostMatrix(cl_command_queue commandQueue);
    void zeroMatrices(cl_command_queue commandQueue);
    void zeroDistsAndOps(cl_command_queue commandQueue);
    
    uint64_t* costMatrix(uint64_t* costMatrix);
    uint64_t* matrices(uint64_t* matrices);
    uint64_t* distsAndOps(uint64_t* distsAndOps);
    
    uint referenceWidth();
    uint operationsWidth();
    uint candidatesCount();
    uint uniqueCount();
    
    cl_mem clReference();
    cl_mem clCandidates();
    cl_mem clCostMatrix();
    cl_mem clMatrices();
    cl_mem clDistsAndOps();
};
    
}}

#endif
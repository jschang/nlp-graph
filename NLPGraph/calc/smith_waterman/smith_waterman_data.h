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
    
    cl_mem _clReference   = 0;
    cl_mem _clCandidates  = 0;
    
    cl_mem _clCostMatrix  = 0;
    cl_mem _clMatrices    = 0;
    cl_mem _clDistsAndOps = 0;
    
public:
    
    SmithWatermanData(cl_context context);
    ~SmithWatermanData();
    
    void free();
    
    void read(const cl_command_queue &queue);
    void write(const cl_command_queue &queue);
    
    void zeroReference(const cl_command_queue &commandQueue);
    void zeroCandidates(const cl_command_queue &commandQueue);
    void zeroMatrices(const cl_command_queue &commandQueue);
    void zeroDistsAndOps(const cl_command_queue &commandQueue);
    
    void reference(const cl_command_queue &commandQueue, const uint64_t *in, const size_t width);
    void candidates(const cl_command_queue &commandQueue, const uint64_t *in, const size_t count);
    void matrices(const cl_command_queue &commandQueue, uint64_t **out);
    void distsAndOps(const cl_command_queue &commandQueue, uint64_t **out);
    
    uint referenceWidth();
    uint operationsWidth();
    uint candidatesCount();
    uint uniqueCount();
    
    cl_mem clReference();
    cl_mem clCandidates();
    cl_mem clMatrices();
    cl_mem clDistsAndOps();
    cl_mem clCostMatrix();
};
    
}}

#endif
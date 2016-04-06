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
    
    cl_mem _clUniques     = 0;
    cl_mem _clCostMatrix  = 0;
    cl_mem _clMatrices    = 0;
    cl_mem _clDistsAndOps = 0;
    
    std::map<uint64_t,uint64_t> m_uniques;
    
public:
    
    SmithWatermanData(cl_context context);
    ~SmithWatermanData();
    
    void reference(const cl_command_queue &commandQueue, const uint64_t *in, const size_t width);
    void candidates(const cl_command_queue &commandQueue, const uint64_t *in, const size_t count);
    
    /**
     * call after setting reference and candidates
     */
    void prepare(const cl_command_queue &commandQueue);
    
    void free();
    void zeroReference(const cl_command_queue &commandQueue);
    void zeroCandidates(const cl_command_queue &commandQueue);
    void zeroMatrices(const cl_command_queue &commandQueue);
    void zeroDistsAndOps(const cl_command_queue &commandQueue);
    
    void matrices(const cl_command_queue &commandQueue, int64_t **out);
    void distsAndOps(const cl_command_queue &commandQueue, uint64_t **out);
    
    uint referenceWidth();
    uint operationsWidth();
    uint candidatesCount();
    uint uniqueCount();
    
    cl_mem clReference();
    cl_mem clCandidates();
    cl_mem clMatrices();
    cl_mem clDistsAndOps();
    /**
     * This is the cost matrix.  Each edge is one of the unique set of members
     * among all candidate and reference uint64_ts.  The edge members are identified
     * by the contents of _clUnqiues.  Each value is a cost determined by how
     * frequently the intersecting pair appears close to one another.
     */
    cl_mem clCostMatrix();
    /**
     * This is a reference for the index along either edge of the cost matrix.
     * Uniques are ordered ascending for lookup within the OpenCL functions.
     */
    cl_mem clUniques();
};
    
}}

#endif
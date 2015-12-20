//
//  levenstein_damerau_reconstruct_data.hpp
//  NLPGraph
//
//  Created by Jonathan Schang on 12/6/15.
//
//

#ifndef levenstein_damerau_data_reconstruct_hpp
#define levenstein_damerau_data_reconstruct_hpp

#include "../../nlpgraph.h"
#include "../../util/logger.h"
#include "../../util/opencl.h"

namespace NLPGraph {
namespace Calc {

/**
 * Each haystack element should have been a needle.
 * Each operation should convert that needle into
 * the originally experienced data.
 */
class LevensteinDamerauReconstructData {

private:

    uint      m_needleWidth    = 0;
    uint      m_haystackCount  = 0;
    uint      m_operationsSize = 0;
    
    uint64_t* m_result         = 0;
    uint64_t* m_haystack       = 0;
    uint64_t* m_operations     = 0;
    
    cl_mem _clResult     = 0;
    cl_mem _clHaystack   = 0;
    cl_mem _clOperations = 0;
     
private:

    void alloc(uint needleWidth, uint haystackSize);
    
public:

    LevensteinDamerauReconstructData(cl_context context, uint needleWidth, uint haystackCount);
    LevensteinDamerauReconstructData(cl_context context, uint needleWidth, uint haystackCount, uint64_t* operations, uint64_t* haystack);
    
    ~LevensteinDamerauReconstructData();
    
    void alloc(cl_context context, uint needleWidth, uint haystackCount, uint64_t* haystack = 0, uint64_t* operations = 0);
    void free();
    void read(cl_command_queue queue);
    
    void zeroResult(cl_command_queue commandQueue);
    void zeroHaystack(cl_command_queue commandQueue);
    void zeroOperations(cl_command_queue commandQueue);
    
    uint getNeedleWidth();
    uint getHaystackCount();
    uint getHaystackSize();
    uint getOperationsSize();
    uint getOperationWidth();

    uint64_t* getResult();
    uint64_t* getHaystack();
    uint64_t* getOperations();
    
    cl_mem clResult();
    cl_mem clHaystack();
    cl_mem clOperations();
};

}}

#endif /* levenstein_damerau_data_hpp */

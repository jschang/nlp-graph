//
//  levenstein_damerau_data.hpp
//  NLPGraph
//
//  Created by Jonathan Schang on 11/24/15.
//
//

#ifndef levenstein_damerau_data_hpp
#define levenstein_damerau_data_hpp

#include "../../nlpgraph.h"
#include "../../util/logger.h"
#include "../../util/opencl.h"

namespace NLPGraph {
namespace Calc {

class LevensteinDamerauData {

private:

    uint      m_needleWidth     = 0;
    uint      m_haystackCount   = 0;
    uint      m_needleCount     = 0;
    uint      m_operationsCount = 0;
    
    cl_mem _clHaystack   = 0; // uint64_t
    cl_mem _clNeedle     = 0; // uint64_t
    cl_mem _clOperations = 0; // int64_t
    cl_mem _clDistances  = 0; // uint64_t
    
    int64_t  *  _distances = 0;
    uint64_t * _operations = 0;
    
public:

    LevensteinDamerauData(cl_context clContext, uint needleWidth, uint needleCount, uint haystackCount, uint64_t* needle, uint64_t* haystack);
    
    ~LevensteinDamerauData();
    
    void free();
    
    uint needleWidth();
    uint haystackCount();
    uint needleCount();
    uint operationsCount();
    uint operationWidth();
    
    void zeroDistances(cl_command_queue commandQueue);
    void zeroOperations(cl_command_queue commandQueue);
    
    void read(cl_command_queue commandQueue);
    
    uint64_t* operations();
    int64_t* distances();
    
    cl_mem clNeedle();
    cl_mem clHaystack();
    cl_mem clOperations();
    cl_mem clDistances();
};

}}

#endif /* levenstein_damerau_data_hpp */

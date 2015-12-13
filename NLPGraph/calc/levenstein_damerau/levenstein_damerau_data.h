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

    uint      m_needleWidth    = 0;
    uint      m_haystackSize   = 0;
    uint      m_operationsSize = 0;
    
    uint64_t* m_needle         = 0;
    uint64_t* m_haystack       = 0;
    uint64_t* m_operations     = 0;
    
     int64_t* m_distances      = 0;
     
private:

    void alloc(uint needleWidth, uint haystackSize);
    
public:

    LevensteinDamerauData(uint needleWidth, uint haystackSize);
    LevensteinDamerauData(uint needleWidth, uint haystackSize, uint64_t* needle, uint64_t* haystack);
    
    ~LevensteinDamerauData();
    
    void      zeroNeedle();
    void      zeroHaystack();
    void      zeroOperations();
    
    void      zeroDistances();
    
    uint      getNeedleWidth();
    uint      getHaystackSize();
    uint      getOperationsSize();
    uint      getOperationWidth();
    
    uint64_t* getNeedle();
    uint64_t* getHaystack();
    uint64_t* getOperations();
    
     int64_t* getDistances();
};

}}

#endif /* levenstein_damerau_data_hpp */

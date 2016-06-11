#define BOOST_LOG_DYN_LINK

#include <util/opencl.h>
#include <util/time_helper.h>
#include <util/math.h>
#include <calc/smith_waterman.h>
#include <boost/test/unit_test.hpp>
#include <boost/compute.hpp>

#include <nlpgraph.h>
#include <util/string.h>
#include "../nlpgraph_tests.h"

#if RUN_TEST_ALL == 1 || RUN_TEST_CALC_SMITH_WATERMAN == 1

#define LOG BOOST_LOG_SEV(logger,severity_level::normal) << __PRETTY_FUNCTION__ << " "

using namespace NLPGraph::Dao;
using namespace NLPGraph::Dto;
using namespace NLPGraph::Util;
using namespace NLPGraph::Calc;
using namespace boost::compute;
using namespace boost;

struct Fixture {
    Fixture() {
        BOOST_TEST_MESSAGE("Fixture setup");
    }
    ~Fixture() {
        BOOST_TEST_MESSAGE("Fixture teardown");
    }
};

void logMatrix(LoggerType logger, uint width, int64_t *mPtr, uint64_t needle[], uint64_t haystack[]) {
    std::vector<uint64_t> columns(width+1);
    for (int i = 0; i < (width+1); i++){
        columns[i] = i;
    }
    LOG << "Column: " << "    " << String::str<uint64_t>(&columns[0],width+1);
    LOG << "Needle: " << "    0," << String::str<uint64_t>(&needle[0],width);
    long itTo = (width+1)*(width+1);
    for(int i=0, j=0; i<itTo; i+=(width+1), j++) {
        LOG << "Matrix: " << j << " " << (j>0?haystack[j-1]:0) << " " << String::str<int64_t>(&mPtr[i],(width+1));
    }
}

BOOST_AUTO_TEST_SUITE( nlpgraph_calc_smith_waterman )

BOOST_AUTO_TEST_CASE( calc_test )
{   
    LoggerType logger = LoggerType(boost::log::keywords::channel="nlpgraph_calc_smith_waterman");
    
    // get the best device
    OpenCLDeviceInfoType deviceInfo = OpenCLDeviceInfoType();
    OpenCL::bestDeviceInfo(deviceInfo);
    OpenCL::log(deviceInfo);
    
    // spin up a context
    device bDevice = device(deviceInfo.id);
    context bContext = context(OpenCL::contextWithDeviceInfo(deviceInfo));    
    command_queue bCommandQueue = command_queue(bContext, bDevice);
    // fire up the calculator
    SmithWaterman alg(bContext);
    
    // flip to on, if you're interested in fixing an issue in the cl code
    alg.clLogOn = true;
    
    // first simple test
    { // perfect match
        LOG << "Testing perfect match";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,2,3,4};
        int64_t matrices[25];
        int64_t *mPtr = &matrices[0];
        uint64_t distsAndOps[5];
        uint64_t *dPtr = &distsAndOps[0];
        memset(matrices,0,sizeof(int64_t)*25);
        // create the data structure
        SmithWatermanDataPtr dataPtr = SmithWatermanDataPtr(new SmithWatermanData(bContext));
        dataPtr->reference(bCommandQueue, needle, 4);
        dataPtr->candidates(bCommandQueue, haystack, 1);
        dataPtr->prepare(bCommandQueue);
        // have the alg calculate the matrices
        alg.createMatrices(dataPtr);
        dataPtr->matrices(bCommandQueue,&mPtr);
        logMatrix(logger,width,mPtr,needle,haystack);
        // have the alg walk the matrices and determine distance and operations
        alg.calculateDistances(dataPtr);
        dataPtr->distsAndOps(bCommandQueue, &dPtr);
        LOG << "DistAndOps: " << String::str<uint64_t>(&dPtr[0],5);
    }
    
    { // perfect match
        LOG << "Testing partial match";
        uint width=8;
        uint haystackSize=1;
        uint64_t needle[]   = {1,2,3,4,5,6,7,8};
        uint64_t haystack[] = {1,2,9,4,5,6,8,8};
        size_t matrixSize = (width+1)*(width+1);
        int64_t matrices[matrixSize];
        int64_t *mPtr = &matrices[0];
        uint64_t distsAndOps[width+1];
        uint64_t *dPtr = &distsAndOps[0];
        memset(matrices,0,sizeof(int64_t)*matrixSize);
        SmithWatermanDataPtr dataPtr = SmithWatermanDataPtr(new SmithWatermanData(bContext));
        dataPtr->reference(bCommandQueue, needle, width);
        dataPtr->candidates(bCommandQueue, haystack, haystackSize);
        dataPtr->prepare(bCommandQueue);
        alg.createMatrices(dataPtr);
        dataPtr->matrices(bCommandQueue,&mPtr);
        logMatrix(logger,width,mPtr,needle,haystack);
        alg.calculateDistances(dataPtr);
        dataPtr->distsAndOps(bCommandQueue, &dPtr);
        LOG << "DistAndOps: " << String::str<uint64_t>(&dPtr[0],width+1);
    }
    
    { // mangled shit
        LOG << "Testing mangled shit";
        
        uint width=8;
        uint haystackSize=1;
        uint64_t needle[]   = {1,2,3,4,5,6,7,8};
        uint64_t haystack[] = {4,5,9,6,8,8,1,2};
        size_t matrixSize = (width+1)*(width+1);
        int64_t matrices[matrixSize];
        int64_t *mPtr = &matrices[0];
        uint64_t distsAndOps[width*2+3];
        uint64_t *dPtr = &distsAndOps[0];
        
        memset(matrices,0,sizeof(int64_t)*matrixSize);
        
        SmithWatermanDataPtr dataPtr = SmithWatermanDataPtr(new SmithWatermanData(bContext));
        dataPtr->reference(bCommandQueue, needle, width);
        dataPtr->candidates(bCommandQueue, haystack, haystackSize);
        dataPtr->prepare(bCommandQueue);
        
        alg.createMatrices(dataPtr);
        dataPtr->matrices(bCommandQueue,&mPtr);
        
        logMatrix(logger,width,mPtr,needle,haystack);
        
        alg.calculateDistances(dataPtr);
        dataPtr->distsAndOps(bCommandQueue, &dPtr);
        LOG << "DistAndOps: " << String::str<uint64_t>(&dPtr[0],dataPtr->operationsWidth());
    }
}

BOOST_AUTO_TEST_SUITE_END()

#endif
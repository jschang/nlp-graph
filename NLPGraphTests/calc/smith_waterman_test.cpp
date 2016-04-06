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
        int64_t matrices[16];
        int64_t *mPtr = &matrices[0];
        memset(matrices,0,sizeof(int64_t)*16);
        SmithWatermanDataPtr dataPtr = SmithWatermanDataPtr(new SmithWatermanData(bContext));
        dataPtr->reference(bCommandQueue, needle, 4);
        dataPtr->candidates(bCommandQueue, haystack, 1);
        dataPtr->prepare(bCommandQueue);
        alg.createMatrices(dataPtr);
        dataPtr->matrices(bCommandQueue,&mPtr);
    }
}

BOOST_AUTO_TEST_SUITE_END()

#endif
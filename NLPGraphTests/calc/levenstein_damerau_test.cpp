#define BOOST_LOG_DYN_LINK
#include <time.h>
#include <util/opencl.h>
#include <util/time_helper.h>
#include <calc/levenstein_damerau.h>
#include <boost/test/unit_test.hpp>
#include <boost/compute.hpp>
#include <boost/random.hpp>

#include <nlpgraph.h>
#include <util/string.h>
#include "../nlpgraph_tests.h"

#define LOG BOOST_LOG_SEV(logger,severity_level::normal) << __PRETTY_FUNCTION__ << " "

using namespace NLPGraph::Dao;
using namespace NLPGraph::Dto;
using namespace NLPGraph::Util;
using namespace NLPGraph::Calc;
using namespace boost::compute;
using namespace boost::random;
using namespace boost;

LoggerType logger(boost::log::keywords::channel="nlpgraph_calc_levenstein_damerau");

struct Fixture {
    Fixture() {
        BOOST_TEST_MESSAGE("Fixture setup");
    }
    ~Fixture() {
        BOOST_TEST_MESSAGE("Fixture teardown");
    }
};

void pfn_notify (
    const char *errinfo, 
    const void *private_info, 
    size_t cb, 
    void *user_data
) {
    LOG << errinfo;
}

BOOST_AUTO_TEST_SUITE( nlpgraph_calc_levenstein_damerau ) 

BOOST_AUTO_TEST_CASE( calc_test )
{       
    LoggerType logger(boost::log::keywords::channel="nlpgraph_calc_levenstein_damerau");

    // get the best device
    OpenCLDeviceInfoType deviceInfo;
    OpenCL::bestDeviceInfo(deviceInfo);
    
    // spin up a context
    context bContext;
    bContext = context(OpenCL::contextWithDeviceInfo(deviceInfo));    
    
    // fire up the calculator
    LevensteinDamerau alg(bContext);
    alg.clLogOn = true;
    
    // first simple test
    { // perfect match
        LOG << "Testing perfect match";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,2,3,4};
        uint64_t distancesOut[1];
        uint64_t operationsOut[haystackSize*(2*width)];
        alg.calculate(width, haystackSize, (uint64_t*)&needle, (uint64_t*)&haystack, (uint64_t*)&distancesOut, (uint64_t*)&operationsOut);
        LOG << distancesOut[0];
        BOOST_CHECK(distancesOut[0] == 0);
    }
    { // transposition requires 2 edits
        LOG << "Testing single transposition - middle";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,3,2,4};
        uint64_t distancesOut[1];
        uint64_t operationsOut[haystackSize*(2*width)];
        alg.calculate(width, haystackSize, (uint64_t*)&needle, (uint64_t*)&haystack, (uint64_t*)&distancesOut, (uint64_t*)&operationsOut);
        LOG << distancesOut[0];
        BOOST_CHECK(distancesOut[0] == 2);
    }
    { // transposition requires 2 edits
        LOG << "Testing single transposition - left edge";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {2,1,3,4};
        uint64_t distancesOut[1];
        uint64_t operationsOut[haystackSize*(2*width)];
        alg.calculate(width, haystackSize, (uint64_t*)&needle, (uint64_t*)&haystack, (uint64_t*)&distancesOut, (uint64_t*)&operationsOut);
        LOG << distancesOut[0];
        BOOST_CHECK(distancesOut[0] == 2);
    }
    { // transposition requires 2 edits
        LOG << "Testing single transposition - right edge";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,2,4,3};
        uint64_t distancesOut[1];
        uint64_t operationsOut[haystackSize*(2*width)];
        alg.calculate(width, haystackSize, (uint64_t*)&needle, (uint64_t*)&haystack, (uint64_t*)&distancesOut, (uint64_t*)&operationsOut);
        LOG << distancesOut[0];
        BOOST_CHECK(distancesOut[0] == 2);
    }
    { // single deletion - middle
        LOG << "Testing single deletion - middle";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,2,4};
        uint64_t distancesOut[1];
        uint64_t operationsOut[haystackSize*(2*width)];
        alg.calculate(width, haystackSize, (uint64_t*)&needle, (uint64_t*)&haystack, (uint64_t*)&distancesOut, (uint64_t*)&operationsOut);
        LOG << distancesOut[0];
        BOOST_CHECK(distancesOut[0] == 1);
    }
    { // single deletion - left edge
        LOG << "Testing single deletion - left edge";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {2,3,4};
        uint64_t distancesOut[1];
        uint64_t operationsOut[haystackSize*(2*width)];
        alg.calculate(width, haystackSize, (uint64_t*)&needle, (uint64_t*)&haystack, (uint64_t*)&distancesOut, (uint64_t*)&operationsOut);
        LOG << distancesOut[0];
        BOOST_CHECK(distancesOut[0] == 1);
    }
    { // single deletion - right edge
        LOG << "Testing single deletion - right edge";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,2,3};
        uint64_t distancesOut[1];
        uint64_t operationsOut[haystackSize*(2*width)];
        alg.calculate(width, haystackSize, (uint64_t*)&needle, (uint64_t*)&haystack, (uint64_t*)&distancesOut, (uint64_t*)&operationsOut);
        LOG << distancesOut[0];
        BOOST_CHECK(distancesOut[0] == 1);
    }
    { // single deletion - middle
        LOG << "Testing single deletion, insertion - middle";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,2,2,4};
        uint64_t distancesOut[1];
        uint64_t operationsOut[haystackSize*(2*width)];
        alg.calculate(width, haystackSize, (uint64_t*)&needle, (uint64_t*)&haystack, (uint64_t*)&distancesOut, (uint64_t*)&operationsOut);
        LOG << distancesOut[0];
        // this is actually correct, there is only one operation: "substitution"
        BOOST_CHECK(distancesOut[0] == 1);
    }
    { // single deletion - left edge
        LOG << "Testing single deletion, insertion - left edge";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {5,2,3,4};
        uint64_t distancesOut[1];
        uint64_t operationsOut[haystackSize*(2*width)];
        alg.calculate(width, haystackSize, (uint64_t*)&needle, (uint64_t*)&haystack, (uint64_t*)&distancesOut, (uint64_t*)&operationsOut);
        LOG << distancesOut[0];
        BOOST_CHECK(distancesOut[0] == 1);
    }
    { // single deletion - right edge
        LOG << "Testing single deletion, insertion - right edge";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,2,3,5};
        uint64_t distancesOut[1];
        uint64_t operationsOut[haystackSize*(2*width)];
        alg.calculate(width, haystackSize, (uint64_t*)&needle, (uint64_t*)&haystack, (uint64_t*)&distancesOut, (uint64_t*)&operationsOut);
        LOG << distancesOut[0];
        BOOST_CHECK(distancesOut[0] == 1);
    }
    { // single deletion - left edge
        LOG << "Testing single deletion, repetition - left edge";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {2,2,3,4};
        uint64_t distancesOut[1];
        uint64_t operationsOut[haystackSize*(2*width)];
        alg.calculate(width, haystackSize, (uint64_t*)&needle, (uint64_t*)&haystack, (uint64_t*)&distancesOut, (uint64_t*)&operationsOut);
        LOG << distancesOut[0];
        BOOST_CHECK(distancesOut[0] == 2);
    }
    { // single deletion - right edge
        LOG << "Testing single deletion, repetition - right edge";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,2,4,4};
        uint64_t distancesOut[1];
        uint64_t operationsOut[haystackSize*(2*width)];
        alg.calculate(width, haystackSize, (uint64_t*)&needle, (uint64_t*)&haystack, (uint64_t*)&distancesOut, (uint64_t*)&operationsOut);
        LOG << distancesOut[0];
        BOOST_CHECK(distancesOut[0] == 2);
    }
}

BOOST_AUTO_TEST_CASE( stress_test ) {

    LoggerType logger(boost::log::keywords::channel="nlpgraph_calc_levenstein_damerau");

    // get the best device
    OpenCLDeviceInfoType deviceInfo;
    OpenCL::bestDeviceInfo(deviceInfo);
    
    // spin up a context
    context bContext;
    bContext = context(OpenCL::contextWithDeviceInfo(deviceInfo));    
    
    // fire up the calculator
    LevensteinDamerau alg(bContext);
    alg.clLogOn = true;

    // initialize the overly complicated initialization for random numbers
    struct timeval tv;
    boost::random::mt19937 randGen(tv.tv_usec);
    uniform_int<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());
    boost::variate_generator<boost::random::mt19937&, uniform_int<uint64_t>> getRand(randGen, dist);
    
    uint testSize = 1;
    uint testWidth = 35;
    
    uint64_t *needle = (uint64_t *)malloc(sizeof(uint64_t)*testSize);
    uint64_t *haystack = new uint64_t[testWidth*testSize];
    uint64_t *distancesOut = new uint64_t[testSize];
    uint64_t *operationsOut = new uint64_t[testSize*(2*testWidth)]; 

    for(int i=0; i<100; i++) {
        memset(haystack,testWidth*testSize,sizeof(uint64_t));
        for(int i=0; i<testSize; i++) {
            for(int j=0; j<testWidth; j++) {
                double a = (double)getRand();
                double b = (double)getRand();
                double c = ::fmin(a,b)/::fmax(a,b);
                haystack[(i*testWidth)+j] = c > .10 ? needle[j] : getRand();
            }
        }
        memset(needle,testSize,sizeof(uint64_t));
        for(int i=0; i<(testWidth); i++) {
            needle[i] = getRand();
        }
        LOG << "needle:" << NLPGraph::Util::String::str(needle,testWidth);
        LOG << "haystacks:" << NLPGraph::Util::String::str(haystack,testWidth*testSize);
        timespec start = TimeHelper::getTimeStruct();
        alg.calculate(testWidth, testSize, (uint64_t*)needle, (uint64_t*)haystack, (uint64_t*)distancesOut, (uint64_t*)operationsOut);
        timespec end = TimeHelper::getTimeStruct();
        LOG << "distances:" << NLPGraph::Util::String::str(distancesOut,testWidth);
        LOG << "time: " << (end.tv_nsec - start.tv_nsec);
    }
    
    delete needle;
    delete haystack;
    delete distancesOut;
    delete operationsOut;
}

BOOST_AUTO_TEST_CASE( perf_test ) {

    LoggerType logger(boost::log::keywords::channel="nlpgraph_calc_levenstein_damerau");

    // get the best device
    OpenCLDeviceInfoType deviceInfo;
    OpenCL::bestDeviceInfo(deviceInfo);
    
    // spin up a context
    context bContext;
    bContext = context(OpenCL::contextWithDeviceInfo(deviceInfo));    
    
    // fire up the calculator
    LevensteinDamerau alg(bContext);

    // initialize the overly complicated initialization for random numbers
    struct timeval tv;
    boost::random::mt19937 randGen(tv.tv_usec);
    uniform_int<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());
    boost::variate_generator<boost::random::mt19937&, uniform_int<uint64_t>> getRand(randGen, dist);
    
    uint testSize = 50000;
    uint testWidth = 35;
    
    uint64_t *needle = new uint64_t[testSize];
    for(int i=0; i<(testWidth); i++) {
        needle[i] = getRand();
    }
    uint64_t *haystack = new uint64_t[testWidth*testSize];
    for(int i=0; i<(testSize*testWidth); i++) {
        haystack[i] = getRand();
    }
    uint64_t *distancesOut = new uint64_t[testSize];
    uint64_t *operationsOut = new uint64_t[testSize*(2*testWidth)]; 
    
    timespec start = TimeHelper::getTimeStruct();
    alg.calculate(testWidth, testSize, (uint64_t*)needle, (uint64_t*)haystack, (uint64_t*)distancesOut, (uint64_t*)operationsOut);
    timespec end = TimeHelper::getTimeStruct();
    LOG << "time: " << (end.tv_nsec - start.tv_nsec);
    
    LOG << distancesOut[0];
    BOOST_CHECK(distancesOut[0] == 1);
    
    // clean up after self
    delete needle;
    delete haystack;
    delete distancesOut;
    delete operationsOut;
}

BOOST_AUTO_TEST_SUITE_END()
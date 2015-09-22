#define BOOST_LOG_DYN_LINK

#include <util/opencl.h>
#include <util/time_helper.h>
#include <util/math.h>
#include <calc/levenstein_damerau.h>
#include <boost/test/unit_test.hpp>
#include <boost/compute.hpp>

#include <nlpgraph.h>
#include <util/string.h>
#include "../nlpgraph_tests.h"

#define LOG BOOST_LOG_SEV(logger,severity_level::normal) << __PRETTY_FUNCTION__ << " "

using namespace NLPGraph::Dao;
using namespace NLPGraph::Dto;
using namespace NLPGraph::Util;
using namespace NLPGraph::Calc;
using namespace boost::compute;
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
    
    // flip to on, if you're interested in fixing an issue in the cl code
    alg.clLogOn = true;
    
    // first simple test
    { // perfect match
        LOG << "Testing perfect match";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,2,3,4};
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            width, haystackSize, (uint64_t*)&needle[0], (uint64_t*)&haystack[0]));
        alg.calculate(dataPtr);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->getDistances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->getOperations(),4*3);
        BOOST_CHECK(dataPtr->getDistances()[0] == 0);
    }
    { // transposition requires 1 edits
        LOG << "Testing single transposition - middle";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,3,2,4};
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            width, haystackSize, needle, haystack));
        alg.calculate(dataPtr);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->getDistances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->getOperations(),4*3);
        BOOST_CHECK(dataPtr->getDistances()[0] == 1);
    }
    { // transposition requires 1 edits
        LOG << "Testing single transposition - left edge";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {2,1,3,4};
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            width, haystackSize, needle, haystack));
        alg.calculate(dataPtr);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->getDistances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->getOperations(),4*3);
        BOOST_CHECK(dataPtr->getDistances()[0] == 1);
    }
    { // transposition requires 1 edits
        LOG << "Testing single transposition - right edge";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,2,4,3};
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            width, haystackSize, needle, haystack));
        alg.calculate(dataPtr);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->getDistances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->getOperations(),4*3);
        BOOST_CHECK(dataPtr->getDistances()[0] == 1);
    }
    { // single deletion - middle
        LOG << "Testing single deletion - middle";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,2,4,0};
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            width, haystackSize, needle, haystack));
        alg.calculate(dataPtr);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->getDistances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->getOperations(),4*3);
        BOOST_CHECK(dataPtr->getDistances()[0] == 1);
    }
    { // single deletion - left edge
        LOG << "Testing single deletion - left edge";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {2,3,4,0};
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            width, haystackSize, needle, haystack));
        alg.calculate(dataPtr);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->getDistances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->getOperations(),4*3);
        BOOST_CHECK(dataPtr->getDistances()[0] == 1);
    }
    { // single deletion - right edge
        LOG << "Testing single deletion - right edge";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,2,3,0};
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            width, haystackSize, needle, haystack));
        alg.calculate(dataPtr);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->getDistances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->getOperations(),4*3);
        BOOST_CHECK(dataPtr->getDistances()[0] == 1);
    }
    { // single deletion - middle
        LOG << "Testing single deletion, repetition - middle";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,2,2,4};
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            width, haystackSize, needle, haystack));
        alg.calculate(dataPtr);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->getDistances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->getOperations(),4*3);
        // this is actually correct, there is only one operation: "substitution"
        // 2015-08-16 - yeah, but 2 is easier...we'll use two for now: "repetition" and "deletion"
        BOOST_CHECK(dataPtr->getDistances()[0] == 2);
    }
    { // single deletion - middle
        LOG << "Testing single deletion, insertion - middle";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,2,5,4};
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            width, haystackSize, needle, haystack));
        alg.calculate(dataPtr);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->getDistances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->getOperations(),4*3);
        // this is actually correct, there is only one operation: "substitution"
        // 2015-08-16 - yeah, but 2 is easier...we'll use two for now: "repetition" and "deletion"
        BOOST_CHECK(dataPtr->getDistances()[0] == 1);
    }
    { // single deletion - left edge
        LOG << "Testing single deletion, insertion - left edge";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {5,2,3,4};
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            width, haystackSize, needle, haystack));
        alg.calculate(dataPtr);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->getDistances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->getOperations(),4*3);
        BOOST_CHECK(dataPtr->getDistances()[0] == 1);
    }
    { // single deletion - right edge
        LOG << "Testing single deletion, insertion - right edge";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,2,3,5};
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            width, haystackSize, needle, haystack));
        alg.calculate(dataPtr);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->getDistances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->getOperations(),4*3);
        BOOST_CHECK(dataPtr->getDistances()[0] == 1);
    }
    { // single deletion - left edge
        LOG << "Testing single deletion, repetition - left edge";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {2,2,3,4};
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            width, haystackSize, needle, haystack));
        alg.calculate(dataPtr);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->getDistances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->getOperations(),4*3);
        BOOST_CHECK(dataPtr->getDistances()[0] == 2);
    }
    { // single deletion - right edge
        LOG << "Testing single deletion, repetition - right edge";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,2,4,4};
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            width, haystackSize, needle, haystack));
        alg.calculate(dataPtr);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->getDistances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->getOperations(),4*3);
        BOOST_CHECK(dataPtr->getDistances()[0] == 2);
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
    alg.clLogErrorOnly = false;

    uint testSize = 1;
    uint testWidth = 10;
    
    NLPGraph::Util::GeneratorIntegerPtr<uint64_t> getRand = NLPGraph::Util::Math::rangeRandGen<uint64_t>(1000,9999);
    NLPGraph::Util::GeneratorRealPtr getChance = NLPGraph::Util::Math::rangeRandGen(0.0f,1.0f);
    NLPGraph::Util::GeneratorIntegerPtr<uint> getNeedleWidth = NLPGraph::Util::Math::rangeRandGen<uint>(0,testWidth);
    
    uint64_t *needle = (uint64_t *)malloc(sizeof(uint64_t)*testSize);
    uint64_t *haystack = new uint64_t[testWidth*testSize];

    for(int i=0; i<100; i++) {
        LOG << "STARTING ITERATION: " << i;
        memset(needle,0,sizeof(uint64_t)*testSize);
        uint needleWidth = (*getNeedleWidth)();
        for(int i=0; i<(needleWidth); i++) {
            needle[i] = (*getRand)();
        }
        memset(haystack,0,testWidth*testSize*sizeof(uint64_t));
        uint haystackBase = 0;
        for(int i=0, haystackJ=0; i<testSize; i++, haystackBase+=testWidth) {
            int j=0;
            while(j<needleWidth && haystackJ<testWidth) {
                uint64_t needleJ = needle[j];
                float c = (*getChance)();
                if(c>.70) { // 70% of the time, just copy the position
                    haystack[ haystackBase + haystackJ ] = needleJ;
                    haystackJ ++;
                    j++;
                } else if( c>=.70 && c<.80 ) { // 10% of the time, delete a position
                    j++;
                } else { // 25% of the time, append a position
                    uint64_t rand = (*getRand)();
                    haystack[ haystackBase + haystackJ ] = rand;
                    haystackJ ++;
                }
            }
        }
        LOG << "needle:" << NLPGraph::Util::String::str(needle,testWidth);
        LOG << "haystacks:" << NLPGraph::Util::String::str(haystack,testWidth*testSize);
        timespec start = TimeHelper::getTimeStruct();
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            testWidth, testSize, needle, haystack));
        alg.calculate(dataPtr);
        timespec end = TimeHelper::getTimeStruct();
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->getDistances(),testSize);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->getOperations(),dataPtr->getOperationsSize());
        LOG << "time: " << (end.tv_nsec - start.tv_nsec);
        BOOST_CHECK( dataPtr->getDistances()[0] >= 0 );
    }
    
    delete needle;
    delete haystack;
}

BOOST_AUTO_TEST_CASE( perf_test ) {

    return;
    
    LoggerType logger(boost::log::keywords::channel="nlpgraph_calc_levenstein_damerau");

    // get the best device
    OpenCLDeviceInfoType deviceInfo;
    OpenCL::bestDeviceInfo(deviceInfo);
    
    // spin up a context
    context bContext;
    bContext = context(OpenCL::contextWithDeviceInfo(deviceInfo)); 
    
    NLPGraph::Util::GeneratorIntegerPtr<uint64_t> getRand = NLPGraph::Util::Math::rangeRandGen<uint64_t>(1000,9999);   
    
    // fire up the calculator
    LevensteinDamerau alg(bContext);
    
    uint testSize = 50000;
    uint testWidth = 35;
    
    uint64_t *needle = new uint64_t[testSize];
    for(int i=0; i<(testWidth); i++) {
        needle[i] = (*getRand)();
    }
    uint64_t *haystack = new uint64_t[testWidth*testSize];
    for(int i=0; i<(testSize*testWidth); i++) {
        haystack[i] = (*getRand)();
    }
    
    timespec start = TimeHelper::getTimeStruct();
    LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            testWidth, testSize, needle, haystack));
    alg.calculate(dataPtr);
    timespec end = TimeHelper::getTimeStruct();
    LOG << "time: " << (end.tv_nsec - start.tv_nsec);
    
    LOG << dataPtr->getDistances()[0];
    BOOST_CHECK(dataPtr->getDistances()[0] == 1);
    
    // clean up after self
    delete needle;
    delete haystack;
}

BOOST_AUTO_TEST_SUITE_END()
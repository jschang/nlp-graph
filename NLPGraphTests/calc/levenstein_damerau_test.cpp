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

#if RUN_TEST_ALL == 1 || RUN_TEST_CALC_LEVENSTEIN_DAMERAU == 1

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

BOOST_AUTO_TEST_SUITE( nlpgraph_calc_levenstein_damerau ) 

BOOST_AUTO_TEST_CASE( calc_test )
{   
    LoggerType logger = LoggerType(boost::log::keywords::channel="nlpgraph_calc_levenstein_damerau");
    
    // get the best device
    OpenCLDeviceInfoType deviceInfo = OpenCLDeviceInfoType();
    OpenCL::bestDeviceInfo(deviceInfo);
    OpenCL::log(deviceInfo);
    
    // spin up a context
    context bContext = context(OpenCL::contextWithDeviceInfo(deviceInfo));    
    
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
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(
            new LevensteinDamerauData(
                bContext,
                width, 
                1,
                haystackSize, 
                (uint64_t*)&needle[0], 
                (uint64_t*)&haystack[0]
            ));
        alg.calculate(dataPtr);
        LOG << "needle:" << NLPGraph::Util::String::str(needle,4);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->distances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->operations(),4*3);
        BOOST_CHECK(dataPtr->distances()[0] == 0);
        LevensteinDamerauReconstructDataPtr reconPtr = LevensteinDamerauReconstructDataPtr(
            new LevensteinDamerauReconstructData(
                bContext,
                4,
                1,
                dataPtr->operations(),
                (uint64_t*)&needle[0]
            ));
        alg.reconstruct(reconPtr);
        LOG << "haystack:" << NLPGraph::Util::String::str(haystack,4);
        LOG << "recreation:" << NLPGraph::Util::String::str(reconPtr->getResult(),4);
        BOOST_CHECK_EQUAL_COLLECTIONS(haystack,haystack+4,reconPtr->getResult(),reconPtr->getResult()+4);
    }
    { // transposition requires 1 edits
        LOG << "Testing single transposition - middle";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,3,2,4};
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            bContext, width, 1, haystackSize, needle, haystack));
        alg.calculate(dataPtr);
        LOG << "needle:" << NLPGraph::Util::String::str(needle,4);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->distances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->operations(),4*3);
        BOOST_CHECK(dataPtr->distances()[0] == 1);
        LevensteinDamerauReconstructDataPtr reconPtr = LevensteinDamerauReconstructDataPtr(
            new LevensteinDamerauReconstructData(
                bContext,
                4,
                1,
                dataPtr->operations(),
                (uint64_t*)&needle[0]
            ));
        alg.reconstruct(reconPtr);
        LOG << "haystack:" << NLPGraph::Util::String::str(haystack,4);
        LOG << "recreation:" << NLPGraph::Util::String::str(reconPtr->getResult(),4);
        BOOST_CHECK_EQUAL_COLLECTIONS(haystack,haystack+4,reconPtr->getResult(),reconPtr->getResult()+4);
    }
    { // transposition requires 1 edits
        LOG << "Testing single transposition - left edge";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {2,1,3,4};
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            bContext, width, 1, haystackSize, needle, haystack));
        alg.calculate(dataPtr);
        LOG << "needle:" << NLPGraph::Util::String::str(needle,4);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->distances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->operations(),4*3);
        BOOST_CHECK(dataPtr->distances()[0] == 1);
        LevensteinDamerauReconstructDataPtr reconPtr = LevensteinDamerauReconstructDataPtr(
            new LevensteinDamerauReconstructData(
                bContext,
                4,
                1,
                dataPtr->operations(),
                (uint64_t*)&needle[0]
            ));
        alg.reconstruct(reconPtr);
        LOG << "haystack:" << NLPGraph::Util::String::str(haystack,4);
        LOG << "recreation:" << NLPGraph::Util::String::str(reconPtr->getResult(),4);
        BOOST_CHECK_EQUAL_COLLECTIONS(haystack,haystack+4,reconPtr->getResult(),reconPtr->getResult()+4);
    }
    { // transposition requires 1 edits
        LOG << "Testing single transposition - right edge";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,2,4,3};
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            bContext, width, 1, haystackSize, needle, haystack));
        alg.calculate(dataPtr);
        LOG << "needle:" << NLPGraph::Util::String::str(needle,4);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->distances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->operations(),4*3);
        BOOST_CHECK(dataPtr->distances()[0] == 1);
        LevensteinDamerauReconstructDataPtr reconPtr = LevensteinDamerauReconstructDataPtr(
            new LevensteinDamerauReconstructData(
                bContext,
                4,
                1,
                dataPtr->operations(),
                (uint64_t*)&needle[0]
            ));
        alg.reconstruct(reconPtr);
        LOG << "haystack:" << NLPGraph::Util::String::str(haystack,4);
        LOG << "recreation:" << NLPGraph::Util::String::str(reconPtr->getResult(),4);
        BOOST_CHECK_EQUAL_COLLECTIONS(haystack,haystack+4,reconPtr->getResult(),reconPtr->getResult()+4);
    }
    { // single deletion - middle
        LOG << "Testing single deletion - middle";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,2,4,0};
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            bContext, width, 1, haystackSize, needle, haystack));
        alg.calculate(dataPtr);
        LOG << "needle:" << NLPGraph::Util::String::str(needle,4);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->distances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->operations(),4*3);
        BOOST_CHECK(dataPtr->distances()[0] == 1);
        LevensteinDamerauReconstructDataPtr reconPtr = LevensteinDamerauReconstructDataPtr(
            new LevensteinDamerauReconstructData(
                bContext,
                4,
                1,
                dataPtr->operations(),
                (uint64_t*)&needle[0]
            ));
        alg.reconstruct(reconPtr);
        LOG << "haystack:" << NLPGraph::Util::String::str(haystack,4);
        LOG << "recreation:" << NLPGraph::Util::String::str(reconPtr->getResult(),4);
        BOOST_CHECK_EQUAL_COLLECTIONS(haystack,haystack+4,reconPtr->getResult(),reconPtr->getResult()+4);
    }
    { // single deletion - left edge
        LOG << "Testing single deletion - left edge";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {2,3,4,0};
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            bContext, width, 1, haystackSize, needle, haystack));
        alg.calculate(dataPtr);
        LOG << "needle:" << NLPGraph::Util::String::str(needle,4);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->distances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->operations(),4*3);
        BOOST_CHECK(dataPtr->distances()[0] == 1);
        LevensteinDamerauReconstructDataPtr reconPtr = LevensteinDamerauReconstructDataPtr(
            new LevensteinDamerauReconstructData(
                bContext,
                4,
                1,
                dataPtr->operations(),
                (uint64_t*)&needle[0]
            ));
        alg.reconstruct(reconPtr);
        LOG << "haystack:" << NLPGraph::Util::String::str(haystack,4);
        LOG << "recreation:" << NLPGraph::Util::String::str(reconPtr->getResult(),4);
        BOOST_CHECK_EQUAL_COLLECTIONS(haystack,haystack+4,reconPtr->getResult(),reconPtr->getResult()+4);
    }
    { // single deletion - right edge
        LOG << "Testing single deletion - right edge";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,2,3,0};
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            bContext, width, 1, haystackSize, needle, haystack));
        alg.calculate(dataPtr);
        LOG << "needle:" << NLPGraph::Util::String::str(needle,4);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->distances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->operations(),4*3);
        BOOST_CHECK(dataPtr->distances()[0] == 1);
        LevensteinDamerauReconstructDataPtr reconPtr = LevensteinDamerauReconstructDataPtr(
            new LevensteinDamerauReconstructData(
                bContext,
                4,
                1,
                dataPtr->operations(),
                (uint64_t*)&needle[0]
            ));
        alg.reconstruct(reconPtr);
        LOG << "haystack:" << NLPGraph::Util::String::str(haystack,4);
        LOG << "recreation:" << NLPGraph::Util::String::str(reconPtr->getResult(),4);
        BOOST_CHECK_EQUAL_COLLECTIONS(haystack,haystack+4,reconPtr->getResult(),reconPtr->getResult()+4);
    }
    { // single deletion - middle
        LOG << "Testing single deletion, repetition - middle";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,2,2,4};
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            bContext, width, 1, haystackSize, needle, haystack));
        alg.calculate(dataPtr);
        LOG << "needle:" << NLPGraph::Util::String::str(needle,4);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->distances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->operations(),4*3);
        BOOST_CHECK(dataPtr->distances()[0] == 1);
        LevensteinDamerauReconstructDataPtr reconPtr = LevensteinDamerauReconstructDataPtr(
            new LevensteinDamerauReconstructData(
                bContext,
                4,
                1,
                dataPtr->operations(),
                (uint64_t*)&needle[0]
            ));
        alg.reconstruct(reconPtr);
        LOG << "haystack:" << NLPGraph::Util::String::str(haystack,4);
        LOG << "recreation:" << NLPGraph::Util::String::str(reconPtr->getResult(),4);
        BOOST_CHECK_EQUAL_COLLECTIONS(haystack,haystack+4,reconPtr->getResult(),reconPtr->getResult()+4);
    }
    { // single deletion - middle
        LOG << "Testing single deletion, insertion - middle";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,2,5,4};
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            bContext, width, 1, haystackSize, needle, haystack));
        alg.calculate(dataPtr);
        LOG << "needle:" << NLPGraph::Util::String::str(needle,4);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->distances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->operations(),4*3);
        BOOST_CHECK(dataPtr->distances()[0] == 1);
        LevensteinDamerauReconstructDataPtr reconPtr = LevensteinDamerauReconstructDataPtr(
            new LevensteinDamerauReconstructData(
                bContext,
                4,
                1,
                dataPtr->operations(),
                (uint64_t*)&needle[0]
            ));
        alg.reconstruct(reconPtr);
        LOG << "haystack:" << NLPGraph::Util::String::str(haystack,4);
        LOG << "recreation:" << NLPGraph::Util::String::str(reconPtr->getResult(),4);
        // this is actually correct, there is only one operation: "substitution"
        // 2015-08-16 - yeah, but 2 is easier...we'll use two for now: "repetition" and "deletion"
        BOOST_CHECK_EQUAL_COLLECTIONS(haystack,haystack+4,reconPtr->getResult(),reconPtr->getResult()+4);
    }
    { // single deletion - left edge
        LOG << "Testing single deletion, insertion - left edge";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {5,2,3,4};
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            bContext, width, 1, haystackSize, needle, haystack));
        alg.calculate(dataPtr);
        LOG << "needle:" << NLPGraph::Util::String::str(needle,4);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->distances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->operations(),4*3);
        BOOST_CHECK(dataPtr->distances()[0] == 1);
        LevensteinDamerauReconstructDataPtr reconPtr = LevensteinDamerauReconstructDataPtr(
            new LevensteinDamerauReconstructData(
                bContext,
                4,
                1,
                dataPtr->operations(),
                (uint64_t*)&needle[0]
            ));
        alg.reconstruct(reconPtr);
        LOG << "haystack:" << NLPGraph::Util::String::str(haystack,4);
        LOG << "recreation:" << NLPGraph::Util::String::str(reconPtr->getResult(),4);
        BOOST_CHECK_EQUAL_COLLECTIONS(haystack,haystack+4,reconPtr->getResult(),reconPtr->getResult()+4);
    }
    { // single deletion - right edge
        LOG << "Testing single deletion, insertion - right edge";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,2,3,5};
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            bContext, width, 1, haystackSize, needle, haystack));
        alg.calculate(dataPtr);
        LOG << "needle:" << NLPGraph::Util::String::str(needle,4);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->distances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->operations(),4*3);
        BOOST_CHECK(dataPtr->distances()[0] == 1);
        LevensteinDamerauReconstructDataPtr reconPtr = LevensteinDamerauReconstructDataPtr(
            new LevensteinDamerauReconstructData(
                bContext,
                4,
                1,
                dataPtr->operations(),
                (uint64_t*)&needle[0]
            ));
        alg.reconstruct(reconPtr);
        LOG << "haystack:" << NLPGraph::Util::String::str(haystack,4);
        LOG << "recreation:" << NLPGraph::Util::String::str(reconPtr->getResult(),4);
        BOOST_CHECK_EQUAL_COLLECTIONS(haystack,haystack+4,reconPtr->getResult(),reconPtr->getResult()+4);
    }
    { // single deletion - left edge
        LOG << "Testing single deletion, repetition - left edge";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {2,2,3,4};
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            bContext, width, 1, haystackSize, needle, haystack));
        alg.calculate(dataPtr);
        LOG << "needle:" << NLPGraph::Util::String::str(needle,4);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->distances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->operations(),4*3);
        BOOST_CHECK(dataPtr->distances()[0] == 1);
        LevensteinDamerauReconstructDataPtr reconPtr = LevensteinDamerauReconstructDataPtr(
            new LevensteinDamerauReconstructData(
                bContext,
                4,
                1,
                dataPtr->operations(),
                (uint64_t*)&needle[0]
            ));
        alg.reconstruct(reconPtr);
        LOG << "haystack:" << NLPGraph::Util::String::str(haystack,4);
        LOG << "recreation:" << NLPGraph::Util::String::str(reconPtr->getResult(),4);
        BOOST_CHECK_EQUAL_COLLECTIONS(haystack,haystack+4,reconPtr->getResult(),reconPtr->getResult()+4);
    }
    { // single deletion - right edge
        LOG << "Testing single deletion, repetition - right edge";
        uint width=4;
        uint haystackSize=1;
        uint64_t needle[] = {1,2,3,4};
        uint64_t haystack[] = {1,2,4,4};
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            bContext, width, 1, haystackSize, needle, haystack));
        alg.calculate(dataPtr);
        LOG << "needle:" << NLPGraph::Util::String::str(needle,4);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->distances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->operations(),4*3);
        BOOST_CHECK(dataPtr->distances()[0] == 1);
        LevensteinDamerauReconstructDataPtr reconPtr = LevensteinDamerauReconstructDataPtr(
            new LevensteinDamerauReconstructData(
                bContext,
                4,
                1,
                dataPtr->operations(),
                (uint64_t*)&needle[0]
            ));
        alg.reconstruct(reconPtr);
        LOG << "haystack:" << NLPGraph::Util::String::str(haystack,4);
        LOG << "recreation:" << NLPGraph::Util::String::str(reconPtr->getResult(),4);
        int res = BOOST_CHECK_EQUAL_COLLECTIONS(haystack,haystack+4,reconPtr->getResult(),reconPtr->getResult()+4);
        LOG << "result: " << res;
    }
}

/**
 * When something fails the stress test, it ends up here
 */
BOOST_AUTO_TEST_CASE( stress_test_crosshooks_recreation_tests ) {

    LoggerType logger = LoggerType(boost::log::keywords::channel="nlpgraph_calc_levenstein_damerau");
    
    // get the best device
    OpenCLDeviceInfoType deviceInfo = OpenCLDeviceInfoType();
    OpenCL::bestDeviceInfo(deviceInfo);
    OpenCL::log(deviceInfo);
    
    // spin up a context
    context bContext = context(OpenCL::contextWithDeviceInfo(deviceInfo));    
    
    // fire up the calculator
    LevensteinDamerau alg(bContext);
    
    // flip to on, if you're interested in fixing an issue in the cl code
    alg.clLogOn = true;
    
    uint width=10;
    uint haystackSize=1;
    uint testCount = 1;
    /**
     * needle/haystack pairs
     */
    uint64_t pairs[][10] = {
        {7561,7260,     7561,7260,8398,7561,3807,5209,4489,7260          },
        {7561,7260,8398,7561,3807,5209,4489,7260,1501,8241}
    };
    
    for(int i = 0, nd = (testCount*2); i < nd; i+=2) {
        uint64_t *needle = (uint64_t*)&pairs[i];
        uint64_t *haystack = (uint64_t*)&pairs[i+1];
        LevensteinDamerauDataPtr dataPtr = LevensteinDamerauDataPtr(new LevensteinDamerauData(
            bContext, width, 1, haystackSize, needle, haystack));
        alg.calculate(dataPtr);
        LOG << "needle:" << NLPGraph::Util::String::str(needle,width);
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->distances(),1);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->operations(),width*kLevensteinOperationsWidth);
        LevensteinDamerauReconstructDataPtr reconPtr = LevensteinDamerauReconstructDataPtr(
            new LevensteinDamerauReconstructData(
                bContext,
                width,
                1,
                dataPtr->operations(),
                needle
            ));
        alg.reconstruct(reconPtr);
        LOG << "haystack:" << NLPGraph::Util::String::str(haystack,width);
        LOG << "recreation:" << NLPGraph::Util::String::str(reconPtr->getResult(),width);
        int res = BOOST_CHECK_EQUAL_COLLECTIONS(haystack,haystack+width,reconPtr->getResult(),reconPtr->getResult()+4);
        if(res!=1) {
            LOG << "result: " << res;
        }
    }
}

BOOST_AUTO_TEST_CASE( stress_test ) {
return;
    LoggerType logger = LoggerType(boost::log::keywords::channel="nlpgraph_calc_levenstein_damerau");

    // get the best device
    OpenCLDeviceInfoType deviceInfo = OpenCLDeviceInfoType();
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
    
    NLPGraph::Math::GeneratorIntegerPtr<uint64_t> getRand = NLPGraph::Util::Math::rangeRandGen<uint64_t>(1000,9999);
    NLPGraph::Math::GeneratorRealPtr getChance = NLPGraph::Util::Math::rangeRandGen(0.0f,1.0f);
    NLPGraph::Math::GeneratorIntegerPtr<uint> getNeedleWidth = NLPGraph::Util::Math::rangeRandGen<uint>(0,testWidth);
    
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
        LevensteinDamerauDataPtr dataPtr(new LevensteinDamerauData(
            bContext, testWidth, 1, testSize, needle, haystack));
        alg.calculate(dataPtr);
        timespec end = TimeHelper::getTimeStruct();
        LOG << "distances:" << NLPGraph::Util::String::str(dataPtr->distances(),testSize);
        LOG << "operations:" << NLPGraph::Util::String::str(dataPtr->operations(),dataPtr->operationWidth()*dataPtr->operationsCount());
        LOG << "time: " << (end.tv_nsec - start.tv_nsec);
        BOOST_CHECK( dataPtr->distances()[0] >= 0 );
        
        LevensteinDamerauReconstructDataPtr reconPtr = LevensteinDamerauReconstructDataPtr(
            new LevensteinDamerauReconstructData(
                bContext,
                testWidth,
                1,
                dataPtr->operations(),
                (uint64_t*)&needle[0]
            ));
        alg.reconstruct(reconPtr);
        LOG << "haystack:" << NLPGraph::Util::String::str(haystack,testWidth);
        LOG << "recreation:" << NLPGraph::Util::String::str(reconPtr->getResult(),testWidth);
        int res = BOOST_CHECK_EQUAL_COLLECTIONS(haystack,haystack+testWidth,reconPtr->getResult(),reconPtr->getResult()+testWidth);
        LOG << "result: " << res;
        if(res!=1) {
            NLPGraph::NLPGraphException exc;
            exc.msg = "Collection mismatch";
            throw exc;
        }
    }
    
    delete needle;
    delete haystack;
}

BOOST_AUTO_TEST_CASE( perf_test ) {
return;
    LoggerType logger = LoggerType(boost::log::keywords::channel="nlpgraph_calc_levenstein_damerau");

    // get the best device
    OpenCLDeviceInfoType deviceInfo = OpenCLDeviceInfoType();
    OpenCL::bestDeviceInfo(deviceInfo);
    
    // spin up a context
    context bContext;
    bContext = context(OpenCL::contextWithDeviceInfo(deviceInfo)); 
    
    NLPGraph::Math::GeneratorIntegerPtr<uint64_t> getRand = NLPGraph::Util::Math::rangeRandGen<uint64_t>(1000,9999);   
    
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
            bContext, testWidth, 1, testSize, needle, haystack));
    alg.calculate(dataPtr);
    timespec end = TimeHelper::getTimeStruct();
    LOG << "time: " << (end.tv_nsec - start.tv_nsec);
    
    LOG << dataPtr->distances()[0];
    BOOST_CHECK(dataPtr->distances()[0] == 1);
    
    // clean up after self
    delete needle;
    delete haystack;
}

BOOST_AUTO_TEST_SUITE_END()

#endif
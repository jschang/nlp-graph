#define BOOST_LOG_DYN_LINK
#include <util/opencl.h>
#include <calc/levenstein_damerau.h>
#include <nlpgraph.h>
#include <boost/test/unit_test.hpp>
#include <boost/compute.hpp>
#include <boost/random.hpp>
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

BOOST_AUTO_TEST_CASE( test_calc )
{       
    LoggerType logger(boost::log::keywords::channel="nlpgraph_calc_levenstein_damerau");

    // get the best device
    OpenCLDeviceInfoType deviceInfo;
    OpenCL::bestDeviceInfo(deviceInfo);
    
    // spin up a context
    context bContext;
    bContext = context(OpenCL::contextWithDeviceInfo(deviceInfo));    
    
    // create my random data
    struct timeval tv;
    boost::random::mt19937 randGen(tv.tv_usec);
    uniform_int<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());
    boost::variate_generator<boost::random::mt19937&, uniform_int<uint64_t>> getRand(randGen, dist);
    uint64_t test = getRand();
    
    // fire up the calculator
    LevensteinDamerau alg(bContext);
}

BOOST_AUTO_TEST_SUITE_END()
#define BOOST_LOG_DYN_LINK

#include <util/opencl.h>
#include <util/time_helper.h>
#include <util/math.h>
#include <neural/neural.h>
#include <boost/test/unit_test.hpp>
#include <boost/compute.hpp>

#include <nlpgraph.h>
#include <util/string.h>
#include "../nlpgraph_tests.h"

#if RUN_TEST_ALL == 1 || RUN_TEST_NEURAL_NETWORK == 1

#define LOG BOOST_LOG_SEV(logger,severity_level::normal) << __PRETTY_FUNCTION__ << " "

using namespace NLPGraph::Dao;
using namespace NLPGraph::Dto;
using namespace NLPGraph::Util;
using namespace NLPGraph::Neural;
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

BOOST_AUTO_TEST_SUITE( nlpgraph_neural_network )

BOOST_AUTO_TEST_CASE( calc_test )
{
    LoggerType logger(boost::log::keywords::channel="nlpgraph_neural_network");
    NeuronPtr n1 = NeuronPtr(new Neuron());
    n1->id(1000);
    n1->threshold(3.0);
    BOOST_CHECK_CLOSE(3.0, n1->threshold(), .000001);
    BOOST_CHECK_CLOSE(3.0, n1->threshold(), .000001);
    BOOST_CHECK_EQUAL(1000, n1->id());
    BOOST_CHECK_EQUAL(1000, n1->id());
    
    NeuronPtr n2 = NeuronPtr(new Neuron());
    n2->id(1001);
    n2->threshold(6.0);
    
    SynapsePtr s = SynapsePtr(new Synapse());
    s->weight(4.0);
    BOOST_CHECK_CLOSE(4.0, s->weight(), .000001);
    BOOST_CHECK_CLOSE(4.0, s->weight(), .000001);
}
    
BOOST_AUTO_TEST_SUITE_END()

#endif
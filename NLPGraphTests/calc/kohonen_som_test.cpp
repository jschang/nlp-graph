#define BOOST_LOG_DYN_LINK

#include <util/opencl.h>
#include <util/time_helper.h>
#include <util/math.h>
#include <calc/kohonen_som.h>
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

struct Fixture {
    Fixture() {
        BOOST_TEST_MESSAGE("Fixture setup");
    }
    ~Fixture() {
        BOOST_TEST_MESSAGE("Fixture teardown");
    }
};

BOOST_AUTO_TEST_SUITE( nlpgraph_calc_kohonen_som )

BOOST_AUTO_TEST_CASE( calc_test )
{
    LoggerType logger(boost::log::keywords::channel="nlpgraph_calc_kohonen_som");
    
    // get the best device
    OpenCLDeviceInfo deviceInfo;
    OpenCL::bestDeviceInfo(deviceInfo);
    
    // spin up a context
    context bContext;
    bContext = context(OpenCL::contextWithDeviceInfo(deviceInfo));
    
    // fire up the calculator
    KohonenSOM alg(bContext);
    
    // flip to on, if you're interested in fixing an issue in the cl code
    alg.clLogOn = true;
    
    /*
    KohonenSOMData(const boost::compute::context &context, 
            const std::vector<double> &nodeWeights, // product(mapDimensions) * nodeWidth
            const std::vector<uint32_t> &mapDimensions, 
            const int nodeWidth)
            */
    double weights[] = {
        1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,
        11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,
        21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,
    };
    std::vector<double> vWeights( weights, weights+(sizeof(weights)/sizeof(double)) );
    LOG << "Weights size: " << vWeights.size();
    uint32_t dimensions[] = {3,3};
    std::vector<uint32_t> vDims( dimensions, dimensions+(sizeof(dimensions)/sizeof(uint32_t)) );
    LOG << "Dimensions size: " << vDims.size();
    KohonenSOMDataPtr data(new KohonenSOMData(
        bContext,
        vWeights,
        vDims,
        3
    ));
    
    /*
    KohonenSOMSampleData(const boost::compute::context &context, 
            const std::vector<double> &sampleData, 
            const uint sampleWidth,
            const uint sampleCount)
            */
    double samples[] = {12.0,12.0,13.0,29.0,28.0,29.0};
    std::vector<double> vSampleData(samples, samples+(sizeof(samples)/sizeof(double)));
    LOG << "Samples size: " << vSampleData.size();
    KohonenSOMSampleDataPtr sampleData(new KohonenSOMSampleData(
        bContext,
        vSampleData,
        3
    ));
            
    // KohonenSOMResultPtr map(const KohonenSOMDataPtr &data, const KohonenSOMSampleDataPtr &sampleData);
    KohonenSOMResultPtr result = alg.map(data,sampleData);     
    LOG << "Result distances size: " << result->distances()->size();
    for(int i = 0; i<result->distances()->size(); i++) {
        LOG << "distances[" << i << "] : " << (*result->distances())[i];
    }
    LOG << "Result bmus size: " << result->indexes()->size();
    for(int i = 0; i<result->distances()->size(); i++) {
        std::ostringstream accum;
        for(int j=0; j<(*result->indexes())[i].size(); j++)
            accum << "," << (*result->indexes())[i][j];
        LOG << "indexes[" << i << "] : " << accum.str();
    }
    
    alg.updateWeights(data,sampleData,result,3,1);
    
    std::vector<double> newWeights;
    data->fromClMem(alg.commandQueue(),newWeights);

    std::ostringstream accum;
    for(int i = 0; i<newWeights.size(); i++) {
        accum << "," << newWeights[i];
    }
    LOG << "new weights: " << accum.str();
}
    
BOOST_AUTO_TEST_SUITE_END()
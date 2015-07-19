#define BOOST_LOG_DYN_LINK
#include <util/logger.h>
#include <util/opencl.h>
#include <nlpgraph.h>
#include <boost/compute.hpp>
#include <boost/test/unit_test.hpp>

#define LOG BOOST_LOG_SEV(logger,NLPGraph::Util::severity_level::normal) << __PRETTY_FUNCTION__ << " "

BOOST_AUTO_TEST_SUITE( nlpgraph_util_opencl )

BOOST_AUTO_TEST_CASE( do_stuff )
{
    NLPGraph::Util::LoggerType logger(boost::log::keywords::channel="nlpgraph_util_opencl::do_stuff");
    
    NLPGraph::Util::OpenCLDeviceInfoType deviceInfo;
    NLPGraph::Util::OpenCL::bestDeviceInfo(deviceInfo);
    
    BOOST_CHECK(deviceInfo.supportsVer1_1);
    
    boost::compute::device device(deviceInfo.id,false);
    LOG << "Device name: " << device.get_info<std::string>(CL_DEVICE_NAME);
}

BOOST_AUTO_TEST_SUITE_END()
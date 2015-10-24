//
//  kohonen_som.cpp
//  NLPGraph
//
//  Created by Jonathan Schang on 10/24/15.
//
//

//
//  levenstein_damerau.cpp
//  NLPGraph
//
//  Created by Jonathan Schang on 6/25/15.
//  Copyright (c) 2015 Jonathan Schang. All rights reserved.
//

#define BOOST_LOG_DYN_LINK
#include "kohonen_som.h"
#include "../util/opencl.h"
#include "../util/string.h"
#include <boost/compute.hpp>
#include <exception>

#define LOG_E BOOST_LOG_SEV(m_logger,severity_level::critical) << __PRETTY_FUNCTION__ << " "
#define LOG_I BOOST_LOG_SEV(m_logger,severity_level::normal) << __PRETTY_FUNCTION__ << " "

using namespace NLPGraph::Util;

namespace NLPGraph {
namespace Calc {
        
const uint CL_LOG_ON         = 0b00000001;
const uint CL_LOG_ERROR_ONLY = 0b00000010;

const char *kKohonenSOMOpenCLHeader = BOOST_COMPUTE_STRINGIZE_SOURCE();
const char *kKohonenSOMOpenCLSource = BOOST_COMPUTE_STRINGIZE_SOURCE(
    __kernel void calc_kohonen_som_bmu(
        ) {
    }
);
const char *kKohonenSOMOpenCLSupprtSource = BOOST_COMPUTE_STRINGIZE_SOURCE();
    
KohonenSOM::KohonenSOM(boost::compute::context &context)
        : m_logger(boost::log::keywords::channel="NLPGraph::Calc::KohonenSOM") {
    m_context = context;
    clLogOn = false;
    clLogErrorOnly = 0;
    boost::compute::device dev(m_context.get_device());
    m_commandQueue = boost::compute::command_queue(m_context, dev);
    
    int headerSize = sizeof(char)*strlen(kKohonenSOMOpenCLHeader);
    int sourceSize = sizeof(char)*strlen(kKohonenSOMOpenCLSource);
    int supportSize = sizeof(char)*strlen(kKohonenSOMOpenCLSupprtSource);
    char * source = (char *)malloc(headerSize+supportSize+sourceSize+1);
    memset(source,0,headerSize+sourceSize+supportSize+1);
    memcpy(source, kKohonenSOMOpenCLHeader, headerSize);
    //memcpy(source+headerSize, kLevensteinDamerauOpenCLSupprtSource, supportSize);
    memcpy(source+headerSize+supportSize, kKohonenSOMOpenCLSource, sourceSize);
    
    LOG_I << "Source:\n" << source;
    
    // I would have used link, but NVIDIA doesn't support OpenCL 1.2
    // and this will prolly end up running on AWS hardware a bunch
    m_program = OpenCL::createAndBuildProgram(source,m_context);
    m_kernel = boost::compute::kernel(m_program, "calc_kohonen_som_bmu");
}
KohonenSOM::~KohonenSOM() {
}

int KohonenSOM::train(KohonenSOMDataPtr data, const std::vector<double> &sampleData) {
    return 0;
}
std::vector<int> KohonenSOM::map(KohonenSOMDataPtr data, const std::vector<double> &sample) {
    return std::vector<int>();
}
        
}}

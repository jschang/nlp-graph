//
//  smith_waterman.cpp
//  NLPGraph
//
//  Created by Jonathan Schang on 2/29/16.
//
//

#include "smith_waterman/smith_waterman_data.h"
#include "smith_waterman.h"

#define BOOST_LOG_DYN_LINK
#include "../util/opencl.h"
#include "../util/string.h"
#include <boost/compute.hpp>
#include <exception>

#define LOG_E BOOST_LOG_SEV((*m_logger),severity_level::critical) << __PRETTY_FUNCTION__ << " "
#define LOG_I BOOST_LOG_SEV((*m_logger),severity_level::normal) << __PRETTY_FUNCTION__ << " "

using namespace NLPGraph::Util;

namespace NLPGraph {
    namespace Calc {

const uint CL_LOG_ON         = 0b00000001;
const uint CL_LOG_ERROR_ONLY = 0b00000010;
        
/* ************* */
/* HEADER SOURCE */
/* ************* */
const char *kSmithWatermanOpenCLHeader = BOOST_COMPUTE_STRINGIZE_SOURCE();
const char *kSmithWatermanOpenCLSource = BOOST_COMPUTE_STRINGIZE_SOURCE();

/* ****************** */
/* SUPPORT LIB SOURCE */
/* ****************** */
const char *kSmithWatermanOpenCLSupprtSource = BOOST_COMPUTE_STRINGIZE_SOURCE();

SmithWaterman::SmithWaterman(const boost::compute::context &context) {
    
    m_logger.reset(new Util::LoggerType(boost::log::keywords::channel="NLPGraph::Calc::SmithWaterman"));
    
    m_context.reset(new boost::compute::context(context));
    
    clLogOn = false;
    clLogErrorOnly = 0;
    boost::compute::device dev = boost::compute::device(m_context->get_device());
    
    m_commandQueue.reset(new boost::compute::command_queue(*m_context, dev));
    
    int headerSize = sizeof(char)*strlen(kSmithWatermanOpenCLHeader);
    int sourceSize = sizeof(char)*strlen(kSmithWatermanOpenCLSource);
    int supportSize = sizeof(char)*strlen(kSmithWatermanOpenCLSupprtSource);
    
    char * source = 0;
    try {
        source = (char *)malloc(headerSize+supportSize+sourceSize+1);
        
        memset(source,0,headerSize+sourceSize+supportSize+1);
        
        memcpy(source, kSmithWatermanOpenCLHeader, headerSize);
        memcpy(source+headerSize, kSmithWatermanOpenCLSupprtSource, supportSize);
        memcpy(source+headerSize+supportSize, kSmithWatermanOpenCLSource, sourceSize);
        
        // LOG_I << "Source:\n" << source;
        
        // I would have used link, but NVIDIA doesn't support OpenCL 1.2
        // and this will prolly end up running on AWS hardware a bunch
        boost::compute::program p = OpenCL::createAndBuildProgram(source,*m_context);
        m_program.reset(new boost::compute::program(p));
        m_kernelCalc.reset(new boost::compute::kernel(*m_program, "calc_smith_waterman"));
        
        delete source;
    } catch(...) {
        if(source!=0) delete source;
        throw;
    }
}
int SmithWaterman::calculate(SmithWatermanDataPtr data) {
    
    uint logLength      = 40000;
    
    int result = 0;
    
    OpenCLExceptionType except;
    except.msg = "";
    
    char * log                 = 0;
    cl_mem logBuf              = 0;
    
    LOG_I << "referenceWidth   : " << data->referenceWidth();
    LOG_I << "candidatesCount  : " << data->candidatesCount();
    
    try {
        OpenCL::alloc<char>(*m_context, logLength, (char **)&log, (cl_mem*)&logBuf, (int)CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR);
        
        uint flags =   (clLogOn        ? CL_LOG_ON         : 0) 
                     | (clLogErrorOnly ? CL_LOG_ERROR_ONLY : 0);
        
        int parm = 0;
        m_kernelCalc->set_arg(parm++,flags);
        m_kernelCalc->set_arg(parm++,logBuf);
        m_kernelCalc->set_arg(parm++,logLength);
        m_kernelCalc->set_arg(parm++,data->referenceWidth());
        m_kernelCalc->set_arg(parm++,data->operationsWidth());
        m_kernelCalc->set_arg(parm++,data->candidatesCount());
        m_kernelCalc->set_arg(parm++,data->clReference());
        m_kernelCalc->set_arg(parm++,data->clCandidates());
        m_kernelCalc->set_arg(parm++,data->clCostMatrix());
        m_kernelCalc->set_arg(parm++,data->clDistsAndOps());
        
        size_t gwo[2] = {0,0};
        size_t gws[2] = {data->haystackCount(),data->needleCount()};
        size_t lws[2] = {1,1};
        m_commandQueue->enqueue_nd_range_kernel(*m_kernelCalc, 2, (size_t*)&gwo, (size_t*)&gws, (size_t*)&lws, boost::compute::wait_list());
        
        OpenCL::read<char>(*m_commandQueue, logLength, log, logBuf);
        data->read(*m_commandQueue);
        
        if(clLogOn) {
            LOG_I << "Run log:\n" << log;
        }
        
        if(except.msg.length()>0) {
            throw except;
        }
        
        delete log;
        clReleaseMemObject (logBuf);
        
    } catch(...) {
        
        if(!log)                 delete log;
        if(!logBuf)              clReleaseMemObject (logBuf);
        
        throw;
    }    
    return result;
}

}}
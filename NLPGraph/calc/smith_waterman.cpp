//
//  smith_waterman.cpp
//  NLPGraph
//
//  Created by Jonathan Schang on 2/29/16.
//
//

#define BOOST_LOG_DYN_LINK

#include "smith_waterman/smith_waterman_data.h"
#include "smith_waterman.h"

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
        
#include "smith_waterman/smith_waterman_util.cl.h"
#include "smith_waterman/smith_waterman_cost.cl.h"
#include "smith_waterman/smith_waterman_matrices.cl.h"
#include "smith_waterman/smith_waterman_dists.cl.h"

SmithWaterman::SmithWaterman(const boost::compute::context &context) {
    
    m_logger.reset(new Util::LoggerType(boost::log::keywords::channel="NLPGraph::Calc::SmithWaterman"));
    
    m_context.reset(new boost::compute::context(context));
    
    clLogOn = false;
    clLogErrorOnly = 0;
    boost::compute::device dev = boost::compute::device(m_context->get_device());
    
    m_commandQueue.reset(new boost::compute::command_queue(*m_context, dev));
    
    size_t utilSrcSize = sizeof(char)*strlen(kSmithWatermanUtilOpenCLSource);
    size_t costMatrixSrcSize = sizeof(char)*strlen(kSmithWatermanCostMatrixOpenCLSource);
    size_t createMatricesSrcSize = sizeof(char)*strlen(kSmithWatermanCreateMatricesOpenCLSource);
    size_t determineDistancesSrcSize = sizeof(char)*strlen(kSmithWatermanDetermineDistancesOpenCLSource);
    
    char * source = 0;
    try {
        source = (char *)malloc(utilSrcSize+costMatrixSrcSize+createMatricesSrcSize+determineDistancesSrcSize+1);
        
        memset(source,0,utilSrcSize+costMatrixSrcSize+createMatricesSrcSize+determineDistancesSrcSize+1);
        
        memcpy(source, 
            kSmithWatermanUtilOpenCLSource, utilSrcSize);
        memcpy(source+utilSrcSize,
            kSmithWatermanCostMatrixOpenCLSource, costMatrixSrcSize);
        memcpy(source+utilSrcSize+costMatrixSrcSize,
            kSmithWatermanCreateMatricesOpenCLSource, createMatricesSrcSize);
        memcpy(source+utilSrcSize+costMatrixSrcSize+createMatricesSrcSize,
            kSmithWatermanDetermineDistancesOpenCLSource, determineDistancesSrcSize);

        // I would have used link, but NVIDIA doesn't support OpenCL 1.2
        // and this will prolly end up running on AWS hardware a bunch
        boost::compute::program p = OpenCL::createAndBuildProgram(source,*m_context);
        m_program.reset(new boost::compute::program(p));
        
        m_kernelCostMatrix.reset(new boost::compute::kernel(*m_program, "calc_smith_waterman_cost_matrix"));
        m_kernelDistances.reset(new boost::compute::kernel(*m_program, "calc_smith_waterman_distances"));
        m_kernelMatrices.reset(new boost::compute::kernel(*m_program, "calc_smith_waterman_matrices"));
        
        delete source;
    } catch(...) {
        if(source!=0) delete source;
        throw;
    }
}

SmithWaterman::~SmithWaterman() {
}

int SmithWaterman::createCostMatrix(SmithWatermanDataPtr data) {
    
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
        m_kernelCostMatrix->set_arg(parm++,flags);
        m_kernelCostMatrix->set_arg(parm++,logBuf);
        m_kernelCostMatrix->set_arg(parm++,logLength);
        m_kernelCostMatrix->set_arg(parm++,data->referenceWidth());
        m_kernelCostMatrix->set_arg(parm++,data->operationsWidth());
        m_kernelCostMatrix->set_arg(parm++,data->candidatesCount());
        m_kernelCostMatrix->set_arg(parm++,data->uniqueCount());
        m_kernelCostMatrix->set_arg(parm++,data->clReference());
        m_kernelCostMatrix->set_arg(parm++,data->clCandidates());
        m_kernelCostMatrix->set_arg(parm++,data->clCostMatrix());
        m_kernelCostMatrix->set_arg(parm++,data->clDistsAndOps());
        m_kernelCostMatrix->set_arg(parm++,data->clUniques());
        
        size_t gwo[2] = {0};
        size_t gws[2] = {data->candidatesCount()};
        size_t lws[2] = {1};
        m_commandQueue->enqueue_nd_range_kernel(*m_kernelCostMatrix, 1, (size_t*)&gwo, (size_t*)&gws, (size_t*)&lws, boost::compute::wait_list());
        
        OpenCL::read<char>(*m_commandQueue, logLength, log, logBuf);
        
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

int SmithWaterman::createMatrices(SmithWatermanDataPtr data) {
    
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
        m_kernelMatrices->set_arg(parm++,flags);
        m_kernelMatrices->set_arg(parm++,logBuf);
        m_kernelMatrices->set_arg(parm++,logLength);
        m_kernelMatrices->set_arg(parm++,data->referenceWidth());
        m_kernelMatrices->set_arg(parm++,data->operationsWidth());
        m_kernelMatrices->set_arg(parm++,data->candidatesCount());
        m_kernelMatrices->set_arg(parm++,data->uniqueCount());
        m_kernelMatrices->set_arg(parm++,data->clReference());
        m_kernelMatrices->set_arg(parm++,data->clCandidates());
        m_kernelMatrices->set_arg(parm++,data->clCostMatrix());
        m_kernelMatrices->set_arg(parm++,data->clDistsAndOps());
        m_kernelMatrices->set_arg(parm++,data->clUniques());
        
        size_t gwo[2] = {0};
        size_t gws[2] = {data->candidatesCount()};
        size_t lws[2] = {1};
        m_commandQueue->enqueue_nd_range_kernel(*m_kernelMatrices, 1, (size_t*)&gwo, (size_t*)&gws, (size_t*)&lws, boost::compute::wait_list());
        
        OpenCL::read<char>(*m_commandQueue, logLength, log, logBuf);
        
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

int SmithWaterman::calculateDistances(SmithWatermanDataPtr data) {
    
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
        m_kernelDistances->set_arg(parm++,flags);
        m_kernelDistances->set_arg(parm++,logBuf);
        m_kernelDistances->set_arg(parm++,logLength);
        m_kernelDistances->set_arg(parm++,data->referenceWidth());
        m_kernelDistances->set_arg(parm++,data->operationsWidth());
        m_kernelDistances->set_arg(parm++,data->candidatesCount());
        m_kernelDistances->set_arg(parm++,data->uniqueCount());
        m_kernelDistances->set_arg(parm++,data->clReference());
        m_kernelDistances->set_arg(parm++,data->clCandidates());
        m_kernelDistances->set_arg(parm++,data->clCostMatrix());
        m_kernelDistances->set_arg(parm++,data->clDistsAndOps());
        m_kernelDistances->set_arg(parm++,data->clUniques());
        
        size_t gwo[2] = {0};
        size_t gws[2] = {data->candidatesCount()};
        size_t lws[2] = {1};
        m_commandQueue->enqueue_nd_range_kernel(*m_kernelDistances, 1, (size_t*)&gwo, (size_t*)&gws, (size_t*)&lws, boost::compute::wait_list());
        
        OpenCL::read<char>(*m_commandQueue, logLength, log, logBuf);
        
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

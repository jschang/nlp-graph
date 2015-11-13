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

const char *kKohonenSOMOpenCLHeader = "";
const char *kKohonenSOMOpenCLSource = BOOST_COMPUTE_STRINGIZE_SOURCE(
    
    inline float _calc_sample_distance(__global float* weights, ulong startIdx, uint nodeWidth, __constant float* sample, ulong sampleIdx) {
        float accum = 0.0f;
        float diff = 0.0f;
        uint i = 0;
        for(i = 0; i<nodeWidth; i++) {
            diff = weights[startIdx+i] - sample[ (sampleIdx*nodeWidth) + i ];
            accum += pow(diff,2);
        }
        accum = pow(accum, .5f);
        return accum;
    }
    
    inline void _calc_coords(uint dimCount, __constant uint* dimSizes, size_t offset, uint* thisCoords) {
        // reversed so, processed as xy, then y
        ulong trim = offset, multi = 0;
        int i = 0, j = 0;
        for(i = dimCount-1; i>=0; i--) {
            multi = 1;
            for(j=i-1; j>=0; j--) {
                multi *= dimSizes[j];
            }
            thisCoords[i] = trim / multi;
            trim = trim % multi; 
        } 
    }
    
    inline float _calc_map_coord_distance(uint dimCount, ulong sampleIdx, __constant uint* bmuCoords, uint* thisCoords) {
        float accum = 0.0f;
        uint i = 0;
        int diff = 0;
        ulong startBmuIdx = dimCount * sampleIdx;
        for(i = 0; i < dimCount; i++) {
            diff = bmuCoords[startBmuIdx+i] - thisCoords[i];
            diff *= diff;
            accum += (float)diff; // THIS line, only, causes CL_DEVICE_NOT_AVAILABLE !!!
        }
        accum = pow(accum,.5f);
        return accum;
    }

    __kernel void calc_kohonen_som_distances(
            // map data
            __global float* weights,      // weights
            uint nodeWidth,               // the number of weights per node
            ulong nodeCount,              // the total number of weights
            __constant float* sampleData, // sample, of nodeWidth wide
            ulong sampleIdx,
            __global float* output        // the output distance of each node to the sample
        ) {
        size_t nodeIndex = get_global_id(0);
        ulong startIdx = nodeIndex * nodeWidth;
        if(nodeIndex>=nodeCount) {
            return;
        }
        output[nodeIndex] = _calc_sample_distance(weights,startIdx,nodeWidth,sampleData,sampleIdx);
    }
    
    __kernel void calc_kohonen_som_update_weights(
            // map data
            __global float* weights,       // weights
            uint nodeWidth,                // the number of weights per node
            uint dimCount,                 // the number of dimensions
            __constant uint* dimSizes,     // the size of each dimension
            __constant float* sampleData,
            ulong sampleIdx,
            __constant uint* bmuCoords,    // the coordinates of the best matching unit, from which we derive offset
            float learningRate,            // calculated on the CPU as per step
            float radius                   // calculated on the CPU as per step
        ) {
        size_t nodeIndex = get_global_id(0);
        ulong startIdx = nodeIndex * nodeWidth;
        
        uint* thisCoords = (uint*)malloc(sizeof(uint)*dimCount);
        memset(thisCoords,0,sizeof(uint)*dimCount);
        
        // determine the coordinates of the offset provided
        if(dimCount!=1) {
            _calc_coords(dimCount,dimSizes,nodeIndex,thisCoords);
        } else {
            thisCoords[0] = nodeIndex;
        }
        
        float distance = _calc_map_coord_distance(dimCount, sampleIdx, bmuCoords, thisCoords);
        if(distance<radius) {
            float influence = exp( (-1*distance)/(2*pow(radius,2.0f)) );
            for(uint i=0;i<nodeWidth;i++) {
                weights[startIdx+i] 
                    = weights[startIdx+i] 
                    + ( influence * learningRate * 
                        ( sampleData[(sampleIdx*nodeWidth)+i] 
                          - weights[startIdx+i]
                        ) 
                    );
            }
        }
    }    
);
const char *kKohonenSOMOpenCLSupprtSource = BOOST_COMPUTE_STRINGIZE_SOURCE();
    
KohonenSOM::KohonenSOM(const boost::compute::context &context)
        : m_logger(boost::log::keywords::channel="NLPGraph::Calc::KohonenSOM") {
    m_context = context;
    clLogOn = false;
    clLogErrorOnly = 0;
    boost::compute::device dev(m_context.get_device());
    m_commandQueue = boost::compute::command_queue(m_context, dev);
    
    int headerSize = sizeof(char)*strlen(kKohonenSOMOpenCLHeader);
    int sourceSize = sizeof(char)*strlen(kKohonenSOMOpenCLSource);
    int supportSize = sizeof(char)*strlen(kKohonenSOMOpenCLSupprtSource);
    char * source = 0;
    try {
        source = (char *)malloc(headerSize+supportSize+sourceSize+1);
        memset(source,0,headerSize+sourceSize+supportSize+1);
        memcpy(source, kKohonenSOMOpenCLHeader, headerSize);
        //memcpy(source+headerSize, kLevensteinDamerauOpenCLSupprtSource, supportSize);
        memcpy(source+headerSize+supportSize, kKohonenSOMOpenCLSource, sourceSize);

        // LOG_I << "Source:\n" << source;
        
        // I would have used link, but NVIDIA doesn't support OpenCL 1.2
        // and this will prolly end up running on AWS hardware a bunch
        m_program = OpenCL::createAndBuildProgram(source,m_context);
        m_mappingKernel = boost::compute::kernel(m_program, "calc_kohonen_som_distances");
        m_weightUpdateKernel = boost::compute::kernel(m_program, "calc_kohonen_som_update_weights");
        
        delete source;
    } catch(...) {
        if(source!=0) delete source;
        throw;
    }
}
KohonenSOM::~KohonenSOM() {
}

void KohonenSOM::train(const KohonenSOMDataPtr &data, const KohonenSOMSampleDataPtr &sampleData) {
}

KohonenSOMResultPtr KohonenSOM::map(const KohonenSOMDataPtr &data, const KohonenSOMSampleDataPtr &sampleData) {

    size_t outputSize = (size_t)data->nodeCount()*sizeof(float);
    std::vector<float> output(data->nodeCount());
    cl_int err = 0;
    cl_mem clOutputData = clCreateBuffer(m_context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,outputSize,(void*)output.data(),&err);
    if (err!=CL_SUCCESS) {
        Util::OpenCLExceptionType except;
        except.msg = "unable to clCreateBuffer clOutputData; ";
        throw except;
    }
    const float zero = 0;
    
    KohonenSOMResultPtr result = KohonenSOMResultPtr(new KohonenSOMResult(m_context,sampleData));
    
    try {
        for(uint32_t sampleIdx=0; sampleIdx < sampleData->count(); sampleIdx++) {
        
            err = clEnqueueFillBuffer(m_commandQueue,clOutputData,&zero,sizeof(float),0,outputSize,0,NULL,NULL);
            if(err!=CL_SUCCESS) {
                OpenCLExceptionType except;
                except.msg = except.msg + "Unable to zero clOutputData; ";
            }
            
            m_mappingKernel.set_arg(0,data->clNodeWeights());
            m_mappingKernel.set_arg(1,data->nodeWidth());
            m_mappingKernel.set_arg(2,data->nodeCount());
            m_mappingKernel.set_arg(3,sampleData->clData());
            m_mappingKernel.set_arg(4,sampleIdx);
            m_mappingKernel.set_arg(5,clOutputData);
            
            m_commandQueue.enqueue_1d_range_kernel(m_mappingKernel, 0, data->nodeCount(), 1);
            
            // find the minimum distance in clOutputData
            err = clEnqueueReadBuffer(m_commandQueue, clOutputData, true, 0, outputSize, output.data(), 0, NULL, NULL);
            if(err!=CL_SUCCESS) {
                OpenCLExceptionType except;
                except.msg = except.msg + "Unable to read logBuf; ";
                throw except;
            }
            
            uint64_t idxOfBMU = 0;
            float lowest = -1.0f;
            for(uint64_t i=0;i<output.size();i++) {
                // LOG_I << "Sample Index: " << sampleIdx << ", Index " << i << " distance: " << output[i];
                if(output[i]<lowest || lowest<0.0f) {
                    lowest = output[i];
                    idxOfBMU = i;
                }
            }
            
            // convert that offset to an nd index vector
            // reversed so, processed as xy, then y
            std::vector<uint32_t> thisCoords(data->mapDimensions()->size());
            uint64_t trim = idxOfBMU, multi = 0;
            for(int64_t i = data->mapDimensions()->size()-1; i>=0; i--) {
                multi = 1;
                for(int64_t j=i-1; j>=0; j--) {
                    multi *= (*data->mapDimensions())[j];
                }
                thisCoords[i] = trim / multi;
                trim = trim % multi; 
            }
            
            // add that to the result set we're building
            (*result->distances())[sampleIdx] = lowest;
            (*result->indexes())[sampleIdx] = std::vector<uint32_t>(thisCoords);
        }
    } catch(...) {
        clReleaseMemObject(clOutputData);
        throw;
    }
    clReleaseMemObject(clOutputData);
    return result;
}

void KohonenSOM::updateWeights(const KohonenSOMDataPtr &data, const KohonenSOMSampleDataPtr &sampleData, const KohonenSOMResultPtr &result, double radius, double learningRate) {
    
    cl_float clLearningRate = learningRate;
    cl_float clRadius = radius;

    result->toClMem(m_context);
    for(uint32_t sampleIdx=0; sampleIdx < result->distances()->size(); sampleIdx++) {
        
        m_weightUpdateKernel.set_arg(0,data->clNodeWeights());
        m_weightUpdateKernel.set_arg(1,data->nodeWidth());
        m_weightUpdateKernel.set_arg(2,(cl_uint)data->mapDimensions()->size());
        m_weightUpdateKernel.set_arg(3,data->clMapDimensions());
        m_weightUpdateKernel.set_arg(4,sampleData->clData());
        m_weightUpdateKernel.set_arg(5,sampleIdx);
        m_weightUpdateKernel.set_arg(6,result->clIndexes());
        m_weightUpdateKernel.set_arg(7,clLearningRate);
        m_weightUpdateKernel.set_arg(8,clRadius);
        
        m_commandQueue.enqueue_1d_range_kernel(m_weightUpdateKernel, 0, data->nodeCount(), 1);
    }
}
        
}}

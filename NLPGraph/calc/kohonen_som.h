//
//  kohonen_som.h
//  NLPGraph
//
//  Created by Jonathan Schang on 10/24/15.
//
//

#ifndef kohonen_som_h
#define kohonen_som_h

#import <numeric>
#include "../nlpgraph.h"
#include "../util/logger.h"
#include "../util/opencl.h"

namespace NLPGraph {
    namespace Calc {
        
class KohonenSOMData {
private:
    // the vector length of each node's weights
    int _nodeWidth;
    
    // OpenCL data
    cl_mem _clNodeWeights = 0;
    uint64_t _nodeCount = 0;
    cl_mem _clMapDimensions = 0;
    boost::shared_ptr< std::vector<uint32_t> > _mapDimensions;
public:
    KohonenSOMData(const boost::compute::context &context, 
            const std::vector<double> &nodeWeights, // product(mapDimensions) * nodeWidth
            const std::vector<uint32_t> &mapDimensions, 
            const int nodeWidth) {
            
        this->_mapDimensions = boost::shared_ptr< std::vector<uint32_t> >( new std::vector<uint32_t>(mapDimensions) );
        this->_nodeWidth = nodeWidth;
        this->_nodeCount = nodeWeights.size()/nodeWidth;
        cl_int err = 0;
        // because i can't cound on the device having cl_khr_fp64
        std::vector<float> floatNodeWeights(nodeWeights.begin(), nodeWeights.end());
        this->_clNodeWeights = clCreateBuffer(context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,(size_t)floatNodeWeights.size()*sizeof(float),(void*)floatNodeWeights.data(),&err);
        if (err!=CL_SUCCESS) {
            Util::OpenCLExceptionType except;
            except.msg = except.msg + "unable to clCreateBuffer _clNodeWeights; ";
            throw except;
        }
        this->_clMapDimensions = clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)mapDimensions.size()*sizeof(uint32_t),(void*)mapDimensions.data(),&err);
        if (err!=CL_SUCCESS) {
            Util::OpenCLExceptionType except;
            except.msg = except.msg + "unable to clCreateBuffer _clMapDimensions; ";
            throw except;
        }
    }
    ~KohonenSOMData() {
        if(_clNodeWeights!=0) {
            clReleaseMemObject(_clNodeWeights);
        }
        if(_clNodeWeights!=0) {
            clReleaseMemObject(_clMapDimensions);
        }
    }
    const int nodeWidth() {
        return _nodeWidth;
    }
    const uint64_t nodeCount() {
        return _nodeCount;
    }
    const std::vector<uint32_t>* mapDimensions() {
        return _mapDimensions.get();
    }
    const cl_mem clNodeWeights() {
        return _clNodeWeights;
    }
    const cl_mem clMapDimensions() {
        return _clMapDimensions;
    }
};

class KohonenSOMSampleData {
private:
    cl_mem _clData = 0;
    uint32_t _width = 0;
    uint32_t _count = 0;
public:
    KohonenSOMSampleData(const boost::compute::context &context, 
            const std::vector<double> &sampleData, 
            const uint32_t sampleWidth) {
            
        this->_width = sampleWidth;
        this->_count = sampleData.size()/sampleWidth;
        cl_int err = 0;
        // because i can't cound on the device having cl_khr_fp64
        std::vector<float> floatSampleData(sampleData.begin(), sampleData.end());
        this->_clData = clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)floatSampleData.size()*sizeof(double),(void*)floatSampleData.data(),&err);
        if (err!=CL_SUCCESS) {
            Util::OpenCLExceptionType except;
            except.msg = "unable to clCreateBuffer _clData; ";
            throw except;
        }
    }
    ~KohonenSOMSampleData() {
        if(this->_clData!=0) {
            clReleaseMemObject(this->_clData);
        }
    }
    cl_mem clData() { return this->_clData; }
    uint width() { return this->_width; }
    uint count() { return this->_count; }
};

class KohonenSOMResult {
private:
    boost::shared_ptr<std::vector<std::vector<uint32_t>>> _indexes;
    boost::shared_ptr<std::vector<float>> _distances;
public:
    KohonenSOMResult(const boost::compute::context &context, const KohonenSOMSampleDataPtr &data) {
        _indexes = boost::shared_ptr< std::vector<std::vector<uint32_t>> >( new std::vector<std::vector<uint32_t>>(data->count()) );
        _distances = boost::shared_ptr< std::vector<float> >( new std::vector<float>(data->count()) );
    }
    ~KohonenSOMResult() {
    }
    std::vector<std::vector<uint32_t>>* indexes() {
        return _indexes.get();
    }
    std::vector<float>* distances() {
        return _distances.get();
    }
};

class KohonenSOM {
private:
    boost::compute::context       m_context;
    boost::compute::kernel        m_mappingKernel;
    boost::compute::kernel        m_weightUpdateKernel;
    boost::compute::program       m_program;
    boost::compute::command_queue m_commandQueue;
    Util::LoggerType              m_logger;
public:
    bool clLogOn;
    bool clLogErrorOnly;
public:
    KohonenSOM(const boost::compute::context &context);
    ~KohonenSOM();
    /**
     * Convenience wrapper that simply iterates over map and updateWeights.
     */
    void train(const KohonenSOMDataPtr &data, const KohonenSOMSampleDataPtr &sampleData);
    /**
     * @return For each sample, in order, a map of node indices, up to max nodes, and the distance from the weights at that nodes index.
     */
    KohonenSOMResultPtr map(const KohonenSOMDataPtr &data, const KohonenSOMSampleDataPtr &sampleData);
    /**
     * Updates map node weights using the result passed in
     */
    void updateWeights(const KohonenSOMDataPtr &data, const KohonenSOMSampleDataPtr &sampleData, const KohonenSOMResultPtr &result);
};

}}

#endif /* kohonen_som_h */

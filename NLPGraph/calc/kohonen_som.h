//
//  kohonen_som.h
//  NLPGraph
//
//  Created by Jonathan Schang on 10/24/15.
//
//

#ifndef kohonen_som_h
#define kohonen_som_h


#include "../nlpgraph.h"
#include "../util/logger.h"
#include "../util/opencl.h"

namespace NLPGraph {
    namespace Calc {
        
class KohonenSOMData {
private:
    // the final map dimensions
    std::vector<int> _mapDimensions;
    // the vector length of each node's weights
    int _nodeWidth;
    
    // OpenCL data
    cl_mem _clNodeWeights = 0;
public:
    KohonenSOMData(const boost::compute::context &context, 
            const std::vector<double> &nodeWeights, // product(mapDimensions) * nodeWidth
            const std::vector<int> &mapDimensions, 
            const int nodeWidth) {
            
        this->_mapDimensions = mapDimensions;
        this->_nodeWidth = nodeWidth;
        cl_int err = 0;
        this->_clNodeWeights = clCreateBuffer(context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,(size_t)nodeWeights.size()*sizeof(double),(void*)nodeWeights.data(),&err);
        if (err!=CL_SUCCESS) {
            Util::OpenCLExceptionType except;
            except.msg = except.msg + "unable to clCreateBuffer clNodeWeights; ";
            throw except;
        }
    }
    ~KohonenSOMData() {
        if(_clNodeWeights!=0) {
            clReleaseMemObject(_clNodeWeights);
        }
    }
    const std::vector<int>& mapDimensions() {
        return _mapDimensions;
    }
    const int nodeWdith() {
        return _nodeWidth;
    }
    const cl_mem getClNodeWeights() {
        return _clNodeWeights;
    }
};

class KohonenSOMSampleData {
private:
    cl_mem _clData = 0;
    uint _width = 0;
    uint _count = 0;
public:
    KohonenSOMSampleData(const boost::compute::context &context, 
            const std::vector<double> &sampleData, 
            const uint sampleWidth,
            const uint sampleCount) {
            
        this->_width = sampleWidth;
        this->_count = sampleCount;
        cl_int err = 0;
        this->_clData = clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)sampleData.size()*sizeof(double),(void*)sampleData.data(),&err);
        if (err!=CL_SUCCESS) {
            Util::OpenCLExceptionType except;
            except.msg = "unable to clCreateBuffer clData; ";
            throw except;
        }
    }
    ~KohonenSOMSampleData() {
        if(this->_clData!=0) {
            clReleaseMemObject(this->_clData);
        }
    }
    uint width() { return this->_width; }
    uint count() { return this->_count; }
};

class KohonenSOMResult {
private:
    std::vector<std::map<int,double>> _result;
public:
    KohonenSOMResult(const boost::compute::context &context, const KohonenSOMDataPtr &data) {
    }
    ~KohonenSOMResult() {
    }
    std::vector<std::map<int,double>> result() {
        return std::vector<std::map<int,double>>();
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
    KohonenSOMResultPtr map(const KohonenSOMDataPtr &data, const KohonenSOMSampleDataPtr &sampleData, uint maxNodes);
    /**
     * Updates map node weights using the result passed in
     */
    void updateWeights(const KohonenSOMDataPtr &data, const KohonenSOMSampleDataPtr &sampleData, const KohonenSOMResultPtr &result);
};

}}

#endif /* kohonen_som_h */

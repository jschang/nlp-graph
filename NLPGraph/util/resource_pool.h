//
//  resource_pool.h
//  NLPGraph
//
//  Created by Jonathan Schang on 6/17/15.
//  Copyright (c) 2015 Jonathan Schang. All rights reserved.
//

#ifndef __NLPGraph__resource_pool__
#define __NLPGraph__resource_pool__

#include <vector>
#include <map>
#include <boost/thread.hpp> 
#include <time.h>

namespace NLPGraph {
namespace Util {

template <class T>
class ResourcePool {
private:
    std::map<T,bool> m_available;
    boost::recursive_mutex m_lock;
    Util::LoggerType m_logger;
public:
    ResourcePool() : m_logger(boost::log::keywords::channel="NLPGraph::Dao::ModelPostgres") {}
    virtual ~ResourcePool() {}
public:
    void release(T resource) {
        m_lock.lock();
        m_available[resource]=true;
        m_lock.unlock();
    }
    void addResource(T resource) { 
        m_lock.lock();
        m_available.insert(std::pair<T,bool>(resource,true));
        m_lock.unlock();
    }
    T obtain() { 
        m_lock.lock();
        T *ret = 0;
        while(ret==0) {
            for(typename std::map<T, bool>::iterator iterator = m_available.begin(); 
                    iterator!=m_available.end(); 
                    iterator++) {
                if(iterator->second) {
                    ret = (T*)&iterator->first;
                    break;
                }
            }
            if(ret!=0) {
                m_available[*ret]=false;
            }
            struct timespec entry, parm2;
            entry.tv_sec = 0;
            entry.tv_nsec = 100;
            if(nanosleep(&entry,&parm2)<0) {
                BOOST_LOG_SEV(m_logger,severity_level::error) << __PRETTY_FUNCTION__ << " line:" << __LINE__ << " " << "Unable to sleep for 100 nanoseconds";
            }
        }
        m_lock.unlock();
        return *ret;
    }
    std::vector<T>* obtainAll() {
        std::vector<T>* ret = new std::vector<T>();
        while(countAvailable()>0) {
            ret->push_back(obtain());
        }
        return ret;
    }
    void releaseAll() {
        m_lock.lock();
        for(typename std::map<T, bool>::iterator iterator = m_available.begin(); 
            iterator!=m_available.end(); 
            iterator++) {
            m_available[iterator->first] = true;
        }
        m_lock.unlock();
    }
    int countAvailable() {
        m_lock.lock();
        int count = 0;
        for(typename std::map<T, bool>::iterator iterator = m_available.begin(); 
            iterator!=m_available.end(); 
            iterator++) {
            if(iterator->second) {
                count += iterator->first ? 1 : 0;
            }
        }
        m_lock.unlock();
        return count;
    }
};

template <class T>
using ResourcePoolPtr = boost::shared_ptr<ResourcePool<T>>;

}}

#endif /* defined(__NLPGraph__resource_pool__) */

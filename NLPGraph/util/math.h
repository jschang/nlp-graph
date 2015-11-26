//
//  math.h
//  NLPGraph
//
//  Created by Jonathan Schang on 7/19/15.
//
//

#ifndef __NLPGraph__math__
#define __NLPGraph__math__

#include "../math/generator.h"

namespace NLPGraph {
namespace Util {

class Math {
private:
    Math();
public:
    static int isEqual(double a, double b, double epsilon=.0000001); 
    static NLPGraph::Math::GeneratorRealPtr rangeRandGen(float low, float high);
public:
    template <class T> static NLPGraph::Math::GeneratorIntegerPtr<T> maxRandGen() {
        // initialize the overly complicated initialization for random numbers
        struct timeval tv;
        boost::random::mt19937 *mt 
            = new boost::random::mt19937(tv.tv_usec);
        boost::uniform_int<T> *dist
            = new boost::uniform_int<T>(0, std::numeric_limits<T>::max());
        NLPGraph::Math::variate_generator<T> *gen = 
            new NLPGraph::Math::variate_generator<T>(*mt, *dist);
        NLPGraph::Math::GeneratorIntegerPtr<T> ret(
            new NLPGraph::Math::Generator< 
                boost::random::mt19937,
                boost::uniform_int<T>,
                NLPGraph::Math::variate_generator<T>,T >(mt,dist,gen));
        return ret;
    };
    template <class T> static NLPGraph::Math::GeneratorIntegerPtr<T> rangeRandGen(T low, T high) {
        // initialize the overly complicated initialization for random numbers
        struct timeval tv;
        boost::random::mt19937 *mt 
            = new boost::random::mt19937(tv.tv_usec);
        boost::uniform_int<T> *dist
            = new boost::uniform_int<T>(low, high);
        NLPGraph::Math::variate_generator<T> *gen 
            = new NLPGraph::Math::variate_generator<T>(*mt, *dist);
        NLPGraph::Math::GeneratorIntegerPtr<T> ret(
            new NLPGraph::Math::Generator< 
                boost::random::mt19937,
                boost::uniform_int<T>,
                NLPGraph::Math::variate_generator<T>,T >(mt,dist,gen));
        return ret;
    };
};

}}

#endif /* defined(__NLPGraph__math__) */

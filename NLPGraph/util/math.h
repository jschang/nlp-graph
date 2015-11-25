//
//  math.h
//  NLPGraph
//
//  Created by Jonathan Schang on 7/19/15.
//
//

#ifndef __NLPGraph__math__
#define __NLPGraph__math__

#include <time.h>
#include <boost/shared_ptr.hpp>
#include <boost/random.hpp>

namespace NLPGraph {
namespace Util {

template <class T>
using variate_generator = boost::variate_generator<boost::random::mt19937&, boost::uniform_int<T>>;
using variate_generator_real = boost::variate_generator<boost::random::mt19937&, boost::uniform_real<float>>;

template<class MT, class DIST, class GEN, class T>
class Generator {
    private:
        boost::shared_ptr<MT> mt;
        boost::shared_ptr<DIST> dist;
        boost::shared_ptr<GEN> gen;
        
    public:
        Generator() {
        }
        Generator(Generator &source) {
            dist = source.dist;
            gen = source.gen;
            mt = source.mt;
        }
        Generator(MT *mt, DIST *dist, GEN *gen) {
            this->mt = boost::shared_ptr<MT>(mt);
            this->gen = boost::shared_ptr<GEN>(gen);
            this->dist = boost::shared_ptr<DIST>(dist);
        }
        ~Generator() {
        }
        T operator()() {
            return (T)(*gen)();
        }
};
template <class T>
using GeneratorIntegerPtr = boost::shared_ptr< Generator< boost::random::mt19937,boost::uniform_int<T>,variate_generator<T>,T > >;
using GeneratorRealPtr = boost::shared_ptr< Generator< boost::random::mt19937,boost::uniform_real<float>,variate_generator_real,float > >;

class Math {
private:
    Math() {}
public:
    
    inline static int isEqual(double a, double b, double epsilon=.0000001) {
        return std::abs(a-b) < epsilon;
    }
    template <class T>
    static GeneratorIntegerPtr<T> maxRandGen() {
        // initialize the overly complicated initialization for random numbers
        struct timeval tv;
        boost::random::mt19937 *mt 
            = new boost::random::mt19937(tv.tv_usec);
        boost::uniform_int<T> *dist
            = new boost::uniform_int<T>(0, std::numeric_limits<T>::max());
        variate_generator<T> *gen = 
            new variate_generator<T>(*mt, *dist);
        GeneratorIntegerPtr<T> ret(
            new Generator< boost::random::mt19937,boost::uniform_int<T>,variate_generator<T>,T >(mt,dist,gen));
        return ret;
    }
    
    template <class T>
    static GeneratorIntegerPtr<T> rangeRandGen(T low, T high) {
        // initialize the overly complicated initialization for random numbers
        struct timeval tv;
        boost::random::mt19937 *mt 
            = new boost::random::mt19937(tv.tv_usec);
        boost::uniform_int<T> *dist
            = new boost::uniform_int<T>(low, high);
        variate_generator<T> *gen 
            = new variate_generator<T>(*mt, *dist);
        GeneratorIntegerPtr<T> ret(
            new Generator< boost::random::mt19937,boost::uniform_int<T>,variate_generator<T>,T >(mt,dist,gen));
        return ret;
    }
    
    static GeneratorRealPtr rangeRandGen(float low, float high) {
        // initialize the overly complicated initialization for random numbers
        struct timeval tv;
        boost::random::mt19937 *mt 
            = new boost::random::mt19937(tv.tv_usec);
        boost::uniform_real<float> *dist
            = new boost::uniform_real<float>(low, high);
        variate_generator_real* gen
            = new variate_generator_real(*mt, *dist);
        GeneratorRealPtr ret(
            new Generator< boost::random::mt19937,boost::uniform_real<float>,variate_generator_real,float >(mt,dist,gen));
        return ret;
    }
};

}}

#endif /* defined(__NLPGraph__math__) */

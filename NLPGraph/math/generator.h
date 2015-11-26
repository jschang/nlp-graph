//
//  math.h
//  NLPGraph
//
//  Created by Jonathan Schang on 7/19/15.
//
//

#ifndef __NLPGraph__math_generator__
#define __NLPGraph__math_generator__

#include <time.h>
#include <boost/shared_ptr.hpp>
#include <boost/random.hpp>

namespace NLPGraph {
namespace Math {

template<class MT, class DIST, class GEN, class T>
class Generator {
private:
    boost::shared_ptr<MT> mt;
    boost::shared_ptr<DIST> dist;
    boost::shared_ptr<GEN> gen;
public:
    Generator() {
    }
    ~Generator() {
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
    T operator()() {
        return (T)(*gen)();
    }
};

template <class T> using variate_generator = boost::variate_generator<boost::random::mt19937&, boost::uniform_int<T>>;
using variate_generator_real = boost::variate_generator<boost::random::mt19937&, boost::uniform_real<float>>;

template <class T> using GeneratorIntegerPtr = boost::shared_ptr< Generator< boost::random::mt19937,boost::uniform_int<T>,variate_generator<T>,T > >;
using GeneratorRealPtr = boost::shared_ptr< Generator< boost::random::mt19937,boost::uniform_real<float>,variate_generator_real,float > >;

}}

#endif
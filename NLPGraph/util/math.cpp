#include "math.h"

namespace NLPGraph {
namespace Util {

int Math::isEqual(double a, double b, double epsilon) {
    return std::abs(a-b) < epsilon;
}

NLPGraph::Math::GeneratorRealPtr Math::rangeRandGen(float low, float high) {
    // initialize the overly complicated initialization for random numbers
    struct timeval tv;
    boost::random::mt19937 *mt 
        = new boost::random::mt19937(tv.tv_usec);
    boost::uniform_real<float> *dist
        = new boost::uniform_real<float>(low, high);
    NLPGraph::Math::variate_generator_real* gen
        = new NLPGraph::Math::variate_generator_real(*mt, *dist);
    NLPGraph::Math::GeneratorRealPtr ret(
        new NLPGraph::Math::Generator< 
            boost::random::mt19937,
            boost::uniform_real<float>,
            NLPGraph::Math::variate_generator_real,float >(mt,dist,gen));
    return ret;
}

}}
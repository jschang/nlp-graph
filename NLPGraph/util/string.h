//
//  string.h
//  NLPGraph
//
//  Created by Jonathan Schang on 7/19/15.
//
//

#ifndef __NLPGraph__string__
#define __NLPGraph__string__

#include <sstream>

namespace NLPGraph {
namespace Util {

class String {

private:
    String() {}
    
public:
    template <typename T>
    static std::string str(const T inArray[], const int len) {
        std::stringstream retStr;
        for(int i=0; i<len; i++) {
            retStr << (T)inArray[i] << ",";
        }
        return retStr.str();
    }
    
    template <size_t _Size=64, typename T>
    static std::string strb(const T inArray[], const int len) {
        std::stringstream retStr;
        for(int i=0; i<len; i++) {
            std::bitset<_Size> y((T)inArray[i]);
            retStr << y << ",";
        }
        return retStr.str();
    }
};

}}

#endif /* defined(__NLPGraph__string__) */

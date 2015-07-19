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
    template <class T>
    static std::string str(const T &inArray, const int len) {
        std::stringstream retStr;
        for(int i=0; i<len; i++) {
            retStr << inArray[i] << ",";
        }
        return retStr.str();
    }
};

}}

#endif /* defined(__NLPGraph__string__) */

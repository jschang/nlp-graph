//
//  file.h
//  NLPGraph
//
//  Created by Jonathan Schang on 6/26/15.
//  Copyright (c) 2015 Jonathan Schang. All rights reserved.
//

#ifndef __NLPGraph__file__
#define __NLPGraph__file__

#include <iostream>

namespace NLPGraph {
namespace Util {

class File {
private:
    File() {}
public:
    static std::string readTextFile(std::string fileName, std::string encoding);
    static char* readFile(std::string fileName);
};

}}

#endif /* defined(__NLPGraph__file__) */

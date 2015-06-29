//
//  time.h
//  NLPGraph
//
//  Created by Jonathan Schang on 6/28/15.
//  Copyright (c) 2015 Jonathan Schang. All rights reserved.
//

#ifndef __NLPGraph__time__
#define __NLPGraph__time__

#include <time.h>

namespace NLPGraph {
namespace Util {

class TimeHelper {
public:
    static void fillTimeStruct(timespec &ts);
    static timespec getTimeStruct();
};

}}

#endif /* defined(__NLPGraph__time__) */

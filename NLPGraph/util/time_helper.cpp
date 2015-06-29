//
//  time.cpp
//  NLPGraph
//
//  Created by Jonathan Schang on 6/28/15.
//  Copyright (c) 2015 Jonathan Schang. All rights reserved.
//

#include "time_helper.h"

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

namespace NLPGraph {
namespace Util {

void TimeHelper::fillTimeStruct(timespec &ts) {
    #ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    ts.tv_sec = mts.tv_sec;
    ts.tv_nsec = mts.tv_nsec;
    #else
    clock_gettime(CLOCK_REALTIME, &ts);
    #endif
}

timespec TimeHelper::getTimeStruct() {
    timespec ts;
    #ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    ts.tv_sec = mts.tv_sec;
    ts.tv_nsec = mts.tv_nsec;
    #else
    clock_gettime(CLOCK_REALTIME, &ts);
    #endif
    return ts;
}

}}

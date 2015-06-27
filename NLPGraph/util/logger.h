//
//  logger.h
//  NLPGraph
//
//  Created by Jonathan Schang on 6/21/15.
//  Copyright (c) 2015 Jonathan Schang. All rights reserved.
//

#include <boost/log/common.hpp>
#include <boost/log/sources/global_logger_storage.hpp>
#include <boost/log/sources/severity_channel_logger.hpp>

#ifndef NLPGraph_logger_h
#define NLPGraph_logger_h

namespace NLPGraph {
namespace Util {

enum severity_level
{
    normal,
    notification,
    warning,
    error,
    critical
};

typedef boost::log::sources::severity_channel_logger_mt<
    severity_level,     // the type of the severity level
    std::string         // the type of the channel name
> LoggerType;

BOOST_LOG_INLINE_GLOBAL_LOGGER_DEFAULT(Logger, LoggerType)

}}

#endif

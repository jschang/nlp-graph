//
//  input_channel.h
//  NLPGraph
//
//  Created by Jonathan Schang on 6/18/15.
//  Copyright (c) 2015 Jonathan Schang. All rights reserved.
//

#ifndef __NLPGraph__input_channel__
#define __NLPGraph__input_channel__

#include "../nlpgraph.h"

namespace NLPGraph {
namespace Dto {

class InputChannel {
private:
    uint64_t m_id;
public:
    InputChannel(uint64_t id) :
            m_id(0) {
        m_id = id;
    }
    virtual uint64_t getId() { return m_id; }
};

}}

#endif /* defined(__NLPGraph__input_channel__) */

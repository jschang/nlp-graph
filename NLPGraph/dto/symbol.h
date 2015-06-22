//
//  symbol.h
//  NLPGraph
//
//  Created by Jonathan Schang on 6/16/15.
//  Copyright (c) 2015 Jonathan Schang. All rights reserved.
//

#ifndef __NLPGraph__symbol__
#define __NLPGraph__symbol__

#include "../nlpgraph.h"

namespace NLPGraph { 
namespace Dto {

class Symbol {
private:
    uint64_t m_id;
    std::vector<SymbolPtr> m_members;
    InputChannelPtr m_inputChannel;
public:
    Symbol(uint64_t id, InputChannelPtr inputChannel) 
            : m_inputChannel(nullptr) {
        m_inputChannel = inputChannel;
        m_id = id;
    }
    virtual ~Symbol() {
    }
    virtual uint64_t getId() {
        return m_id;
    }
    virtual unsigned long memberCount() {
        return m_members.size();
    }
    virtual SymbolPtr getMember(long i) {
        return m_members[i];
    }
    virtual InputChannelPtr getInputChannel() {
        return m_inputChannel;
    }
    virtual void addMember(SymbolPtr member) {
        m_members.push_back(member);
    }
};

}}

#endif /* defined(__NLPGraph__symbol__) */

// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <linux/can/netlink.h>
#include <linux/rtnetlink.h>
#include <net/if.h>
#include <sys/socket.h>
#include <unistd.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace opensprite_can {
constexpr int execution_hz = 500;
constexpr int slots_per_cycle = 4;
struct BusBudget { double utilization; double worst_slot_seconds; };
inline BusBudget CheckBudget(unsigned motors, unsigned arbitration, unsigned data) {
    if (!motors || motors>8 || !arbitration || !data || data<arbitration)
        throw std::runtime_error("invalid CAN FD population/bitrate");
    // Conservative standard-ID, 8-byte FD+BRS request AND response allowance.
    // Includes stuffing/control/interframe allowance, not USB/drive latency.
    const double frame_seconds=64./arbitration+128./data;
    BusBudget result{motors*execution_hz*2*frame_seconds,
                     ((motors+3)/4)*2*frame_seconds};
    if (result.utilization>.8 || result.worst_slot_seconds>.0004)
        throw std::runtime_error("500Hz CAN FD wire budget exceeds 80 percent");
    return result;
}
inline bool Due(std::size_t slot, int phase) {
    return phase>=0 && phase<slots_per_cycle && int(slot%slots_per_cycle)==phase;
}
struct Link { unsigned arbitration=0,data=0,flags=0; bool up=false,can=false; };
inline void ParseCanAttributes(const rtattr* attribute,int length,Link& link) {
    for (; RTA_OK(attribute,length); attribute=RTA_NEXT(attribute,length)) {
        const auto type=attribute->rta_type&NLA_TYPE_MASK;
        if(type==IFLA_CAN_BITTIMING && RTA_PAYLOAD(attribute)>=sizeof(can_bittiming)) {
            can_bittiming timing{};std::memcpy(&timing,RTA_DATA(attribute),sizeof(timing));
            link.arbitration=timing.bitrate;
        } else if(type==IFLA_CAN_DATA_BITTIMING && RTA_PAYLOAD(attribute)>=sizeof(can_bittiming)) {
            can_bittiming timing{};std::memcpy(&timing,RTA_DATA(attribute),sizeof(timing));
            link.data=timing.bitrate;
        } else if(type==IFLA_CAN_CTRLMODE && RTA_PAYLOAD(attribute)>=sizeof(can_ctrlmode)) {
            can_ctrlmode mode{};std::memcpy(&mode,RTA_DATA(attribute),sizeof(mode));link.flags=mode.flags;
        }
    }
}
inline Link QueryLink(const std::string& name) {
    const auto index=if_nametoindex(name.c_str());
    if (!index) throw std::runtime_error("CAN interface not found: "+name);
    const int fd=socket(AF_NETLINK,SOCK_RAW|SOCK_CLOEXEC,NETLINK_ROUTE);
    if(fd<0)throw std::runtime_error("cannot query CAN link configuration");
    struct Guard { int fd; ~Guard(){close(fd);} } guard{fd};
    timeval timeout{1,0};setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
    struct Request { nlmsghdr header; ifinfomsg info; } request{};
    request.header.nlmsg_len=NLMSG_LENGTH(sizeof(ifinfomsg));
    request.header.nlmsg_type=RTM_GETLINK;request.header.nlmsg_flags=NLM_F_REQUEST;
    request.header.nlmsg_seq=1;request.info.ifi_family=AF_UNSPEC;request.info.ifi_index=int(index);
    sockaddr_nl kernel{};kernel.nl_family=AF_NETLINK;
    if(sendto(fd,&request,request.header.nlmsg_len,0,reinterpret_cast<sockaddr*>(&kernel),sizeof(kernel))<0)
        throw std::runtime_error("CAN rtnetlink request failed");
    alignas(nlmsghdr) std::array<char,16384> buffer{};
    const auto received=recv(fd,buffer.data(),buffer.size(),0);
    if(received<=0)throw std::runtime_error("CAN rtnetlink reply failed");
    int remaining=int(received);Link link;
    for(auto* message=reinterpret_cast<nlmsghdr*>(buffer.data());NLMSG_OK(message,remaining);
        message=NLMSG_NEXT(message,remaining)) {
        if(message->nlmsg_type==NLMSG_ERROR)throw std::runtime_error("CAN link query rejected");
        if(message->nlmsg_type!=RTM_NEWLINK || message->nlmsg_len<NLMSG_LENGTH(sizeof(ifinfomsg)))continue;
        auto* info=static_cast<ifinfomsg*>(NLMSG_DATA(message));
        if(info->ifi_index!=int(index))continue;
        link.up=bool(info->ifi_flags&IFF_UP);
        int bytes=IFLA_PAYLOAD(message);
        for(auto* a=IFLA_RTA(info);RTA_OK(a,bytes);a=RTA_NEXT(a,bytes)) {
            if((a->rta_type&NLA_TYPE_MASK)!=IFLA_LINKINFO)continue;
            int nested=RTA_PAYLOAD(a);
            for(auto* b=static_cast<rtattr*>(RTA_DATA(a));RTA_OK(b,nested);b=RTA_NEXT(b,nested)) {
                if((b->rta_type&NLA_TYPE_MASK)==IFLA_INFO_KIND && RTA_PAYLOAD(b)>=4)
                    link.can=std::memcmp(RTA_DATA(b),"can\0",4)==0;
                if((b->rta_type&NLA_TYPE_MASK)==IFLA_INFO_DATA)
                    ParseCanAttributes(static_cast<rtattr*>(RTA_DATA(b)),RTA_PAYLOAD(b),link);
            }
        }
    }
    return link;
}
inline BusBudget CheckLinkConfiguration(const Link& link,unsigned motors) {
    if(!link.up || !link.can || !(link.flags&CAN_CTRLMODE_FD) ||
       (link.flags&(CAN_CTRLMODE_LISTENONLY|CAN_CTRLMODE_LOOPBACK)))
        throw std::runtime_error("500Hz requires an UP physical CAN FD link in normal transmit mode");
    return CheckBudget(motors,link.arbitration,link.data);
}
inline BusBudget CheckLink(const std::string& name,unsigned motors) {
    return CheckLinkConfiguration(QueryLink(name),motors);
}
} // namespace opensprite_can

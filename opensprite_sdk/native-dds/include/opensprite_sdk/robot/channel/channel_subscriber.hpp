// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <opensprite_sdk/robot/channel/channel_factory.hpp>
namespace opensprite_sdk { namespace robot {
template<class T> using ChannelSubscriber = ::opensprite_sdk::ChannelSubscriber<T>;
template<class T> using ChannelSubscriberPtr = ::opensprite_sdk::ChannelSubscriberPtr<T>;
}}

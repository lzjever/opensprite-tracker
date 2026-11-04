// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <opensprite_sdk/robot/channel/channel_factory.hpp>
namespace opensprite_sdk { namespace robot {
template<class T> using ChannelPublisher = ::opensprite_sdk::ChannelPublisher<T>;
template<class T> using ChannelPublisherPtr = ::opensprite_sdk::ChannelPublisherPtr<T>;
}}

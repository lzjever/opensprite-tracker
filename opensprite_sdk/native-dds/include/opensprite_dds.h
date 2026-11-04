/* OpenSprite typed Fast DDS C ABI. SPDX-License-Identifier: AGPL-3.0-only */
#ifndef OPENSPRITE_DDS_H
#define OPENSPRITE_DDS_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Error text belongs to the calling thread and lasts until its next error. */
const char* osdds_error(void);
/* interface: existing IPv4 interface/address; NULL selects loopback. */
void* osdds_open(unsigned domain, const char* interface, const char* topic,
                 const char* message_type, const char* role);
/* Explicit bounded receive history. Legacy osdds_open keeps depth1. Publishers
 * require depth1; subscribers accept1..64. This does not change writer QoS. */
void* osdds_open_with_history(unsigned domain, const char* interface, const char* topic,
                 const char* message_type, const char* role, unsigned reader_history);
/* Local ABI uses SDK JSON envelopes; DDS itself carries generated typed CDR. */
int osdds_write(void* handle, const char* payload, size_t size);
/* Returns JSON byte count, 0 for no sample, -1 for error. timeout_ms <= 10000. */
int osdds_take(void* handle, char* output, size_t capacity, unsigned timeout_ms);
int osdds_matches(void* handle);
/* DDS operations may run concurrently. The caller must synchronize lifetime:
 * wait for all operations before close and never reuse a closed handle. */
void osdds_close(void* handle);
#ifdef __cplusplus
}
#endif
#endif

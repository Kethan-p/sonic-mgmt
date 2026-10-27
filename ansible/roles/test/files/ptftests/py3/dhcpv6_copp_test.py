"""PTF sender for the GPINS DHCPv6 CoPP test (tests/dhcp_relay/test_dhcpv6_copp_gpins.py).

Floods DHCPv6 Solicit multicast on one PTF port for ``duration`` seconds as
fast as the PTF host allows (optionally capped by ``send_rate_limit``) and
reports the achieved TX rate. The *receive* side (how many packets reached the
DUT CPU) is measured by the pytest side on the DUT, not here: upstream
copp_tests.py measured it with a ptf_nn_agent installed inside the syncd
Docker container, which GPINS does not have.

The test fails itself if the achieved TX rate is below ``min_tx_pps`` so a
slow sender cannot turn into a false "policer works" pass.

Parameters (ptf --test-params):
    port_index      PTF port to send on (DUT client port)
    duration        seconds to send (default 30)
    send_rate_limit upper bound on pps via a per-packet sleep (default 0 = unlimited)
    min_tx_pps      fail if the achieved rate is below this (default 0 = no check)
    client_mac      source MAC (default: MAC of the PTF port)
"""
import logging
import time

import ptf
import ptf.packet as scapy
import ptf.testutils as testutils
from ptf.base_tests import BaseTest
from scapy.layers.dhcp6 import DHCP6_Solicit, DHCP6OptClientId, DHCP6OptElapsedTime, DHCP6OptIA_NA, DUID_LLT

logger = logging.getLogger(__name__)

DHCPV6_MAC_MULTICAST = "33:33:00:01:00:02"
DHCPV6_IP_MULTICAST = "ff02::1:2"
DHCPV6_UDP_CLIENT_PORT = 546
DHCPV6_UDP_SERVER_PORT = 547


def eui64_link_local(mac):
    octets = [int(x, 16) for x in mac.split(":")]
    octets[0] ^= 0x02
    eui = octets[:3] + [0xff, 0xfe] + octets[3:]
    return "fe80::{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}".format(*eui)


class DHCPv6FloodTest(BaseTest):

    def setUp(self):
        self.dataplane = ptf.dataplane_instance
        self.params = testutils.test_params_get()
        self.port_index = int(self.params["port_index"])
        self.duration = float(self.params.get("duration", 30))
        self.send_rate_limit = float(self.params.get("send_rate_limit", 0) or 0)
        self.min_tx_pps = float(self.params.get("min_tx_pps", 0) or 0)
        self.client_mac = self.params.get("client_mac") or self.dataplane.get_mac(0, self.port_index)
        if isinstance(self.client_mac, bytes):
            self.client_mac = self.client_mac.decode()
        self.dataplane.flush()

    def build_solicit(self):
        pkt = (scapy.Ether(src=self.client_mac, dst=DHCPV6_MAC_MULTICAST)
               / scapy.IPv6(src=eui64_link_local(self.client_mac), dst=DHCPV6_IP_MULTICAST, hlim=1)
               / scapy.UDP(sport=DHCPV6_UDP_CLIENT_PORT, dport=DHCPV6_UDP_SERVER_PORT)
               / DHCP6_Solicit(trid=0xC0FFEE)
               / DHCP6OptClientId(duid=DUID_LLT(lladdr=self.client_mac, timeval=0))
               / DHCP6OptElapsedTime(elapsedtime=0)
               / DHCP6OptIA_NA(iaid=1, T1=0, T2=0))
        return bytes(pkt)

    def runTest(self):
        pkt = self.build_solicit()
        interval = 1.0 / self.send_rate_limit if self.send_rate_limit > 0 else 0.0
        sent = 0
        start = time.time()
        end = start + self.duration
        while time.time() < end:
            testutils.send_packet(self, self.port_index, pkt)
            sent += 1
            if interval:
                time.sleep(interval)
        elapsed = time.time() - start
        tx_pps = sent / elapsed if elapsed else 0.0
        msg = "DHCPv6FloodTest: sent %d Solicits on port %d in %.1fs = %.0f pps" % (
            sent, self.port_index, elapsed, tx_pps)
        logger.info(msg)
        print(msg)
        assert tx_pps >= self.min_tx_pps, (
            "PTF sender too slow for a conclusive CoPP test: %.0f pps < required %.0f pps" % (
                tx_pps, self.min_tx_pps))

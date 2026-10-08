"""DHCPv6 packet-reception tests with an ACL table bound to the client port (GPINS).

Upstream (sonic-mgmt) checks that an ingress ``L3V6`` ACL table on the
client-facing port does not stop DHCPv6 Solicit multicast (ff02::1:2 / UDP 547)
from reaching the DUT CPU, both with an empty table and with an explicit
"accept DHCPv6 multicast + drop everything else" rule set.

GPINS differences from upstream:
  * no ``config``/``acl-loader``/``sonic-cfggen`` CLI -> ACL_TABLE / ACL_RULE are
    written straight to CONFIG_DB (``gpins_utils.set_acl_table`` / ``set_acl_rules``)
    and the OpenConfig JSON that ``acl-loader`` consumed upstream is translated
    by ``gpins_utils.acl_rules_from_openconfig``;
  * orchagent reports STATE_DB ``ACL_TABLE_TABLE|<name> status=Active`` which is
    used as the convergence signal (observed on the reference switch);
  * no minigraph / dual-ToR: the client port and its PTF index come from the
    ``relay_mode`` discovery shared with test_dhcpv6_relay.py, so the test runs in
    svi / l3_gua / l3_lla like the rest of the suite;
  * no ``DATAACL`` table exists on GPINS, so the upstream "free TCAM" dance is gone.

TODO(GPINS, Virgo testbed): confirm the built-in ``L3V6`` table type is accepted
by the Bloom ACL pipeline (STATE_DB status must become Active); if not, the
platform's own IPv6-capable table type has to be used instead.
"""
import logging
import json
import os

import pytest

from scapy.layers.dhcp6 import DHCP6_Solicit
from tests.common.helpers.assertions import pytest_assert, pytest_require
from tests.common.utilities import capture_and_check_packet_on_dut
from tests.dhcp_relay import gpins_utils
from tests.dhcp_relay.test_dhcpv6_relay import (  # noqa F401
    skip_if_dhcp_relay_unavailable,
    relay_common_facts,
)

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.topology("t0", "m0", 'mx', "m1")
]

ACL_TABLE_NAME_DHCPV6_PKT_RECV_TEST = "DHCPV6_PKT_RECV_TEST"
ACL_STAGE_INGRESS = "ingress"
ACL_TABLE_TYPE_L3V6 = "L3V6"

ACL_RULE_FILE_PATH_MULTICAST_ACCEPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "acl", "dhcpv6_pkt_recv_multicast_accept.json")

DHCPV6_MAC_MULTICAST = "33:33:00:01:00:02"
DHCPV6_IP_MULTICAST = "ff02::1:2"
DHCPV6_UDP_CLIENT_PORT = 546
DHCPV6_UDP_SERVER_PORT = 547

ACL_CONVERGE_TIMEOUT = 60


def eui64_link_local(mac):
    """Link-local address a client with ``mac`` would auto-configure (RFC 4291 EUI-64)."""
    octets = [int(x, 16) for x in mac.split(":")]
    octets[0] ^= 0x02
    eui = octets[:3] + [0xff, 0xfe] + octets[3:]
    return "fe80::{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}".format(*eui)


def _load_multicast_accept_rules():
    """CONFIG_DB ACL_RULE fields for the upstream acl-loader JSON (accept DHCPv6 multicast, drop rest)."""
    with open(ACL_RULE_FILE_PATH_MULTICAST_ACCEPT) as fh:
        return gpins_utils.acl_rules_from_openconfig(
            json.load(fh), ACL_TABLE_NAME_DHCPV6_PKT_RECV_TEST, ACL_TABLE_TYPE_L3V6)


@pytest.fixture(scope="module")
def pkt_recv_target(duthosts, rand_one_dut_hostname, tbinfo, relay_mode, dhcp_relay_loopback_ipv6):
    """The port the PTF client sits on for ``relay_mode`` plus everything the test needs about it.

    Returns {relay_interface, client_port (CONFIG_DB name), kernel_iface, ptf_port_id}.
    """
    duthost = duthosts[rand_one_dut_hostname]
    common = relay_common_facts(duthost, tbinfo)
    targets = gpins_utils.relay_targets(duthost, relay_mode, common['ports'], tbinfo)
    pytest_require(targets, "No relay targets for mode {}".format(relay_mode))
    target = targets[0]
    client_port = target["client_port"]
    info = {
        "relay_interface": target["relay_interface"],
        "client_port": client_port,
        "kernel_iface": gpins_utils.to_kernel_ifname(client_port),
        "ptf_port_id": gpins_utils.ptf_index_for_port(tbinfo, duthost, common['ports'][client_port], client_port),
    }
    logger.info("pkt_recv target (%s): %s", relay_mode, info)
    return info


class Dhcpv6PktRecvBase:
    """Shared ACL table lifecycle + the actual packet-reception check."""

    # Subclasses set this to the ACL_RULE dict to install (None = empty table).
    acl_rules = None

    @pytest.fixture(scope="class", autouse=True)
    def setup_teardown_acl(self, duthosts, rand_one_dut_hostname, pkt_recv_target):
        duthost = duthosts[rand_one_dut_hostname]
        table = ACL_TABLE_NAME_DHCPV6_PKT_RECV_TEST
        snapshot = gpins_utils.snapshot_config_db(
            duthost, ["ACL_TABLE|{}".format(table), "ACL_RULE|{}|*".format(table)])
        if snapshot:
            logger.warning("Stale %s found in CONFIG_DB, replacing it for the test: %s", table, snapshot)
            gpins_utils.del_acl_table(duthost, table)

        gpins_utils.set_acl_table(duthost, table, ACL_TABLE_TYPE_L3V6, stage=ACL_STAGE_INGRESS,
                                  ports=[pkt_recv_target["client_port"]],
                                  description="dhcp_relay pkt_recv test")
        rules = self.acl_rules or {}
        if rules:
            gpins_utils.set_acl_rules(duthost, table, rules)

        try:
            pytest_assert(
                gpins_utils.wait_acl_active(duthost, table, rules=list(rules), timeout=ACL_CONVERGE_TIMEOUT),
                "ACL table {} (type {}, rules {}) not Active in STATE_DB after {}s: table={} rules={}. "
                "If the platform rejects the {} table type this is a platform finding, not a relay bug."
                .format(table, ACL_TABLE_TYPE_L3V6, sorted(rules), ACL_CONVERGE_TIMEOUT,
                        gpins_utils.get_acl_table_state(duthost, table),
                        {r: gpins_utils.get_acl_rule_state(duthost, table, r) for r in rules},
                        ACL_TABLE_TYPE_L3V6))
            yield
        finally:
            gpins_utils.del_acl_table(duthost, table)
            if snapshot:
                gpins_utils.restore_config_db(duthost, snapshot)

    def test_dhcpv6_multicast_recv(self, duthosts, rand_one_dut_hostname, ptfhost, pkt_recv_target):
        """
        Test the DUT can receive DHCPv6 multicast packet
        """
        duthost = duthosts[rand_one_dut_hostname]
        intf = pkt_recv_target["kernel_iface"]
        ptf_port_id = pkt_recv_target["ptf_port_id"]
        logger.info("Start to verify dhcpv6 multicast with interface=%s and ptf_port_id=%s", intf, ptf_port_id)

        def func(pkts):
            matched = [pkt for pkt in pkts if DHCP6_Solicit in pkt and pkt[DHCP6_Solicit].trid == test_trid]
            pytest_assert(len(matched) > 0, "Didn't get packet with expected transaction id")
        src_mac = ptfhost.shell("cat /sys/class/net/eth{}/address".format(ptf_port_id))["stdout"].strip()
        test_trid = 234
        pkts_filter = "ether src %s and udp dst port %s" % (src_mac, DHCPV6_UDP_SERVER_PORT)
        with capture_and_check_packet_on_dut(
            duthost=duthost,
            interface=intf,
            pkts_filter=pkts_filter,
            pkts_validator=func
        ):
            # Upstream sources the Solicit from the DUT's own link-local; a real
            # client uses its own LLA, which is also what the relay will see.
            link_local_ipv6_addr = eui64_link_local(src_mac)
            send_cmd = (
                "python3 -c \"from scapy.all import Ether, IPv6, UDP, sendp; "
                "from scapy.layers.dhcp6 import DHCP6_Solicit; "
                "pkt = Ether(dst='{dst_mac}', src='{src_mac}')/"
                "IPv6(src='{src_ip}', dst='{dst_ip}')/"
                "UDP(sport={sport}, dport={dport})/"
                "DHCP6_Solicit(trid={trid}); "
                "sendp(pkt, iface='eth{port}', verbose=False)\""
            ).format(
                dst_mac=DHCPV6_MAC_MULTICAST,
                src_mac=src_mac,
                src_ip=link_local_ipv6_addr,
                dst_ip=DHCPV6_IP_MULTICAST,
                sport=DHCPV6_UDP_CLIENT_PORT,
                dport=DHCPV6_UDP_SERVER_PORT,
                trid=test_trid,
                port=ptf_port_id,
            )
            ptfhost.shell(send_cmd)


class TestDhcpv6WithEmptyAclTable(Dhcpv6PktRecvBase):
    """
    Test the DUT with empty ACL table
    """
    acl_rules = None


class TestDhcpv6WithMulticastAccpectAcl(Dhcpv6PktRecvBase):
    """
    Test the DUT with multicast accept ACL rule and default drop all rule.
    The drop all rule is added by default for L3V6 table type by acl-loader
    (replicated here by ``gpins_utils.acl_rules_from_openconfig``).
    """
    acl_rules = _load_multicast_accept_rules()

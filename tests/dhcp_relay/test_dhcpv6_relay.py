import ipaddress
import pytest
import time
import logging

from tests.common.fixtures.ptfhost_utils import copy_ptftests_directory   # noqa F401
from tests.common.fixtures.ptfhost_utils import change_mac_addresses      # noqa F401
from tests.common.fixtures.ptfhost_utils import copy_arp_responder_py     # noqa F401
from tests.common.fixtures.ptfhost_utils import setup_ptf_ip_responder, teardown_ptf_ip_responder
from tests.ptf_runner import ptf_runner
from tests.common.utilities import wait_until
from tests.common.helpers.assertions import pytest_assert, pytest_require
from tests.dhcp_relay import gpins_utils

pytestmark = [
    pytest.mark.topology('t0', 'm0', 'mx', 't0-2vlans'),
    pytest.mark.device_type('vs')
]

# GPINS (Virgo-on-Bloom) has no dual-ToR; only single-ToR mode is supported.
SINGLE_TOR_MODE = 'single'
NEW_COUNTER_VALUE_FORMAT = (
    "{'Unknown':'0','Solicit':'0','Advertise':'0','Request':'0','Confirm':'0','Renew':'0','Rebind':'0','Reply':'0',"
    "'Release':'0','Decline':'0','Reconfigure':'0','Information-Request':'0','Relay-Forward':'0','Relay-Reply':'0',"
    "'Malformed':'0'}"
)

logger = logging.getLogger(__name__)


DHCPV6_RELAY_ARP_RESPONDER_CONF = "/tmp/dhcpv6_downstream_relay_arp_responder.json"
# Arbitrary GUA outside the DUT VLAN prefix, used to simulate the downstream relay
# agent (Relay A) that the RFC 8415 relay-to-relay return path forwards to.
DOWNSTREAM_RELAY_IP = "fc02:ffff::1"

# DHCPv6 servers the relay forwards to when the DUT has no DHCP_RELAY entry yet.
# Upstream reads these from minigraph; GPINS has none. The PTF host answers on
# behalf of these addresses (see dhcpv6_relay_test.py).
# TODO(GPINS): make this a testbed/tbinfo property once the Virgo-on-Bloom
# testbed exists (a real DHCP server may replace the PTF responder).
DEFAULT_DHCPV6_SERVERS = ["fc02:2000::1", "fc02:2000::2"]


def _counter_key(duthost, ifname):
    """Return the STATE_DB DHCPv6_COUNTER_TABLE key for ``ifname``.

    GPINS may key the table by the CONFIG_DB name (``Ethernet1/33``) or by the
    kernel name (``Ethernet1_33``); probe both and prefer the one that exists.
    TODO(GPINS, Virgo testbed): pin this down once the relay runs on real hardware.
    """
    candidates = [ifname, gpins_utils.to_kernel_ifname(ifname), gpins_utils.to_config_ifname(ifname)]
    seen = []
    for name in candidates:
        if name in seen:
            continue
        seen.append(name)
        key = "DHCPv6_COUNTER_TABLE|{}".format(name)
        if gpins_utils.redis_exists(duthost, "STATE_DB", key):
            return key
    return "DHCPv6_COUNTER_TABLE|{}".format(ifname)


def check_dhcpv6_relay_counter(duthost, ifname, type, dir):
    # new counter table
    # STATE_DB hgetall 'DHCPv6_COUNTER_TABLE|Vlan1000'
    # {'TX': "{'Unknown':'0','Solicit':'0','Advertise':'0','Request':'0','Confirm':'0','Renew':'0','Rebind':'0',
    #  'Reply':'0', 'Release':'0','Decline':'0','Reconfigure':'0','Information-Request':'0','Relay-Forward':'0',
    #  'Relay-Reply':'0','Malformed':'0'}", 'RX': "{'Unknown':'0','Solicit':'0','Advertise':'0','Request':'0',
    #  'Confirm':'0','Renew':'0','Rebind':'0','Reply':'0', 'Release':'0','Decline':'0','Reconfigure':'0',
    #  'Information-Request':'0','Relay-Forward':'0','Relay-Reply':'0','Malformed':'0'}"}
    #
    # old counter table
    # STATE_DB hgetall 'DHCPv6_COUNTER_TABLE|Vlan1000'
    # {'Unknown':'0','Solicit':'0','Advertise':'0','Request':'0','Confirm':'0','Renew':'0','Rebind':'0','Reply':'0',
    #  'Release':'0','Decline':'0','Reconfigure':'0','Information-Request':'0','Relay-Forward':'0','Relay-Reply':'0',
    #  'Malformed':'0'}
    #
    key = _counter_key(duthost, ifname)
    output_new = gpins_utils.redis_hget(duthost, "STATE_DB", key, dir)
    if output_new:
        counters = eval(output_new)
        assert int(counters[type]) > 0, "{}({}) missing {} count".format(ifname, dir, type)
    else:
        output_old = gpins_utils.redis_hget(duthost, "STATE_DB", key, type)
        assert output_old is not None and int(output_old) > 0, "{} missing {} count".format(ifname, type)


def init_counter(duthost, ifname, types):
    key = _counter_key(duthost, ifname)
    output_new = gpins_utils.redis_hget(duthost, "STATE_DB", key, "RX")
    if output_new:
        gpins_utils.redis_hset(duthost, "STATE_DB", key,
                               {"RX": NEW_COUNTER_VALUE_FORMAT, "TX": NEW_COUNTER_VALUE_FORMAT})
    else:
        gpins_utils.redis_hset(duthost, "STATE_DB", key, {type: "0" for type in types})


@pytest.fixture(scope="module", autouse=True)
def skip_if_dhcp_relay_unavailable(duthosts, rand_one_dut_hostname):
    """Skip the module on GPINS builds that do not ship the DHCP-relay container yet."""
    duthost = duthosts[rand_one_dut_hostname]
    if not gpins_utils.dhcp_relay_available(duthost):
        pytest.skip("DHCP relay feature is not available on this build "
                    "(no {} / FEATURE|dhcp_relay)".format(gpins_utils.DHCP_RELAY_INIT_SCRIPT))


@pytest.fixture(scope="module")
def testing_config(duthosts, rand_one_dut_hostname):
    """GPINS only supports single-ToR; kept for signature compatibility with upstream tests."""
    duthost = duthosts[rand_one_dut_hostname]
    yield SINGLE_TOR_MODE, duthost


@pytest.fixture(scope="module")
def dut_dhcp_relay_data(duthosts, rand_one_dut_hostname, tbinfo, relay_mode, dhcp_relay_loopback_ipv6):
    """Return a list of dicts, one per DHCPv6 relay instance on the DUT, for ``relay_mode``.

    GPINS version. Upstream derives everything from minigraph (VLANs, neighbor
    types, port channels); GPINS has no minigraph, so the data is built from
    CONFIG_DB + the kernel. The test plan requires every test to run in three
    scenarios, selected by the parametrized ``relay_mode`` fixture (conftest):

    * ``svi``    - relay on ``Vlan<N>`` with an IPv6 ``VLAN_INTERFACE``; the PTF
                   client sits on the first physical ``VLAN_MEMBER``
    * ``l3_gua`` - relay on a downlink port that has an IPv6 ``INTERFACE`` entry
    * ``l3_lla`` - relay on a downlink port with link-local only; Interface-ID
                   is ``[<loopback>]:<port>`` so the server can route replies back

    Common: downlink/uplink split by ``PORT.port_direction``; loopback from
    ``LOOPBACK_INTERFACE``; PTF indices from ``PORT.index`` + ``tbinfo.topo.ptf_map``;
    servers from ``DHCP_RELAY|<intf>`` if present, else ``DEFAULT_DHCPV6_SERVERS``.

    Keys are kept compatible with the upstream fixture (``downlink_vlan_iface``
    etc.) so the test bodies work unchanged; GPINS-specific keys are added:
    ``relay_mode``, ``relay_interface`` (CONFIG_DB name), ``relay_kernel_iface``
    (Linux netdev), ``client_port``, ``uplink_ports``, ``interface_id``.
    """
    duthost = duthosts[rand_one_dut_hostname]
    common = relay_common_facts(duthost, tbinfo)

    # Relay targets for this mode (relay_mode fixture already skipped if none).
    # Only oper-up, PTF-reachable downlinks are returned (12 of 62 on ju3u1m1).
    targets = gpins_utils.relay_targets(duthost, relay_mode, common['ports'], tbinfo)
    pytest_require(targets, "No relay targets for mode {}".format(relay_mode))

    # Prefer targets that already carry a DHCP_RELAY entry; otherwise take the
    # first target with the default server list.
    dhcp_relay_table = gpins_utils.get_table(duthost, "CONFIG_DB", "DHCP_RELAY")
    configured = []
    for t in targets:
        servers = gpins_utils.parse_dhcpv6_servers(dhcp_relay_table.get(t["relay_interface"], {}))
        if servers:
            configured.append((t, servers))
    if not configured:
        logger.info("No DHCP_RELAY entry on any %s target; using %s with default servers %s",
                    relay_mode, targets[0]["relay_interface"], DEFAULT_DHCPV6_SERVERS)
        configured = [(targets[0], list(DEFAULT_DHCPV6_SERVERS))]

    dhcp_relay_data_list = [build_relay_data(duthost, tbinfo, relay_mode, target, servers, common)
                            for target, servers in configured]

    logger.info("GPINS dhcp relay data (%s): %s", relay_mode, dhcp_relay_data_list)
    return dhcp_relay_data_list


def relay_common_facts(duthost, tbinfo):
    """Facts shared by every relay instance: ports, uplinks (+PTF indices, MAC), loopback."""
    ports = gpins_utils.get_ports(duthost)
    pytest_require(ports, "CONFIG_DB PORT table is empty")
    _, uplinks = gpins_utils.classify_ports(ports, duthost, tbinfo)
    pytest_require(uplinks, "No uplink ports (no PORT has port_direction={}, no INTERFACE prefix contains a "
                   "BGP_NEIGHBOR address, and tbinfo lists no VM-facing port)".format(
                       gpins_utils.UPLINK_PORT_DIRECTION))

    loopback_name = gpins_utils.DEFAULT_LOOPBACK
    loopback_ipv6, loopback_plen = gpins_utils.get_loopback_ipv6(duthost, loopback_name)
    if not loopback_ipv6:
        # Upstream-style lab t0 configs (config_dut.py) give the uplinks /126 GUAs
        # but leave Loopback0 without an IPv6 address. The relay-reachable address
        # the PTF "server" replies to is then the first uplink's GUA - the same
        # address the kernel selects as source toward a server behind that uplink.
        for uplink in uplinks:
            loopback_ipv6, loopback_plen = gpins_utils.get_interface_ipv6(duthost, uplink)
            if loopback_ipv6:
                logger.warning("No IPv6 on LOOPBACK_INTERFACE|%s; using uplink %s GUA %s/%s as the relay's "
                               "server-facing address", loopback_name, uplink, loopback_ipv6, loopback_plen)
                break
    pytest_require(loopback_ipv6, "No IPv6 address on LOOPBACK_INTERFACE|{} and none on any uplink INTERFACE"
                   .format(loopback_name))

    return {
        'ports': ports,
        'uplinks': list(uplinks),
        'uplink_port_indices': [gpins_utils.ptf_index_for_port(tbinfo, duthost, ports[p]) for p in uplinks],
        'uplink_mac': gpins_utils.get_port_mac(duthost, uplinks[0]),
        'loopback_name': loopback_name,
        'loopback_ipv6': loopback_ipv6,
        'loopback_plen': loopback_plen,
    }


def build_relay_data(duthost, tbinfo, relay_mode, target, dhcpv6_servers, common):
    """Build the upstream-compatible relay data dict for one relay target (see dut_dhcp_relay_data)."""
    ports = common['ports']
    relay_interface = target["relay_interface"]
    client_port = target["client_port"]
    kernel_iface = gpins_utils.to_kernel_ifname(relay_interface)
    lla, lla_with_prefix = gpins_utils.get_port_link_local(duthost, relay_interface)

    downlink_iface = {
        'name': kernel_iface,
        'addr': target["gua"],
        'mask': target["gua_plen"],
        'mac': gpins_utils.get_port_mac(duthost, relay_interface),
        'dhcpv6_server_addrs': list(dhcpv6_servers),
    }
    client_iface = {
        'name': client_port,
        'alias': ports.get(client_port, {}).get('alias', client_port),
        'port_idx': gpins_utils.ptf_index_for_port(tbinfo, duthost, ports.get(client_port, {})),
    }

    return {
        # upstream-compatible keys
        'downlink_vlan_iface': downlink_iface,
        'client_iface': client_iface,
        'uplink_interfaces': [gpins_utils.to_kernel_ifname(p) for p in common['uplinks']],
        'uplink_port_indices': common['uplink_port_indices'],
        'down_interface_link_local': lla,
        'down_interface_link_local_with_prefix_len': lla_with_prefix,
        'loopback_iface': [{'name': common['loopback_name'], 'addr': common['loopback_ipv6'],
                            'prefixlen': common['loopback_plen']}],
        'loopback_ipv6': common['loopback_ipv6'],
        'is_dualtor': False,
        'uplink_mac': common['uplink_mac'],
        'portchannels': {},
        # GPINS-specific keys
        'relay_mode': relay_mode,
        'relay_interface': relay_interface,
        'relay_kernel_iface': kernel_iface,
        'client_port': client_port,
        'uplink_ports': list(common['uplinks']),
        # LLA-only mode uses the interface name for Interface-ID (Appendix step 1);
        # other modes use whatever DHCP_RELAY|<intf> configures (None = default).
        'interface_id': (relay_interface
                         if relay_mode == gpins_utils.RELAY_MODE_L3_LLA else None),
    }


def _ptf_port_mac(ptfhost, port_index):
    """MAC of the PTF interface wired to DUT port ``port_index`` (``eth<idx>``)."""
    res = ptfhost.shell("cat /sys/class/net/eth{}/address".format(port_index), module_ignore_errors=True)
    return res["stdout"].strip() if res["rc"] == 0 else None


def _ptf_port_lla(ptfhost, port_index):
    """Link-local IPv6 address of PTF ``eth<port_index>`` (or EUI-64 fallback from its MAC)."""
    cmd = "ip -6 addr show dev eth{} scope link | awk '/inet6/ {{print $2; exit}}' | cut -d/ -f1".format(port_index)
    res = ptfhost.shell(cmd, module_ignore_errors=True)
    lla = res["stdout"].strip() if res["rc"] == 0 else ""
    if lla:
        return lla
    mac = _ptf_port_mac(ptfhost, port_index)
    if not mac:
        return None
    octets = [int(x, 16) for x in mac.split(":")]
    octets[0] ^= 0x02
    eui = octets[:3] + [0xff, 0xfe] + octets[3:]
    return "fe80::{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}".format(*eui)


@pytest.fixture(scope="module")
def validate_dut_routes_exist(request, duthosts, rand_one_dut_hostname, ptfhost, dut_dhcp_relay_data):
    """Make sure the DUT can forward relayed packets toward every DHCPv6 server.

    Upstream only *checks* that BGP gave the DUT a route. On GPINS the suite must
    not depend on bgp-con (Boyang, 2026-10-08), so this is a two-way door driven
    by ``--dhcp_relay_static_route``:
      * ``never``  - upstream behaviour: assert a route exists.
      * ``auto``   - (default) if the DUT already has an explicit (non-default)
                     route to the server, use it; otherwise install a static
                     /128 route + permanent neighbour for the server on the first
                     uplink pointing at the PTF interface behind that uplink
                     (Appendix step 2), and remove both at module teardown.
      * ``always`` - install the static path even if an explicit route exists.
    """
    duthost = duthosts[rand_one_dut_hostname]
    policy = request.config.getoption("--dhcp_relay_static_route")
    dhcp_servers = set()
    for dhcp_relay in dut_dhcp_relay_data:
        dhcp_servers |= set(dhcp_relay['downlink_vlan_iface']['dhcpv6_server_addrs'])
    uplink = dut_dhcp_relay_data[0]['uplink_ports'][0]
    uplink_ptf_idx = dut_dhcp_relay_data[0]['uplink_port_indices'][0]

    created = []

    def _ensure_routes():
        for server in sorted(dhcp_servers):
            has_route = gpins_utils.ipv6_route_exists(duthost, server)
            has_explicit = gpins_utils.ipv6_explicit_route_exists(duthost, server)
            if policy == "never" or (policy == "auto" and has_explicit):
                pytest_assert(has_route, "Failed to find route to DHCP server '{}' (--dhcp_relay_static_route={})"
                              .format(server, policy))
                logger.info("Using existing route to DHCPv6 server %s", server)
                continue
            ptf_mac = _ptf_port_mac(ptfhost, uplink_ptf_idx)
            pytest_require(ptf_mac, "Cannot read MAC of PTF eth{} (behind uplink {}); needed for the static "
                           "server path".format(uplink_ptf_idx, uplink))
            created.extend(gpins_utils.add_static_server_path(duthost, server, uplink, nexthop_mac=ptf_mac))
            pytest_assert(gpins_utils.ipv6_route_exists(duthost, server),
                          "Static route to DHCP server '{}' via {} was not installed".format(server, uplink))

    _ensure_routes()

    yield _ensure_routes

    if created:
        gpins_utils.del_static_server_path(duthost, created)


@pytest.fixture(scope="module")
def dhcp_relay_config(duthosts, rand_one_dut_hostname, dut_dhcp_relay_data):
    """Make sure every relay instance in ``dut_dhcp_relay_data`` is configured in CONFIG_DB.

    GPINS has no minigraph, so the ``DHCP_RELAY|<interface>`` entries that
    upstream gets for free may be missing. Discovery-first (two-way door):
      * entry already present -> leave it untouched, just wait for it to apply;
      * entry missing -> write it (servers from the fixture; ``l3_lla`` also
        enables Option 18 with the ``[<loopback>]:<port>`` form from the design
        doc) and restore CONFIG_DB on module teardown.
    Returns {relay_interface: applied CONFIG_DB fields}.
    """
    duthost = duthosts[rand_one_dut_hostname]
    snapshot = gpins_utils.snapshot_config_db(duthost, ["DHCP_RELAY|*"])
    applied = {}
    created = []
    for dhcp_relay in dut_dhcp_relay_data:
        intf = dhcp_relay['relay_interface']
        gpins_utils.verify_interface_mode(duthost, intf, dhcp_relay['relay_mode'])
        existing = gpins_utils.get_dhcp_relay_config(duthost, intf)
        if gpins_utils.parse_dhcpv6_servers(existing):
            logger.info("Using existing DHCP_RELAY|%s: %s", intf, existing)
            applied[intf] = existing
            continue
        options = {}
        if dhcp_relay['relay_mode'] == gpins_utils.RELAY_MODE_L3_LLA:
            options = {"enable_interface_id": True, "interface_id": dhcp_relay['interface_id']}
        applied[intf] = gpins_utils.set_dhcp_relay_config(
            duthost, intf, dhcp_relay['downlink_vlan_iface']['dhcpv6_server_addrs'],
            relay_mode=dhcp_relay['relay_mode'], **options)
        created.append(intf)

    if created:
        # STATE_DB DHCP_RELAY|<intf> is the applied configuration (Boyang, 2026-10-08):
        # it appears only when the relay manager accepted the entry. An entry on an
        # LLA-only port without a unique interface_id, or with a duplicate
        # interface_id, is rejected and never shows up here.
        for intf in created:
            pytest_assert(
                wait_until(60, 5, 0, gpins_utils.dhcp_relay_config_converged, duthost, intf),
                "DHCP_RELAY|{} not applied (STATE_DB) within 60s - rejected by the relay manager? "
                "CONFIG_DB={} STATE_DB={}".format(
                    intf, gpins_utils.get_dhcp_relay_config(duthost, intf),
                    gpins_utils.get_dhcp_relay_state(duthost, intf)))
    pytest_assert(gpins_utils.wait_dhcp6relay_bound(duthost),
                  "dhcp6relay is not running/listening (listeners: {})".format(
                      gpins_utils.dhcp6relay_listeners(duthost)))

    yield applied

    if created:
        gpins_utils.restore_config_db(duthost, snapshot)


def gpins_ptf_params(dhcp_relay, relay_cfg):
    """PTF DHCPTest parameters that depend on the relay mode / applied DHCP_RELAY config.

    * link_addr: Relay-Forward link-address. GUA of the relay interface if it
      has one (svi, l3_gua), otherwise '::' (l3_lla).
    * relay_listen_ip: where the simulated server sends Relay-Reply. The relay
      GUA if present, else the loopback (design doc: LLA mode replies are
      routed to the loopback carried in Interface-ID).
    * interface_id / remote_id: the Option 18 / 37 values the DUT is expected
      to insert, derived from the applied CONFIG_DB entry (None = not expected).
    * interface_id_gua: with enable_interface_id=true and no interface_id on a port
      with a GUA, the relay's default Option 18 is that GUA.
    * rfc6939_support: whether Option 79 (Client Link-Layer Address) is expected
      in Relay-Forward (default True unless rfc6939_support == 'false').
    * downstream_reply_src_ip: source of the nested Relay-Reply the DUT forwards
      to a downstream relay; '' (don't care) when the port has no GUA.
    """
    gua = dhcp_relay['downlink_vlan_iface']['addr']
    relay_cfg = relay_cfg or {}

    interface_id = None
    interface_id_gua = None
    if relay_cfg.get("enable_interface_id", "").lower() == "true":
        if relay_cfg.get("interface_id"):
            interface_id = relay_cfg["interface_id"]
        elif gua:
            interface_id_gua = str(gua)
        else:
            interface_id = dhcp_relay['interface_id']
    elif "enable_interface_id" not in relay_cfg and dhcp_relay['relay_mode'] == gpins_utils.RELAY_MODE_L3_LLA:
        # TODO(GPINS, Virgo testbed): confirm the relay's default for Option 18 in LLA mode.
        interface_id = dhcp_relay['interface_id']

    remote_id = None
    if relay_cfg.get("enable_remote_id", "").lower() == "true":
        remote_id = relay_cfg.get("remote_id")

    rfc6939_support = str(relay_cfg.get("rfc6939_support", "true")).lower() != "false"

    return {
        "link_addr": str(gua) if gua else "::",
        "relay_listen_ip": str(gua) if gua else str(dhcp_relay['loopback_ipv6']),
        "interface_id": interface_id,
        "interface_id_gua": interface_id_gua,
        "remote_id": remote_id,
        "rfc6939_support": rfc6939_support,
        "downstream_reply_src_ip": str(gua) if gua else "",
    }


def check_interface_status(duthost):
    """True if dhcp6relay is running and listening (UDP 547 or AF_PACKET; see gpins_utils)."""
    return gpins_utils.is_dhcp6relay_bound(duthost)


def ensure_client_reachability(duthost, vlan_name):
    """
    Ensure the DHCP client's link-local address is reachable from the relay agent.
    This prevents the "neighbor solicitation" issue during DHCPv6 relay.
    """
    logger.info(f"Ensuring client reachability on VLAN {vlan_name}")

    # Send multicast ping to discover all link-local addresses on the segment
    duthost.shell("ping6 -I {} -c 3 ff02::1".format(vlan_name), module_ignore_errors=True)

    # Wait a moment for ND table to be populated
    time.sleep(2)


@pytest.fixture(scope="function")
def unconfigured_relay_target(duthosts, rand_one_dut_hostname, tbinfo, relay_mode, dut_dhcp_relay_data):
    """A relay target of the current mode that has *no* DHCP_RELAY entry.

    GPINS replacement for upstream's "add a VLAN without dhcpv6_servers": there
    is no ``config vlan`` here, so instead we pick an existing downlink that is
    not configured for relay. Skips if every target is already configured.
    Returns a relay data dict (same shape as dut_dhcp_relay_data entries).
    """
    duthost = duthosts[rand_one_dut_hostname]
    configured = {d['relay_interface'] for d in dut_dhcp_relay_data}
    dhcp_relay_table = gpins_utils.get_table(duthost, "CONFIG_DB", "DHCP_RELAY")
    common = relay_common_facts(duthost, tbinfo)
    for target in gpins_utils.relay_targets(duthost, relay_mode, common['ports'], tbinfo):
        intf = target['relay_interface']
        if intf in configured or intf in dhcp_relay_table:
            continue
        gpins_utils.verify_interface_mode(duthost, intf, relay_mode, ports=common['ports'])
        data = build_relay_data(duthost, tbinfo, relay_mode, target,
                                dut_dhcp_relay_data[0]['downlink_vlan_iface']['dhcpv6_server_addrs'], common)
        pytest_assert(not gpins_utils.get_dhcp_relay_config(duthost, intf),
                      "DHCP_RELAY|{} unexpectedly exists".format(intf))
        return data
    pytest.skip("Every {} relay target already has a DHCP_RELAY entry; "
                "need one unconfigured downlink for this test".format(relay_mode))


def setup_downstream_relay_responder(duthost, ptfhost, downlink_vlan, client_ptf_index):
    """Configure PTF's ndp responder + host route for the simulated downstream relay
    agent (Relay A, DOWNSTREAM_RELAY_IP). Its GUA is outside the DUT VLAN prefix, so a
    host route is installed and PTF answers NDP for it, letting DUT Relay B forward the
    RFC 8415 nested Relay-Reply back to it.

    Because setup_ptf_ip_responder() clears the DUT NDP cache, we must re-establish
    client reachability afterwards so the DUT can resolve the client link-local
    address when relaying DHCPv6 Advertise/Reply back to the client.

    Returns the responder pairs for teardown."""
    downstream_relay_responder = [(DOWNSTREAM_RELAY_IP, downlink_vlan, client_ptf_index)]
    setup_ptf_ip_responder(
        duthost, ptfhost, DHCPV6_RELAY_ARP_RESPONDER_CONF, downstream_relay_responder, route=True)
    ensure_client_reachability(duthost, downlink_vlan)
    return downstream_relay_responder


def teardown_downstream_relay_responder(duthost, ptfhost, downstream_relay_responder):
    teardown_ptf_ip_responder(
        duthost, ptfhost, DHCPV6_RELAY_ARP_RESPONDER_CONF, downstream_relay_responder, route=True)


def run_no_relay_test(ptfhost, duthost, target):
    """PTF DHCPNoRelayTest: a client Solicit on ``target``'s port must not be relayed upstream."""
    params = {"hostname": duthost.hostname,
              "client_port_index": target['client_iface']['port_idx'],
              "leaf_port_indices": repr(target['uplink_port_indices']),
              "num_dhcp_servers": len(target['downlink_vlan_iface']['dhcpv6_server_addrs']),
              "server_ip": str(target['downlink_vlan_iface']['dhcpv6_server_addrs'][0]),
              "relay_iface_ip": str(target['downlink_vlan_iface']['addr']),
              "relay_iface_mac": str(target['downlink_vlan_iface']['mac']),
              "relay_link_local": str(target['down_interface_link_local']),
              "vlan_ip": str(target['downlink_vlan_iface']['addr']),
              "uplink_mac": str(target['uplink_mac']),
              "loopback_ipv6": str(target['loopback_ipv6']),
              "downstream_relay_ip": DOWNSTREAM_RELAY_IP,
              "is_dualtor": "False",
              "kvm_support": True}
    ptf_runner(ptfhost, "ptftests", "dhcpv6_relay_test.DHCPNoRelayTest", platform_dir="ptftests",
               params=params, log_file="/tmp/dhcpv6_relay_test.DHCPNoRelayTest.log", is_python3=True)


def test_interface_binding(duthosts, rand_one_dut_hostname, ptfhost, dut_dhcp_relay_data, dhcp_relay_config,
                           unconfigured_relay_target):
    """dhcp6relay must bind only to interfaces that have a DHCP_RELAY entry and
    must (re)bind to an interface's link-local address when it (re)appears.

    1. A downlink without DHCP_RELAY config must not relay a client Solicit
       (behavioural check; upstream grepped ``ss`` for a per-VLAN socket).
    2. Delete the LLA of every relay interface -> the wildcard socket stays, the
       per-interface LLA sockets disappear.
    3. Re-add the LLAs -> the LLA sockets come back within dhcp6relay's 60 s
       LLA poll interval (70 s budget).
    Runs once per relay mode.
    """
    duthost = duthosts[rand_one_dut_hostname]
    pytest_assert(gpins_utils.wait_dhcp6relay_bound(duthost), "dhcp6relay is not running/listening")

    # 1. Unconfigured interface must not relay.
    run_no_relay_test(ptfhost, duthost, unconfigured_relay_target)

    # 2./3. LLA sockets follow the interface link-local address. Only observable
    # when the relay opens per-address UDP 547 sockets (upstream dhcp6relay); the
    # GPINS relay listens on one AF_PACKET socket, so there is nothing to inspect.
    if not gpins_utils.dhcp6relay_uses_address_sockets(duthost):
        pytest.skip("dhcp6relay on {} has no per-address UDP 547 sockets ({}); LLA socket (re)binding "
                    "is not observable - step 1 (unconfigured interface must not relay) passed".format(
                        duthost.hostname, gpins_utils.dhcp6relay_listeners(duthost)))

    llas = {d['relay_kernel_iface']: d['down_interface_link_local'] for d in dut_dhcp_relay_data
            if d['down_interface_link_local']}
    pytest_require(llas, "No link-local address on the relay interfaces; cannot test LLA binding")
    had_wildcard = "::" in gpins_utils.dhcp6relay_bound_addresses(duthost)
    delete_cmds = ["ip -6 address del {} dev {}".format(d["down_interface_link_local_with_prefix_len"],
                                                        d["relay_kernel_iface"]) for d in dut_dhcp_relay_data]
    add_cmds = ["ip -6 address add {} dev {}".format(d["down_interface_link_local_with_prefix_len"],
                                                     d["relay_kernel_iface"]) for d in dut_dhcp_relay_data]

    def _lla_sockets(expect_exist):
        bound = gpins_utils.dhcp6relay_bound_addresses(duthost)
        res = {iface: ipaddress.ip_address(lla).compressed in bound for iface, lla in llas.items()}
        logger.info("dhcp6relay LLA sockets: %s (all bound: %s)", res, sorted(bound))
        return all(res.values()) if expect_exist else not any(res.values())

    try:
        duthost.shell_cmds(cmds=delete_cmds)
        pytest_assert(gpins_utils.restart_dhcp_relay(duthost), "dhcp_relay container did not come back after restart")
        time.sleep(15)

        if had_wildcard:
            pytest_assert("::" in gpins_utils.dhcp6relay_bound_addresses(duthost),
                          "Wildcard UDP 547 socket for dhcp6relay is not found")
        # No per-interface LLA sockets while the LLAs are removed.
        pytest_assert(_lla_sockets(False), "LLA sockets are found, which is unexpected")

        duthost.shell_cmds(cmds=add_cmds)
        # dhcp6relay re-checks LLAs on its check timer; allow 70s.
        pytest_assert(wait_until(70, 5, 0, _lla_sockets, True), "Expected LLA sockets are not found")
    finally:
        for cmd in add_cmds:
            duthost.shell(cmd, module_ignore_errors=True)
        gpins_utils.restart_dhcp_relay(duthost)
        pytest_assert(gpins_utils.wait_dhcp6relay_bound(duthost), "dhcp6relay is not listening after cleanup")


def test_dhcpv6_relay_counter(ptfhost, duthosts, rand_one_dut_hostname, dut_dhcp_relay_data, dhcp_relay_config):
    """Verify dhcp6relay's per-interface STATE_DB counters (DHCPv6_COUNTER_TABLE).

    PTF sends every client message type (plus one malformed) on the client port
    and every server message type (plus a bare and an unknown Relay-Reply) on an
    uplink; afterwards the RX/TX counters of the relay interface must have moved.
    Runs once per relay mode. GPINS has no dhcpmon, so only the relay's own
    counters are checked.
    """
    duthost = duthosts[rand_one_dut_hostname]

    message_types = ["Unknown", "Solicit", "Advertise", "Request", "Confirm", "Renew", "Rebind", "Reply", "Release",
                     "Decline", "Reconfigure", "Information-Request", "Relay-Forward", "Relay-Reply", "Malformed"]

    for dhcp_relay in dut_dhcp_relay_data:
        relay_intf = dhcp_relay['relay_interface']
        client_intf = dhcp_relay['client_port']
        # svi: client is a physical VLAN member, counters exist for both the port and the SVI.
        # l3_*: the relay interface *is* the client port -> one set of counters.
        counter_ifaces = [relay_intf] if client_intf == relay_intf else [client_intf, relay_intf]
        for ifname in counter_ifaces:
            init_counter(duthost, ifname, message_types)

        mode_params = gpins_ptf_params(dhcp_relay, dhcp_relay_config.get(relay_intf))
        params = {"hostname": duthost.hostname,
                  "client_port_index": dhcp_relay['client_iface']['port_idx'],
                  "leaf_port_indices": repr(dhcp_relay['uplink_port_indices']),
                  "num_dhcp_servers": len(dhcp_relay['downlink_vlan_iface']['dhcpv6_server_addrs']),
                  "server_ip": str(dhcp_relay['downlink_vlan_iface']['dhcpv6_server_addrs'][0]),
                  "relay_iface_ip": str(dhcp_relay['downlink_vlan_iface']['addr']),
                  "relay_iface_mac": str(dhcp_relay['downlink_vlan_iface']['mac']),
                  "relay_link_local": str(dhcp_relay['down_interface_link_local']),
                  "dut_mac": str(dhcp_relay['uplink_mac']),
                  "vlan_ip": str(dhcp_relay['downlink_vlan_iface']['addr']),
                  "loopback_ipv6": str(dhcp_relay['loopback_ipv6']),
                  "is_dualtor": str(dhcp_relay['is_dualtor']),
                  "kvm_support": True,
                  "link_addr": mode_params["link_addr"],
                  "relay_listen_ip": mode_params["relay_listen_ip"],
                  "interface_id": mode_params["interface_id"]}
        # Send the DHCP relay traffic on the PTF host
        ptf_runner(ptfhost,
                   "ptftests",
                   "dhcpv6_counter_test.DHCPCounterTest",
                   platform_dir="ptftests",
                   params=params,
                   log_file="/tmp/dhcpv6_relay_test.DHCPCounterTest.{}.log".format(dhcp_relay['relay_mode']),
                   is_python3=True)

        for ifname in counter_ifaces:
            logger.info("DHCPv6 counters %s: %s", ifname, gpins_utils.get_dhcpv6_counters(duthost, ifname))

        # Validate dhcp6relay counters (STATE_DB DHCPv6_COUNTER_TABLE)
        for type in message_types:
            if type in ["Solicit", "Request", "Confirm", "Renew", "Rebind", "Release", "Decline",
                        "Information-Request"]:
                for ifname in counter_ifaces:
                    check_dhcpv6_relay_counter(duthost, ifname, type, "RX")
            if type in ["Malformed"]:
                # Malformed DHCPv6 Client packet, depend on malformed content.
                # If Type is good but option is malformed
                # First Type will be increased on downlink Ethernet interface first. Then increase Malformed counter
                # on the relay interface.
                check_dhcpv6_relay_counter(duthost, relay_intf, type, "RX")
            if type in ["Unknown"]:
                # dhcp6relay counts two packets as Unknown:
                # 1. Relay-Reply with no relay-msg option (bare Relay-Reply)
                # 2. Relay-Reply wrapping an unrecognized inner message type (msgtype=20)
                check_dhcpv6_relay_counter(duthost, relay_intf, type, "RX")
            if type in ["Advertise", "Reply", "Reconfigure"]:
                for ifname in counter_ifaces:
                    check_dhcpv6_relay_counter(duthost, ifname, type, "TX")
            if type in ["Relay-Forward"]:
                # Relay-Forward, send out from the relay interface first, then send out from uplink interfaces
                check_dhcpv6_relay_counter(duthost, relay_intf, type, "TX")
                # TBD, add uplink interface TX counter check in future
            if type in ["Relay-Reply"]:
                # Single tor, Relay-Reply is accounted on the relay interface
                check_dhcpv6_relay_counter(duthost, relay_intf, type, "RX")

    pytest_assert(wait_until(60, 5, 0, check_interface_status, duthost),
                  "dhcp6relay is no longer running/listening after the test")


def run_dhcptest(ptfhost, duthost, dhcp_relay, relay_cfg):
    """Run the PTF DHCPTest exchange against one relay instance (any relay mode)."""
    relay_intf = dhcp_relay['relay_interface']
    client_port_idx = dhcp_relay['client_iface']['port_idx']
    mode_params = gpins_ptf_params(dhcp_relay, relay_cfg)
    logger.info("DHCPTest on %s (%s): %s", relay_intf, dhcp_relay['relay_mode'], mode_params)

    # Relay A's GUA is arbitrary and outside the DUT VLAN prefix; the
    # responder helper must install a route so DUT Relay B can resolve it.
    downstream_relay_responder = setup_downstream_relay_responder(
        duthost, ptfhost, dhcp_relay['downlink_vlan_iface']['name'],
        client_port_idx)
    # Appendix step 1: resolve the client LLA MAC explicitly if NDv6 does not work.
    client_neigh = gpins_utils.add_static_client_neighbor(
        duthost, relay_intf, _ptf_port_lla(ptfhost, client_port_idx),
        _ptf_port_mac(ptfhost, client_port_idx))

    # Run the DHCP relay test on the PTF host
    try:
        params = {"hostname": duthost.hostname,
                  "client_port_index": client_port_idx,
                  "leaf_port_indices": repr(dhcp_relay['uplink_port_indices']),
                  "num_dhcp_servers": len(dhcp_relay['downlink_vlan_iface']['dhcpv6_server_addrs']),
                  "server_ip": str(dhcp_relay['downlink_vlan_iface']['dhcpv6_server_addrs'][0]),
                  "relay_iface_ip": str(dhcp_relay['downlink_vlan_iface']['addr']),
                  "relay_iface_mac": str(dhcp_relay['downlink_vlan_iface']['mac']),
                  "relay_link_local": str(dhcp_relay['down_interface_link_local']),
                  "vlan_ip": str(dhcp_relay['downlink_vlan_iface']['addr']),
                  "uplink_mac": str(dhcp_relay['uplink_mac']),
                  "loopback_ipv6": str(dhcp_relay['loopback_ipv6']),
                  "downstream_relay_ip": DOWNSTREAM_RELAY_IP,
                  "is_dualtor": str(dhcp_relay['is_dualtor']),
                  "kvm_support": True}
        params.update(mode_params)
        ptf_runner(ptfhost,
                   "ptftests",
                   "dhcpv6_relay_test.DHCPTest",
                   platform_dir="ptftests",
                   params=params,
                   log_file="/tmp/dhcpv6_relay_test.DHCPTest.{}.log".format(dhcp_relay['relay_mode']),
                   is_python3=True)
    finally:
        gpins_utils.del_static_client_neighbor(duthost, client_neigh)
        teardown_downstream_relay_responder(duthost, ptfhost, downstream_relay_responder)

    logger.info("STATE_DB DHCPv6 counters for %s after DHCPTest: %s",
                relay_intf, gpins_utils.get_dhcpv6_counters(duthost, relay_intf))


def set_uplinks(duthost, dhcp_relay, up):
    """Bring the kernel netdevs of all uplinks up/down (``ifconfig`` is not on GPINS).

    TODO(GPINS, Virgo testbed): if the kernel netdev state does not follow the
    physical port on this platform, switch to CONFIG_DB ``PORT|<p>`` admin_status.
    """
    state = "up" if up else "down"
    for iface in dhcp_relay['uplink_interfaces']:
        duthost.shell("ip link set {} {}".format(iface, state), module_ignore_errors=not up)


def uplinks_ready(duthost, dhcp_relay):
    """True when every uplink netdev is oper-up and the kernel has a route to every DHCP server."""
    for iface in dhcp_relay['uplink_interfaces']:
        res = duthost.shell("ip -o link show {}".format(iface), module_ignore_errors=True)
        if res.get("rc", 1) != 0 or "state UP" not in res["stdout"]:
            return False
    for server in dhcp_relay['downlink_vlan_iface']['dhcpv6_server_addrs']:
        res = duthost.shell("ip -6 route get {}".format(server), module_ignore_errors=True)
        if res.get("rc", 1) != 0 or "unreachable" in res["stdout"]:
            return False
    return True


def wait_uplinks_ready(duthost, dhcp_relay, timeout=180):
    """Replacement for upstream wait_all_bgp_up: GPINS routes come from the
    controller, not BGP, so wait on link state + kernel routes instead."""
    pytest_assert(wait_until(timeout, 10, 0, uplinks_ready, duthost, dhcp_relay),
                  "uplinks {} not up / no route to DHCP servers after {}s"
                  .format(dhcp_relay['uplink_interfaces'], timeout))


def test_dhcp_relay_default(ptfhost, dut_dhcp_relay_data, validate_dut_routes_exist, testing_config,
                            dhcp_relay_config):
    """Verify the DUT relays a full DHCPv6 exchange (Solicit/Advertise/Request/Reply)
    plus the RFC 8415 nested Relay-Reply return path.

    Runs once per relay mode (svi / l3_gua / l3_lla, see conftest). GPINS has no
    dhcpmon, so counters are read from the relay's own STATE_DB table only for
    logging here; the strict counter check lives in test_dhcpv6_relay_counter.
    """
    _, duthost = testing_config

    for dhcp_relay in dut_dhcp_relay_data:
        run_dhcptest(ptfhost, duthost, dhcp_relay, dhcp_relay_config.get(dhcp_relay['relay_interface']))

    pytest_assert(wait_until(60, 5, 0, check_interface_status, duthost),
                  "dhcp6relay is no longer running/listening after the test")


def test_dhcp_relay_after_link_flap(ptfhost, dut_dhcp_relay_data, validate_dut_routes_exist, testing_config,
                                    dhcp_relay_config):
    """With the relay running, flap all uplinks, then verify the relay still
    relays a full DHCPv6 exchange. Runs once per relay mode.

    GPINS: uplinks are toggled with ``ip link`` and readiness is link state +
    kernel routes to the DHCP servers (no BGP dependency, per Boyang).
    """
    _, duthost = testing_config

    for dhcp_relay in dut_dhcp_relay_data:
        try:
            set_uplinks(duthost, dhcp_relay, up=False)
            # Give the relay time to notice the uplinks are gone
            time.sleep(20)
        finally:
            set_uplinks(duthost, dhcp_relay, up=True)
            if callable(validate_dut_routes_exist):
                validate_dut_routes_exist()
        wait_uplinks_ready(duthost, dhcp_relay)

        run_dhcptest(ptfhost, duthost, dhcp_relay, dhcp_relay_config.get(dhcp_relay['relay_interface']))

    pytest_assert(wait_until(60, 5, 0, check_interface_status, duthost),
                  "dhcp6relay is no longer running/listening after the test")


def test_dhcp_relay_start_with_uplinks_down(ptfhost, dut_dhcp_relay_data, validate_dut_routes_exist, testing_config,
                                            dhcp_relay_config):
    """Bring the uplinks down, restart the relay while they are down, bring them
    back and verify the relay relays a full DHCPv6 exchange. Runs once per relay mode.

    GPINS: the container is restarted through its init script
    (``gpins_utils.restart_dhcp_relay``) instead of systemctl.
    """
    _, duthost = testing_config

    for dhcp_relay in dut_dhcp_relay_data:
        try:
            set_uplinks(duthost, dhcp_relay, up=False)
            time.sleep(20)

            # Restart the relay while the uplinks are still down; it must come
            # up and bind even though it cannot reach any server yet.
            pytest_assert(gpins_utils.restart_dhcp_relay(duthost), "dhcp_relay container did not restart")
            pytest_assert(gpins_utils.wait_dhcp6relay_bound(duthost),
                          "dhcp6relay did not come up while uplinks were down")
        finally:
            set_uplinks(duthost, dhcp_relay, up=True)
            if callable(validate_dut_routes_exist):
                validate_dut_routes_exist()
        wait_uplinks_ready(duthost, dhcp_relay)

        run_dhcptest(ptfhost, duthost, dhcp_relay, dhcp_relay_config.get(dhcp_relay['relay_interface']))

    pytest_assert(wait_until(60, 5, 0, check_interface_status, duthost),
                  "dhcp6relay is no longer running/listening after the test")


class TestDhcpv6RelayWithMultipleInterfaces:
    """Relay configured on several interfaces at once (GPINS replacement for
    upstream's multiple-VLAN test): every interface must relay with its *own*
    link-address / Interface-ID, i.e. the relay must not mix up interfaces.
    Requires at least two relay targets for the current mode; skips otherwise.
    """

    @pytest.fixture(scope="class")
    def multi_relay_data(self, duthosts, rand_one_dut_hostname, tbinfo, relay_mode, dut_dhcp_relay_data):
        """Configure DHCP_RELAY on every target of this mode; restore afterwards."""
        duthost = duthosts[rand_one_dut_hostname]
        common = relay_common_facts(duthost, tbinfo)
        targets = gpins_utils.relay_targets(duthost, relay_mode, common['ports'], tbinfo)[:2]
        if len(targets) < 2:
            pytest.skip("Need >= 2 {} relay targets, found {}".format(relay_mode, len(targets)))
        servers = dut_dhcp_relay_data[0]['downlink_vlan_iface']['dhcpv6_server_addrs']

        snapshot = gpins_utils.snapshot_config_db(duthost, ["DHCP_RELAY|*"])
        data_list = []
        for target in targets:
            data = build_relay_data(duthost, tbinfo, relay_mode, target, servers, common)
            intf = data['relay_interface']
            gpins_utils.verify_interface_mode(duthost, intf, relay_mode, ports=common['ports'])
            cfg = gpins_utils.get_dhcp_relay_config(duthost, intf)
            if not gpins_utils.parse_dhcpv6_servers(cfg):
                options = {}
                if relay_mode == gpins_utils.RELAY_MODE_L3_LLA:
                    options = {"enable_interface_id": True, "interface_id": data['interface_id']}
                cfg = gpins_utils.set_dhcp_relay_config(duthost, intf, servers, relay_mode=relay_mode, **options)
            data_list.append((data, cfg))
        pytest_assert(gpins_utils.wait_dhcp6relay_bound(duthost), "dhcp6relay is not running/listening")
        # Give the relay a moment to pick up the new entries (config convergence
        # is asserted by test_dhcp_relay_config_convergence).
        time.sleep(10)

        yield data_list

        gpins_utils.restore_config_db(duthost, snapshot)
        gpins_utils.wait_dhcp6relay_bound(duthost)

    def test_dhcp_relay_default(self, ptfhost, duthosts, rand_one_dut_hostname, validate_dut_routes_exist,
                                multi_relay_data):
        """Each configured interface relays with the correct link-address / Interface-ID."""
        duthost = duthosts[rand_one_dut_hostname]
        for dhcp_relay, cfg in multi_relay_data:
            run_dhcptest(ptfhost, duthost, dhcp_relay, cfg)
        pytest_assert(wait_until(60, 5, 0, check_interface_status, duthost),
                      "dhcp6relay is no longer running/listening after the test")

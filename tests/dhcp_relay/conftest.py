import pytest
import ipaddress
import logging

from tests.common.utilities import wait_until
from tests.common.helpers.assertions import pytest_assert as py_assert
try:
    from tests.common.dhcp_relay_utils import check_routes_to_dhcp_server
except ImportError:
    from tests.dhcp_relay.dhcp_relay_utils import check_routes_to_dhcp_server
from tests.dhcp_relay import gpins_dpb
from tests.dhcp_relay import gpins_utils

logger = logging.getLogger(__name__)

SINGLE_TOR_MODE = 'single'
DUAL_TOR_MODE = 'dual'


def pytest_addoption(parser):
    """
    Adds options to pytest that are used by the COPP tests.
    """
    parser.addoption(
        "--stress_restart_round",
        action="store",
        type=int,
        default=10,
        help="Set custom restart rounds",
    )
    parser.addoption(
        "--stress_restart_duration",
        action="store",
        type=int,
        default=90,
        help="Set custom restart rounds",
    )
    parser.addoption(
        "--stress_restart_pps",
        action="store",
        type=int,
        default=100,
        help="Set custom restart rounds",
    )
    parser.addoption(
        "--max_packets_per_sec",
        action="store",
        type=int,
        help="Set maximum packets per second for stress test",
    )
    parser.addoption(
        "--dhcp_relay_mode",
        action="store",
        default=",".join(gpins_utils.RELAY_MODES),
        help="Comma-separated GPINS relay modes to run: {} (default: all). "
             "Modes the DUT is not configured for are skipped.".format(", ".join(gpins_utils.RELAY_MODES)),
    )
    parser.addoption(
        "--dhcp_relay_enable_feature",
        action="store_true",
        default=False,
        help="Bring up DHCP relay for the session if it is not already: set FEATURE|dhcp_relay=enabled "
             "(the GPINS relay only forwards while it is enabled) and start dhcprelay-con via its init "
             "script (nothing on GPINS starts it from the flag). Both are reverted at session end. "
             "Without this flag the directory is skipped when the relay is not up (upstream behaviour).",
    )
    parser.addoption(
        "--dhcp_relay_static_route",
        action="store",
        choices=("auto", "always", "never"),
        default="auto",
        help="How relayed packets reach the DHCPv6 server without depending on bgp-con: 'auto' (default) "
             "installs a static on-link route + neighbour toward the PTF behind the first uplink only when "
             "the DUT has no route to the server; 'always' installs it unconditionally; 'never' requires an "
             "existing (BGP) route. Entries are removed at module teardown.",
    )
    parser.addoption(
        "--dhcp_relay_loopback_ipv6",
        action="store",
        default=gpins_utils.DEFAULT_LOOPBACK_IPV6,
        help="IPv6 address/prefix to put on Loopback0 for the session if it has none (default {}). "
             "dhcp6relay binds its loopback listen socket there once Interface-ID is enabled and exits "
             "without it. 'none' disables the fixture.".format(gpins_utils.DEFAULT_LOOPBACK_IPV6),
    )
    # --- Dynamic Port Breakout (test_dhcpv6_relay_dpb_gpins.py). Only per-testbed values are options; the verified
    # gNMI client, target, prefix and access recipe are constants in gpins_dpb.py.
    parser.addoption(
        "--dhcp_relay_dpb_parent",
        action="store",
        default=gpins_dpb.DEFAULT_PARENT,
        help="Cage master port (lane 1, has a BREAKOUT_CFG row) to break out; must not be a test uplink or the "
             "PTF downlink (default {}: an un-cabled spare on ju3u1m1).".format(gpins_dpb.DEFAULT_PARENT),
    )
    parser.addoption(
        "--dhcp_relay_dpb_mode",
        action="store",
        default=gpins_dpb.DEFAULT_MODE,
        help="Target breakout mode of the whole 8-lane cage in platform.json notation, e.g. '4x200G' or the "
             "mixed '2x200G(4)+1x400G(4)' (default) which splits only the first 400G port.",
    )
    parser.addoption(
        "--dhcp_relay_dpb_allow_root_replace",
        action="store_true",
        default=False,
        help="Acknowledge that the breakout is a gNMI 'replace' at the root path (cage component + interfaces of the "
             "new layout, as sonic-pins sends it). Evidence that it is safe: the UMF binary implements entry-level "
             "partial replace ('Since its partial replace ...') and the request passed end to end on ju3u1m1; the "
             "opt-in remains because a whole-config replace could take the management path with it. Without it "
             "test_dpb_preflight and test_dhcpv6_relay_dpb skip.",
    )
    parser.addoption(
        "--dhcp_relay_dpb_recovery",
        action="store",
        choices=("stack-restart", "none"),
        default="stack-restart",
        help="How to undo the breakout in the test's finally block: 'stack-restart' (default; gpins-stack restart "
             "reloads config_db.json, then the pre-DPB snapshot is re-applied) or 'none' (leave the port layout "
             "for manual inspection; CONFIG_DB/kernel session state is still re-applied).",
    )
    parser.addoption(
        "--dhcp_relay_dpb_settle_timeout",
        action="store",
        type=int,
        default=180,
        help="Seconds to wait for the new ports in kernel/CONFIG_DB/APPL_DB after the gNMI Set (default 180).",
    )


def _selected_relay_modes(config):
    raw = config.getoption("--dhcp_relay_mode") or ""
    raw_modes = [m.strip() for m in raw.split(",") if m.strip()]
    unknown = [m for m in raw_modes if m not in gpins_utils.RELAY_MODE_ALIASES]
    if unknown:
        raise pytest.UsageError("--dhcp_relay_mode: unknown mode(s) {}; expected {}".format(
            unknown, list(gpins_utils.RELAY_MODES)))
    modes = []
    for m in raw_modes:
        norm = gpins_utils.normalize_relay_mode(m)
        if norm not in modes:
            modes.append(norm)
    return modes or list(gpins_utils.RELAY_MODES)


def pytest_generate_tests(metafunc):
    """Run every test that (transitively) uses ``relay_mode`` once per selected GPINS relay mode.

    The test plan requires all DHCP-relay tests to be exercised in l3_lla,
    l3_gua and vlan scenarios without writing separate tests per mode.
    Module scope configures the DUT once per mode and keeps ``dut_dhcp_relay_data`` cached.
    """
    if "relay_mode" in metafunc.fixturenames:
        metafunc.parametrize("relay_mode", _selected_relay_modes(metafunc.config), indirect=True, scope="module")


@pytest.fixture(scope="module")
def relay_mode(request, duthost, tbinfo):
    """Configure the DUT for the GPINS relay mode under test (l3_lla / l3_gua / vlan) and tear it down after."""
    mode = gpins_utils.normalize_relay_mode(request.param)
    supported, reason, state = gpins_utils.configure_relay_mode(duthost, mode, tbinfo=tbinfo)
    if not supported:
        pytest.skip("relay mode not available on {}: {}".format(duthost.hostname, reason))
    try:
        yield mode
    finally:
        gpins_utils.teardown_relay_mode(duthost, state)


DHCP_RELAY_FEATURE = gpins_utils.DHCP_RELAY_FEATURE
DHCP_RELAY_FEATURE_START_TIMEOUT = 120


@pytest.fixture(scope="session")
def dhcp_relay_feature_enabled(request, duthost):
    """Make sure DHCP relay is up for the session, if ``--dhcp_relay_enable_feature`` was given.

    On GPINS two independent things must hold (verified on ju3u1m1, 2026-10-08):
      1. ``FEATURE|dhcp_relay state=enabled`` - the relay manager only spawns
         dhcp6relay while this is set; nothing else reads it.
      2. ``dhcprelay-con`` running - started by ``/etc/init.d/dhcprelay-container``,
         *not* by the flag (the image's hostcfgd has no feature handler).
    Each is changed only if needed and each change is reverted at session end,
    so a DUT that already had the relay up is left untouched.

    Returns "enabled" when both hold, otherwise the FEATURE state found.
    """
    state = gpins_utils.feature_state(duthost, DHCP_RELAY_FEATURE)
    relay_container = gpins_utils.DHCP_RELAY_CONTAINER
    running = gpins_utils.is_container_running(duthost)
    if state == "enabled" and running:
        return "enabled"
    if not request.config.getoption("--dhcp_relay_enable_feature"):
        return state if running else "{} (container {} not running)".format(state, relay_container)

    if state != "enabled":
        logger.info("FEATURE|%s state=%r on %s; enabling for this session", DHCP_RELAY_FEATURE, state, duthost.hostname)
        gpins_utils.set_feature_state(duthost, DHCP_RELAY_FEATURE, "enabled")

        def _restore_flag():
            logger.info("Restoring FEATURE|%s state=%r on %s", DHCP_RELAY_FEATURE, state, duthost.hostname)
            gpins_utils.set_feature_state(duthost, DHCP_RELAY_FEATURE, state or "disabled")

        request.addfinalizer(_restore_flag)

    if not running:
        logger.info("%s not running on %s; starting it for this session", relay_container, duthost.hostname)
        request.addfinalizer(lambda: gpins_utils.stop_dhcp_relay(duthost))
        py_assert(
            gpins_utils.start_dhcp_relay(duthost, timeout=DHCP_RELAY_FEATURE_START_TIMEOUT),
            "{} did not start within {}s ({} start)".format(
                relay_container, DHCP_RELAY_FEATURE_START_TIMEOUT, gpins_utils.DHCP_RELAY_INIT_SCRIPT))

    return "enabled"


@pytest.fixture(scope="module", autouse=True)
def check_dhcp_feature_status(duthost, dhcp_relay_feature_enabled):
    """Skip every test in this directory unless DHCP relay is up on the DUT.

    GPINS has no ``show`` CLI, so FEATURE|dhcp_relay is read straight from CONFIG_DB
    and the container state from ``ctr tasks ls``.
    Pass ``--dhcp_relay_enable_feature`` to bring it up for the session instead of skipping.
    """
    state = dhcp_relay_feature_enabled
    if state != "enabled":
        pytest.skip("dhcp_relay is not up (FEATURE|dhcp_relay state={!r}); "
                    "pass --dhcp_relay_enable_feature to enable it for the session".format(state))


@pytest.fixture(scope="session")
def dhcp_relay_loopback_ipv6(request, duthost):
    """Guarantee Loopback0 has an IPv6 address for the session (see gpins_utils.ensure_loopback_ipv6).

    Verified on ju3u1m1 (2026-10-09): without it dhcp6relay exits 1 as soon as
    Interface-ID is enabled ("Failed to create loopback listen socket"), which
    Boyang's rule makes mandatory on LLA-only ports. Only touches the DUT when
    Loopback0 has no IPv6; whatever it adds is removed at session end.
    Returns the loopback's (address, prefixlen) or (None, None) if disabled.
    """
    wanted = request.config.getoption("--dhcp_relay_loopback_ipv6")
    if wanted and wanted.lower() != "none":
        created = gpins_utils.ensure_loopback_ipv6(duthost, wanted)
        if created:
            request.addfinalizer(lambda: gpins_utils.remove_loopback_ipv6(duthost, created))
    return gpins_utils.get_loopback_ipv6(duthost)


@pytest.fixture(scope="function")
def dhcp_relay_config_snapshot(duthosts, rand_one_dut_hostname, dhcp_relay_config):
    """Snapshot ``DHCP_RELAY|*`` around a GPINS test that rewrites it; restore and re-converge afterwards.

    Depends on ``dhcp_relay_config`` (test_dhcpv6_relay.py, imported by the GPINS test modules) so the
    baseline being snapshotted is the one the rest of the module runs against.
    """
    duthost = duthosts[rand_one_dut_hostname]
    snapshot = gpins_utils.snapshot_config_db(duthost, ["DHCP_RELAY|*"])
    yield snapshot
    if not gpins_utils.is_container_running(duthost):
        gpins_utils.start_dhcp_relay(duthost, timeout=120)
    gpins_utils.restore_config_db(duthost, snapshot)
    for key in snapshot["entries"]:
        gpins_utils.assert_dhcp_relay_converged(duthost, key.split("|", 1)[1])
    py_assert(gpins_utils.wait_dhcp6relay_bound(duthost),
              "dhcp6relay not running/listening after restoring DHCP_RELAY config")


@pytest.fixture(scope="module")
def dut_dhcp_relay_data(duthosts, rand_one_dut_hostname, ptfhost, tbinfo):
    """ Fixture which returns a list of dictionaries where each dictionary contains
        data necessary to test one instance of a DHCP relay agent running on the DuT.
        This fixture is scoped to the module, as the data it gathers can be used by
        all tests in this module. It does not need to be run before each test.
    """
    duthost = duthosts[rand_one_dut_hostname]
    dhcp_relay_data_list = []

    mg_facts = duthost.get_extended_minigraph_facts(tbinfo)

    switch_loopback_ip = mg_facts['minigraph_lo_interfaces'][0]['addr']

    # SONiC spawns one DHCP relay agent per VLAN interface configured on the DUT
    vlan_dict = mg_facts['minigraph_vlans']
    for vlan_iface_name, vlan_info_dict in list(vlan_dict.items()):
        # Filter(remove) PortChannel interfaces from VLAN members list
        vlan_members = [port for port in vlan_info_dict['members'] if 'PortChannel' not in port]
        # Gather information about the downlink VLAN interface this relay agent is listening on
        downlink_vlan_iface = {}
        downlink_vlan_iface['name'] = vlan_iface_name

        for vlan_interface_info_dict in mg_facts['minigraph_vlan_interfaces']:
            if vlan_interface_info_dict['attachto'] == vlan_iface_name:
                downlink_vlan_iface['addr'] = vlan_interface_info_dict['addr']
                downlink_vlan_iface['mask'] = vlan_interface_info_dict['mask']
                subnet = ipaddress.IPv4Interface("{}/{}".format(vlan_interface_info_dict['addr'],
                                                 vlan_interface_info_dict['mask'])).network
                downlink_vlan_iface['link_selection_ip'] = str(subnet.network_address)
                break

        # Obtain MAC address of the VLAN interface
        res = duthost.shell('cat /sys/class/net/{}/address'.format(vlan_iface_name))
        downlink_vlan_iface['mac'] = res['stdout']

        downlink_vlan_iface['dhcp_server_addrs'] = mg_facts['dhcp_servers']

        # We choose the physical interface where our DHCP client resides to be index of first interface
        # with alias (ignore PortChannel) in the VLAN
        client_iface = {}
        for port in vlan_members:
            if port in mg_facts['minigraph_port_name_to_alias_map']:
                break
        else:
            continue
        client_iface['name'] = port
        client_iface['alias'] = mg_facts['minigraph_port_name_to_alias_map'][client_iface['name']]
        client_iface['port_idx'] = mg_facts['minigraph_ptf_indices'][client_iface['name']]

        # Obtain uplink port indices for this DHCP relay agent

        uplink_interfaces, uplink_port_indices = calculate_uplink_interfaces_and_port_indices(mg_facts)
        other_client_ports_indices = []
        for iface_name in vlan_members:
            if mg_facts['minigraph_ptf_indices'][iface_name] == client_iface['port_idx']:
                pass
            else:
                other_client_ports_indices.append(mg_facts['minigraph_ptf_indices'][iface_name])

        dhcp_relay_data = {}
        dhcp_relay_data['downlink_vlan_iface'] = downlink_vlan_iface
        dhcp_relay_data['client_iface'] = client_iface
        dhcp_relay_data['other_client_ports'] = other_client_ports_indices
        dhcp_relay_data['uplink_interfaces'] = uplink_interfaces
        dhcp_relay_data['uplink_port_indices'] = uplink_port_indices
        dhcp_relay_data['switch_loopback_ip'] = str(switch_loopback_ip)
        dhcp_relay_data['portchannels'] = mg_facts['minigraph_portchannels']
        dhcp_relay_data['vlan_members'] = vlan_members

        # Add loopback interface name (needed for source_interface)
        loopback_iface = mg_facts['minigraph_lo_interfaces'][0]['name']
        dhcp_relay_data['loopback_iface'] = loopback_iface
        portchannels_with_ips = {}
        portchannels_ip_list = []

        for portchannel_name, portchannel_info in mg_facts['minigraph_portchannels'].items():
            for pc_interface in mg_facts['minigraph_portchannel_interfaces']:
                if pc_interface['attachto'] == portchannel_name:
                    ip_with_mask = f"{pc_interface['addr']}/{pc_interface['mask']}"

                    # Optional: format to standard CIDR
                    # formatted_ip = str(ipaddress.ip_interface(ip_with_mask))
                    ip_obj = ipaddress.ip_interface(ip_with_mask)
                    # Skip IPv6 if needed
                    if ip_obj.version != 4:
                        continue
                    hosts = list(ip_obj.network.hosts())
                    if len(hosts) < 2:
                        logger.warning(f"Not enough hosts for nexthop in {ip_with_mask}")
                        continue

                    nexthop = str(hosts[1]) if str(ip_obj.ip) == str(hosts[0]) else str(hosts[0])
                    if portchannel_name not in portchannels_with_ips:
                        portchannels_with_ips[portchannel_name] = []
                    # Save as flat dictionary
                    portchannels_with_ips[portchannel_name] = {
                        "ip": str(ip_obj),
                        "nexthop": nexthop
                    }
                    # Append the IP to the list
                    portchannels_ip_list.append(str(ip_obj))

        dhcp_relay_data['portchannels_with_ips'] = portchannels_with_ips
        dhcp_relay_data['portchannels_ip_list'] = portchannels_ip_list

        # Obtain MAC address of an uplink interface because vlan mac may be different than that of physical interfaces
        res = duthost.shell('cat /sys/class/net/{}/address'.format(uplink_interfaces[0]))
        dhcp_relay_data['uplink_mac'] = res['stdout']
        # get standby duthost if dualtor
        if 'dualtor' in tbinfo['topo']['name']:
            standby_duthost = [duthost for duthost in duthosts if duthost != duthosts[rand_one_dut_hostname]][0]
            res = standby_duthost.shell('cat /sys/class/net/{}/address'.format(uplink_interfaces[0]))
            dhcp_relay_data['standby_uplink_mac'] = res['stdout']
            dhcp_relay_data['standby_dut_lo_addr'] = \
                mg_facts["minigraph_devices"][standby_duthost.sonichost.hostname]['lo_addr']
            standby_mg_facts = standby_duthost.get_extended_minigraph_facts(tbinfo)
            standby_uplink_interfaces, standby_uplink_port_indices = \
                calculate_uplink_interfaces_and_port_indices(standby_mg_facts)
            dhcp_relay_data['standby_uplink_port_indices'] = standby_uplink_port_indices
        dhcp_relay_data['default_gw_ip'] = mg_facts['minigraph_mgmt_interface']['gwaddr']

        dhcp_relay_data_list.append(dhcp_relay_data)

    return dhcp_relay_data_list


def calculate_uplink_interfaces_and_port_indices(mg_facts):
    uplink_interfaces = []
    uplink_port_indices = []
    for iface_name, neighbor_info_dict in list(mg_facts['minigraph_neighbors'].items()):
        if neighbor_info_dict['name'] in mg_facts['minigraph_devices']:
            neighbor_device_info_dict = mg_facts['minigraph_devices'][neighbor_info_dict['name']]
            if 'type' in neighbor_device_info_dict and neighbor_device_info_dict['type'] in \
                    ['LeafRouter', 'MgmtLeafRouter', 'BackEndLeafRouter']:
                # If this uplink's physical interface is a member of a portchannel interface,
                # we record the name of the portchannel interface here, as this is the actual
                # interface the DHCP relay will listen on.
                iface_is_portchannel_member = False
                for portchannel_name, portchannel_info_dict in list(mg_facts['minigraph_portchannels'].items()):
                    if 'members' in portchannel_info_dict and iface_name in portchannel_info_dict['members']:
                        iface_is_portchannel_member = True
                        if portchannel_name not in uplink_interfaces:
                            uplink_interfaces.append(portchannel_name)
                        break
                    # If the uplink's physical interface is not a member of a portchannel,
                    # add it to our uplink interfaces list
                if not iface_is_portchannel_member:
                    uplink_interfaces.append(iface_name)

                uplink_port_indices.append(mg_facts['minigraph_ptf_indices'][iface_name])
    return uplink_interfaces, uplink_port_indices


@pytest.fixture(scope="module")
def validate_dut_routes_exist(duthosts, rand_one_dut_hostname, dut_dhcp_relay_data):
    """Fixture to valid a route to each DHCP server exist
    """
    py_assert(wait_until(360, 5, 0, check_routes_to_dhcp_server, duthosts[rand_one_dut_hostname],
                         dut_dhcp_relay_data),
              "Packets relayed to DHCP server should go through default route via upstream neighbor, but now it's" +
              " going through mgmt interface, which means device is in an unhealthy status")


@pytest.fixture(scope="module")
def testing_config(duthosts, rand_one_dut_hostname, tbinfo):
    duthost = duthosts[rand_one_dut_hostname]

    if 'dualtor' in tbinfo['topo']['name']:
        yield DUAL_TOR_MODE, duthost
    else:
        yield SINGLE_TOR_MODE, duthost


@pytest.fixture(scope="function")
def clean_processes_after_stress_test(ptfhost):
    yield
    ptfhost.shell("kill -9 $(ps aux | grep  dhcp_relay_stress_test | grep -v 'grep' | awk '{print $2}')",
                  module_ignore_errors=True)

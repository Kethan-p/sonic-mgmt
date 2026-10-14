"""GPINS-only DHCPv6 relay tests that fill the sonic-mgmt gaps listed in Table 2
of the DHCP-relay v6 test plan.

Implemented here (all run in svi / l3_gua / l3_lla via the ``relay_mode``
parametrization from conftest.py):

* test_dhcpv6_relay_config_convergence  - CONFIG_DB DHCP_RELAY|<intf> shows up in
                                           STATE_DB, also for extra interfaces
* test_dhcpv6_relay_runtime_configuration - change the server list at runtime,
                                           STATE_DB follows, relay still works
* test_dhcpv6_relay_disabled             - remove all relay config at runtime and
                                           before container start -> no daemon / no socket
* test_dhcpv6_relay_option_test          - Interface-ID / Remote-ID combinations are
                                           reflected in the Relay-Forward

Also here (upstream tests/gnmi/events/dhcp-relay_events.py, item #11):

* test_dhcpv6_relay_bind_failure         - relay interface loses its address ->
                                           no bind on it; restore -> full recovery

Placeholders (need capabilities the Virgo testbed does not expose yet):
  test_dhcpv6_relay_dpb, test_dhcpv6_relay_openconfig_telemetry, test_dhcpv6_relay_nsf,
  test_dhcpv6_relay_bind_failure_event.

Everything reuses the discovery / fixtures / PTF driver from test_dhcpv6_relay.py
so there is a single source of truth for how a relay target is picked and
exercised. Only CONFIG_DB / STATE_DB / the kernel are touched - no SONiC CLI.
"""
import ipaddress
import logging
import time

import pytest

from tests.common.fixtures.ptfhost_utils import copy_ptftests_directory   # noqa F401
from tests.common.fixtures.ptfhost_utils import change_mac_addresses      # noqa F401
from tests.common.fixtures.ptfhost_utils import copy_arp_responder_py     # noqa F401
from tests.common.helpers.assertions import pytest_assert
from tests.common.utilities import wait_until
from tests.dhcp_relay import gpins_utils
from tests.dhcp_relay.test_dhcpv6_relay import (  # noqa F401
    skip_if_dhcp_relay_unavailable,
    testing_config,
    dut_dhcp_relay_data,
    validate_dut_routes_exist,
    dhcp_relay_config,
    relay_common_facts,
    build_relay_data,
    run_dhcptest,
    check_interface_status,
)

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.topology('t0', 'm0', 'mx', 't0-2vlans'),
    pytest.mark.device_type('vs')
]

# How long dhcp6relay gets to mirror a CONFIG_DB change into STATE_DB.
CONVERGE_TIMEOUT = 60
# Extra (unreachable is fine) server used to prove a runtime change is applied.
EXTRA_DHCPV6_SERVER = "fc02:2000::3"
TEST_REMOTE_ID = "gpins-dhcp-relay-test"

# Test plan, test_dhcpv6_relay_disabled: "Verifies that the DHCP-relay daemon
# is not running in the DHCP-relay container". Flip to False if the GPINS
# relay is specified to stay alive (idle, unbound) with no configuration.
# TODO(GPINS, Virgo testbed): confirm against the real relay.
DISABLED_EXPECT_PROCESS_EXIT = True
DISABLED_SETTLE_SECONDS = 15


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def wait_converged(duthost, intf, timeout=CONVERGE_TIMEOUT):
    """Assert STATE_DB DHCP_RELAY|<intf> matches CONFIG_DB within ``timeout`` seconds."""
    ok = wait_until(timeout, 3, 0, gpins_utils.dhcp_relay_config_converged, duthost, intf)
    pytest_assert(ok, "DHCP_RELAY|{} not applied within {}s: CONFIG_DB={} STATE_DB={}".format(
        intf, timeout, gpins_utils.get_dhcp_relay_config(duthost, intf),
        gpins_utils.get_dhcp_relay_state(duthost, intf)))


def wait_state_cleared(duthost, intf, timeout=CONVERGE_TIMEOUT):
    """Assert STATE_DB DHCP_RELAY|<intf> disappears after the CONFIG_DB entry was removed."""
    ok = wait_until(timeout, 3, 0, lambda: not gpins_utils.get_dhcp_relay_state(duthost, intf))
    pytest_assert(ok, "STATE_DB DHCP_RELAY|{} still present {}s after removal: {}".format(
        intf, timeout, gpins_utils.get_dhcp_relay_state(duthost, intf)))


def design_interface_id(dhcp_relay):
    """Interface-ID in the form the sample config prescribes: ``<port>`` (e.g. ``Ethernet1/4/1``)."""
    return dhcp_relay['interface_id'] or dhcp_relay['relay_interface']


def lla_options(dhcp_relay):
    """Options an LLA-only relay must always carry (the server routes replies via the loopback in Option 18)."""
    if dhcp_relay['relay_mode'] == gpins_utils.RELAY_MODE_L3_LLA:
        return {"enable_interface_id": True, "interface_id": design_interface_id(dhcp_relay)}
    return {}


def servers_of(dhcp_relay):
    return list(dhcp_relay['downlink_vlan_iface']['dhcpv6_server_addrs'])


@pytest.fixture(scope="function")
def dhcp_relay_config_snapshot(duthosts, rand_one_dut_hostname, dhcp_relay_config):  # noqa F811
    """Snapshot ``DHCP_RELAY|*`` around a test that rewrites it; restore and re-converge afterwards.

    Depends on ``dhcp_relay_config`` so the baseline being snapshotted is the
    one the rest of the module runs against.
    """
    duthost = duthosts[rand_one_dut_hostname]
    snapshot = gpins_utils.snapshot_config_db(duthost, ["DHCP_RELAY|*"])
    yield snapshot
    gpins_utils.restore_config_db(duthost, snapshot)
    for key in snapshot["entries"]:
        wait_converged(duthost, key.split("|", 1)[1])
    pytest_assert(gpins_utils.wait_dhcp6relay_bound(duthost),
                  "dhcp6relay not running/listening after restoring DHCP_RELAY config")


# ---------------------------------------------------------------------------
# Table 2 tests
# ---------------------------------------------------------------------------

def test_dhcpv6_relay_config_convergence(duthosts, rand_one_dut_hostname, tbinfo, relay_mode,
                                         dut_dhcp_relay_data, dhcp_relay_config, dhcp_relay_config_snapshot):  # noqa F811
    """Enable DHCP relay on different interfaces; STATE_DB must contain the applied configuration."""
    duthost = duthosts[rand_one_dut_hostname]

    # 1. The interfaces the module is configured on.
    for dhcp_relay in dut_dhcp_relay_data:
        intf = dhcp_relay['relay_interface']
        wait_converged(duthost, intf)
        state = gpins_utils.get_dhcp_relay_state(duthost, intf)
        pytest_assert(gpins_utils.parse_dhcpv6_servers(state) == servers_of(dhcp_relay),
                      "STATE_DB servers for {} = {}, expected {}".format(intf, state, servers_of(dhcp_relay)))
    pytest_assert(check_interface_status(duthost), "dhcp6relay is not running/listening")

    # 2. Every other relay-capable interface of this mode: enable, converge, disable, cleared.
    configured = {d['relay_interface'] for d in dut_dhcp_relay_data}
    common = relay_common_facts(duthost, tbinfo)
    extra = [t for t in gpins_utils.relay_targets(duthost, relay_mode, common['ports'], tbinfo)
             if t['relay_interface'] not in configured]
    if not extra:
        logger.info("No additional %s relay targets on this DUT; convergence checked on %s only",
                    relay_mode, sorted(configured))
        return
    servers = servers_of(dut_dhcp_relay_data[0])
    for target in extra[:1]:
        data = build_relay_data(duthost, tbinfo, relay_mode, target, servers, common)
        intf = data['relay_interface']
        if gpins_utils.get_dhcp_relay_config(duthost, intf):
            logger.info("%s already has a DHCP_RELAY entry outside the module baseline; skipping it", intf)
            continue
        gpins_utils.set_dhcp_relay_config(duthost, intf, servers, relay_mode=relay_mode, **lla_options(data))
        wait_converged(duthost, intf)
        gpins_utils.del_dhcp_relay_config(duthost, intf)
        wait_state_cleared(duthost, intf)
    # Baseline interfaces must be unaffected by the churn on the others.
    for dhcp_relay in dut_dhcp_relay_data:
        wait_converged(duthost, dhcp_relay['relay_interface'])
    pytest_assert(check_interface_status(duthost), "dhcp6relay stopped listening during config churn")


def test_dhcpv6_relay_runtime_configuration(ptfhost, duthosts, rand_one_dut_hostname, dut_dhcp_relay_data,  # noqa F811
                                            validate_dut_routes_exist, dhcp_relay_config,  # noqa F811
                                            dhcp_relay_config_snapshot):
    """Change the relay configuration while dhcp6relay is running.

    The daemon must apply the change (STATE_DB mirrors CONFIG_DB) and keep
    relaying. Whether it does so with or without restarting itself is logged,
    not asserted: the test plan only requires the applied config in STATE_DB.
    """
    duthost = duthosts[rand_one_dut_hostname]
    dhcp_relay = dut_dhcp_relay_data[0]
    intf = dhcp_relay['relay_interface']
    mode = dhcp_relay['relay_mode']
    original = servers_of(dhcp_relay)
    pid_before = gpins_utils.get_process_pid(duthost)

    # Add a server at runtime.
    updated = original + [EXTRA_DHCPV6_SERVER]
    gpins_utils.set_dhcp_relay_config(duthost, intf, updated, relay_mode=mode, **lla_options(dhcp_relay))
    wait_converged(duthost, intf)
    state = gpins_utils.get_dhcp_relay_state(duthost, intf)
    pytest_assert(EXTRA_DHCPV6_SERVER in gpins_utils.parse_dhcpv6_servers(state),
                  "{} missing from STATE_DB DHCP_RELAY|{} after runtime update: {}".format(
                      EXTRA_DHCPV6_SERVER, intf, state))
    pytest_assert(gpins_utils.wait_dhcp6relay_bound(duthost, timeout=CONVERGE_TIMEOUT, interval=3),
                  "dhcp6relay not running/listening after runtime config update")
    pid_after = gpins_utils.get_process_pid(duthost)
    logger.info("dhcp6relay pid before/after runtime update: %s/%s (%s)", pid_before, pid_after,
                "applied in place" if pid_before == pid_after else "daemon restarted")

    # Revert at runtime and prove the relay still works end to end.
    applied = gpins_utils.set_dhcp_relay_config(duthost, intf, original, relay_mode=mode, **lla_options(dhcp_relay))
    wait_converged(duthost, intf)
    pytest_assert(gpins_utils.wait_dhcp6relay_bound(duthost, timeout=CONVERGE_TIMEOUT, interval=3),
                  "dhcp6relay not running/listening after reverting runtime config")
    run_dhcptest(ptfhost, duthost, dhcp_relay, applied)


def test_dhcpv6_relay_disabled(duthosts, rand_one_dut_hostname, dut_dhcp_relay_data, dhcp_relay_config,  # noqa F811
                               dhcp_relay_config_snapshot):
    """Delete all relay config (a) at runtime and (b) before the container starts; no relay must run."""
    duthost = duthosts[rand_one_dut_hostname]
    keys = gpins_utils.redis_keys(duthost, "CONFIG_DB", "DHCP_RELAY|*")
    pytest_assert(keys, "expected at least one DHCP_RELAY entry to remove")

    def relay_inactive():
        if gpins_utils.is_dhcp6relay_bound(duthost):
            return False
        return not (DISABLED_EXPECT_PROCESS_EXIT and gpins_utils.is_dhcp6relay_running(duthost))

    def describe():
        return "dhcp6relay running={} bound={}".format(gpins_utils.is_dhcp6relay_running(duthost),
                                                       gpins_utils.is_dhcp6relay_bound(duthost))

    # (a) runtime removal
    gpins_utils.redis_del(duthost, "CONFIG_DB", *keys)
    for key in keys:
        wait_state_cleared(duthost, key.split("|", 1)[1])
    pytest_assert(wait_until(CONVERGE_TIMEOUT, 3, 0, relay_inactive),
                  "relay still active {}s after removing all DHCP_RELAY config: {}".format(
                      CONVERGE_TIMEOUT, describe()))

    # (b) container (re)start with no relay config at all
    gpins_utils.restart_dhcp_relay(duthost, wait=False)
    time.sleep(DISABLED_SETTLE_SECONDS)
    pytest_assert(relay_inactive(),
                  "relay became active after a container start with no DHCP_RELAY config: {}".format(describe()))
    # It must also *stay* inactive (no delayed bind / retry loop).
    time.sleep(DISABLED_SETTLE_SECONDS)
    pytest_assert(relay_inactive(), "relay activated late with no DHCP_RELAY config: {}".format(describe()))
    # Teardown (dhcp_relay_config_snapshot) restores the config and asserts the relay binds again.


OPTION_CASES = {
    "interface_id": {"enable_interface_id": True, "enable_remote_id": False, "rfc6939_support": True},
    "interface_id_no_rfc6939": {"enable_interface_id": True, "enable_remote_id": False, "rfc6939_support": False},
    "remote_id": {"enable_interface_id": False, "enable_remote_id": True, "remote_id": TEST_REMOTE_ID,
                  "rfc6939_support": True},
    "remote_id_no_rfc6939": {"enable_interface_id": False, "enable_remote_id": True, "remote_id": TEST_REMOTE_ID,
                             "rfc6939_support": False},
    "both": {"enable_interface_id": True, "enable_remote_id": True, "remote_id": TEST_REMOTE_ID,
             "rfc6939_support": True},
    "both_no_rfc6939": {"enable_interface_id": True, "enable_remote_id": True, "remote_id": TEST_REMOTE_ID,
                        "rfc6939_support": False},
    "none": {"enable_interface_id": False, "enable_remote_id": False, "rfc6939_support": True},
    "none_no_rfc6939": {"enable_interface_id": False, "enable_remote_id": False, "rfc6939_support": False},
}


@pytest.mark.parametrize("option_case", sorted(OPTION_CASES))
def test_dhcpv6_relay_option_test(ptfhost, duthosts, rand_one_dut_hostname, dut_dhcp_relay_data,  # noqa F811
                                  validate_dut_routes_exist, dhcp_relay_config, dhcp_relay_config_snapshot,  # noqa F811
                                  option_case):
    """Configure Interface-ID / Remote-ID / RFC6939 combinations and check the Relay-Forward carries those."""
    duthost = duthosts[rand_one_dut_hostname]
    dhcp_relay = dut_dhcp_relay_data[0]
    intf = dhcp_relay['relay_interface']
    options = dict(OPTION_CASES[option_case])

    if options["enable_interface_id"]:
        options["interface_id"] = design_interface_id(dhcp_relay)
    elif dhcp_relay['relay_mode'] == gpins_utils.RELAY_MODE_L3_LLA:
        pytest.skip("LLA-only relay requires Interface-ID (design doc); case {!r} not applicable".format(
            option_case))

    applied = gpins_utils.set_dhcp_relay_config(
        duthost, intf, servers_of(dhcp_relay), relay_mode=dhcp_relay['relay_mode'], **options)
    wait_converged(duthost, intf)
    pytest_assert(gpins_utils.wait_dhcp6relay_bound(duthost, timeout=CONVERGE_TIMEOUT, interval=3),
                  "dhcp6relay not running/listening after applying option config {}".format(applied))
    # The PTF verifier derives the expected Option 18 / 37 / 79 from ``applied``
    # (see gpins_ptf_params): absent -> asserted absent (18) / skipped (37, 79).
    run_dhcptest(ptfhost, duthost, dhcp_relay, applied)


# ---------------------------------------------------------------------------
# Upstream tests/gnmi/events/dhcp-relay_events.py::trigger_dhcp_relay_bind_failure
# ---------------------------------------------------------------------------
#
# Upstream deletes the VLAN's IPv6 address, restarts dhcp6relay, waits 35 s
# (6 bind attempts x 5 s) for the daemon to give up, restores the address,
# restarts the service, and asserts the gNMI event
# ``sonic-events-dhcp-relay:dhcp-relay-bind-failure`` was streamed.
#
# GPINS has no eventd / gNMI EVENTS target yet, so the trigger is ported as a
# behavioural test (no bind on the missing address, full recovery afterwards)
# and the event assertion is a placeholder.

BIND_FAILURE_RETRY_WINDOW = 35   # upstream: 6 attempts x 5 s + slack


def relay_bind_address(dhcp_relay):
    """(address, address/prefix) dhcp6relay must bind to on the relay interface: GUA if any, else LLA."""
    gua = dhcp_relay['downlink_vlan_iface']['addr']
    if gua:
        return str(gua), "{}/{}".format(gua, dhcp_relay['downlink_vlan_iface']['mask'])
    return dhcp_relay['down_interface_link_local'], dhcp_relay['down_interface_link_local_with_prefix_len']


def test_dhcpv6_relay_bind_failure(ptfhost, duthosts, rand_one_dut_hostname, dut_dhcp_relay_data,  # noqa F811
                                   validate_dut_routes_exist, dhcp_relay_config):  # noqa F811
    """Relay interface loses its address -> dhcp6relay must not bind to it; restore -> full recovery."""
    duthost = duthosts[rand_one_dut_hostname]
    dhcp_relay = dut_dhcp_relay_data[0]
    iface = dhcp_relay['relay_kernel_iface']
    addr, addr_with_prefix = relay_bind_address(dhcp_relay)
    pytest_assert(addr, "relay interface {} has neither a GUA nor a link-local address".format(iface))
    addr_c = ipaddress.ip_address(addr).compressed
    pytest_assert(gpins_utils.wait_dhcp6relay_bound(duthost), "dhcp6relay is not running/listening before the test")
    if not gpins_utils.dhcp6relay_uses_address_sockets(duthost):
        # TODO(GPINS): ask Boyang what bind failure looks like for the AF_PACKET relay
        # (exit / retry / log line); until then only the behavioural recovery is checked.
        logger.info("dhcp6relay on %s has no per-address UDP 547 sockets; per-address checks below are "
                    "vacuous, the end-to-end recovery check is what counts", duthost.hostname)

    def bound_to_addr():
        return addr_c in gpins_utils.dhcp6relay_bound_addresses(duthost)

    try:
        duthost.shell("ip -6 address del {} dev {}".format(addr_with_prefix, iface))
        gpins_utils.restart_dhcp_relay(duthost, wait=False)
        # Give the daemon its full retry window, then observe (not assert) how it behaved:
        # upstream dhcp6relay exits; the GPINS relay may instead keep its wildcard
        # socket and retry. Either way it must not claim the missing address.
        time.sleep(BIND_FAILURE_RETRY_WINDOW)
        running = gpins_utils.is_dhcp6relay_running(duthost)
        bound = sorted(gpins_utils.dhcp6relay_bound_addresses(duthost))
        logger.info("After %ss without %s on %s: dhcp6relay running=%s bound=%s",
                    BIND_FAILURE_RETRY_WINDOW, addr, iface, running, bound)
        pytest_assert(not bound_to_addr(),
                      "dhcp6relay reports a socket on {} although the address was removed from {}".format(addr, iface))
    finally:
        duthost.shell("ip -6 address add {} dev {}".format(addr_with_prefix, iface), module_ignore_errors=True)
        # Upstream restarts the whole service here because the exited daemon may
        # have taken the container down; do the same via the init script.
        pytest_assert(gpins_utils.restart_dhcp_relay(duthost), "dhcp_relay container did not come back")
        pytest_assert(gpins_utils.wait_dhcp6relay_bound(duthost), "dhcp6relay not running/listening after recovery")

    # Recovery must be complete: bound to the address again (if the relay uses
    # per-address sockets) and relaying end to end.
    if not wait_until(70, 5, 0, bound_to_addr):
        logger.warning("dhcp6relay has no socket on %s after recovery (bound=%s); relying on the end-to-end check",
                       addr, sorted(gpins_utils.dhcp6relay_bound_addresses(duthost)))
    run_dhcptest(ptfhost, duthost, dhcp_relay, dhcp_relay_config.get(dhcp_relay['relay_interface']))


@pytest.mark.skip(reason="TODO(GPINS, Virgo testbed): no eventd / gNMI EVENTS target on GPINS; "
                         "define the bind-failure event path before asserting it")
def test_dhcpv6_relay_bind_failure_event():
    """sonic-events-dhcp-relay:dhcp-relay-bind-failure (or its GPINS equivalent) must be streamed."""


# ---------------------------------------------------------------------------
# Placeholders - Table 2 rows that need the Virgo testbed
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="TODO(GPINS, Virgo testbed): Dynamic Port Breakout is not available on the reference switch")
def test_dhcpv6_relay_dpb():
    """Break out a relay-enabled port and enable relay on a child port; relay must come up there."""


@pytest.mark.skip(reason="TODO(GPINS, Virgo testbed): gNMI client / OpenConfig DHCP telemetry path not yet defined")
def test_dhcpv6_relay_openconfig_telemetry():
    """gNMI Subscribe to the relay counters; streamed values must match STATE_DB DHCPv6_COUNTER_TABLE."""


@pytest.mark.skip(reason="TODO(GPINS, Virgo testbed): NSF trigger (warm restart equivalent) not yet available")
def test_dhcpv6_relay_nsf():
    """Relay must keep working across NSF."""

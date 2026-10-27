"""GPINS-only DHCPv6 relay tests that fill the sonic-mgmt gaps listed in Table 2
of the DHCP-relay v6 test plan.

All run once per relay mode (l3_lla, l3_gua; vlan once GPINS supports VLANs) via the
``relay_mode`` parametrization from conftest.py:

* test_dhcpv6_relay_config_convergence    - CONFIG_DB DHCP_RELAY|<intf> shows up in
                                            STATE_DB, also for extra interfaces
* test_dhcpv6_relay_runtime_configuration - change the server list at runtime,
                                            STATE_DB follows, relay still works
* test_dhcpv6_relay_disabled              - no_config: remove all relay config at runtime and before
                                            container start -> no daemon / no socket;
                                            feature_disabled: FEATURE|dhcp_relay disabled -> dhcp6relay
                                            never starts although DHCP_RELAY is configured
* test_dhcpv6_relay_option_test           - Interface-ID / Remote-ID / RFC 6939 combinations are
                                            reflected in the Relay-Forward; L3+GUA without interface_id
                                            -> Option 18 is the port's GUA
* test_dhcpv6_relay_bind_failure          - relay interface loses its address -> no bind on it;
                                            restore -> full recovery (upstream
                                            tests/gnmi/events/dhcp-relay_events.py, item #11)
* test_dhcpv6_relay_openconfig_telemetry  - gNMI ON_CHANGE (options/state, no counters), SAMPLE / ONCE /
                                            Get of the openconfig-relay-agent paths: schema, options, counters
* test_dhcpv6_relay_openconfig_configuration - DHCP_RELAY: create / modify / read / delete the relay through
                                            OpenConfig gNMI; FEATURE: disable / enable FEATURE|dhcp_relay
                                            through gNMI; CONFIG_DB, STATE_DB and traffic follow
* test_dhcpv6_relay_nsf                   - the relay keeps working across a GPINS NSF warm reboot

Dynamic Port Breakout is tested in test_dhcpv6_relay_dpb_gpins.py.

Everything reuses the discovery / fixtures / PTF driver from test_dhcpv6_relay.py
so there is a single source of truth for how a relay target is picked and
exercised. Configuration goes straight into CONFIG_DB - no SONiC CLI.
"""
import ipaddress
import logging
import threading
import time


import pytest

from tests.common.fixtures.ptfhost_utils import copy_ptftests_directory   # noqa F401
from tests.common.fixtures.ptfhost_utils import change_mac_addresses      # noqa F401
from tests.common.fixtures.ptfhost_utils import copy_arp_responder_py     # noqa F401
from tests.common.helpers.assertions import pytest_assert
from tests.common.utilities import wait_until
from tests.dhcp_relay import gpins_dpb
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
    run_no_relay_test,
    check_interface_status,
)

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.topology('t0', 'm0', 'mx', 't0-2vlans'),
    pytest.mark.device_type('vs')
]

# How long dhcp6relay gets to mirror a CONFIG_DB change into STATE_DB.
CONVERGE_TIMEOUT = gpins_utils.DHCP_RELAY_CONVERGE_TIMEOUT
TEST_REMOTE_ID = "gpins-dhcp-relay-test"

# Test plan, test_dhcpv6_relay_disabled: "Verifies that the DHCP-relay daemon is not running in the DHCP-relay
# container" (the GPINS relay exits when it has no configuration; verified on ju3u1m1).
DISABLED_SETTLE_SECONDS = 15


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
        gpins_utils.assert_dhcp_relay_converged(duthost, intf)
        state = gpins_utils.get_dhcp_relay_state(duthost, intf)
        expected = gpins_utils.relay_servers(dhcp_relay)
        pytest_assert(gpins_utils.parse_dhcpv6_servers(state) == expected,
                      "STATE_DB servers for {} = {}, expected {}".format(intf, state, expected))
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
    servers = gpins_utils.relay_servers(dut_dhcp_relay_data[0])
    for target in extra[:1]:
        data = build_relay_data(duthost, tbinfo, relay_mode, target, servers, common)
        intf = data['relay_interface']
        if gpins_utils.get_dhcp_relay_config(duthost, intf):
            logger.info("%s already has a DHCP_RELAY entry outside the module baseline; skipping it", intf)
            continue
        gpins_utils.set_dhcp_relay_config(duthost, intf, servers, relay_mode=relay_mode,
                                          **gpins_utils.lla_relay_options(data))
        gpins_utils.assert_dhcp_relay_converged(duthost, intf)
        gpins_utils.del_dhcp_relay_config(duthost, intf)
        gpins_utils.assert_dhcp_relay_state_cleared(duthost, intf)
    # Baseline interfaces must be unaffected by the churn on the others.
    for dhcp_relay in dut_dhcp_relay_data:
        gpins_utils.assert_dhcp_relay_converged(duthost, dhcp_relay['relay_interface'])
    pytest_assert(check_interface_status(duthost), "dhcp6relay stopped listening during config churn")


RUNTIME_BURST_WAVES = 5
RUNTIME_BURST_OPS_PER_WAVE = 500
RUNTIME_BURST_ACTIVE_COUNT = 8
RUNTIME_BURST_WAVE_INTERVAL = 0.6


def _verify_counter_deltas(duthost, intf, before_counters, expected_deltas, timeout=5.0):
    """Poll STATE_DB DHCPv6_COUNTER_TABLE|<intf> until all expected counter deltas are observed."""
    deadline = time.time() + timeout
    mismatches = {}
    after_counters = {}
    while time.time() < deadline:
        raw = gpins_utils.get_dhcpv6_counters(duthost, intf)
        after_counters = {k: int(v) for k, v in raw.items() if str(v).lstrip("-").isdigit()}
        mismatches = {}
        for field, exp_delta in expected_deltas.items():
            base_val = before_counters.get(field, 0)
            act_val = after_counters.get(field, 0)
            act_delta = act_val - base_val
            if act_delta < exp_delta:
                mismatches[field] = (exp_delta, act_delta, base_val, act_val)
        if not mismatches:
            logger.info("STATE_DB DHCPv6_COUNTER_TABLE|%s deltas verified: %s",
                        intf, {f: after_counters.get(f, 0) - before_counters.get(f, 0) for f in expected_deltas})
            return
        time.sleep(0.25)

    details = ", ".join(
        "{}: delta={} (expected >={}, before={}, after={})".format(k, act_d, exp_d, b_val, a_val)
        for k, (exp_d, act_d, b_val, a_val) in mismatches.items()
    )
    pytest_assert(False, "STATE_DB DHCPv6_COUNTER_TABLE|{} counter mismatch: {}".format(intf, details))


def test_dhcpv6_relay_runtime_configuration(ptfhost, duthosts, rand_one_dut_hostname, dut_dhcp_relay_data,  # noqa F811
                                            validate_dut_routes_exist, dhcp_relay_config,  # noqa F811
                                            dhcp_relay_config_snapshot):
    """Multi-wave runtime CONFIG_DB burst churn + STATE_DB & dataplane verification per wave.

    For each wave (1..RUNTIME_BURST_WAVES):
      1. Execute a pipelined burst of ``RUNTIME_BURST_OPS_PER_WAVE`` random
         ``HSET`` / ``DEL`` operations on ``CONFIG_DB`` ``DHCP_RELAY|*`` across
         all candidate LLA-capable ports and converge to a deterministic active
         port set (always including the primary test interface).
      2. Verify ``STATE_DB`` ``DHCP_RELAY|*`` matches ``expected_configs``, all
         deleted interfaces are cleaned from ``STATE_DB``, and ``dhcp6relay``
         PID is stable.
      3. Send end-to-end DHCPv6 relay packets via PTF (verifying the wave's
         updated Option 18 / Option 37 in ``Relay-Forward``) and verify
         ``STATE_DB`` ``DHCPv6_COUNTER_TABLE|<intf>`` increments.
      4. Re-verify ``STATE_DB`` ``DHCP_RELAY|*`` remains intact after traffic.
    """
    duthost = duthosts[rand_one_dut_hostname]
    dhcp_relay = dut_dhcp_relay_data[0]
    intf = dhcp_relay['relay_interface']
    mode = dhcp_relay['relay_mode']
    servers = gpins_utils.relay_servers(dhcp_relay)
    primary_iid = gpins_utils.design_interface_id(dhcp_relay)

    gpins_utils.verify_interface_mode(duthost, intf, mode)
    gpins_utils.set_feature_state(duthost, "dhcp_relay", "enabled")
    pytest_assert(gpins_utils.feature_state(duthost, "dhcp_relay") == "enabled",
                  "FEATURE|dhcp_relay state was not set to enabled")

    candidates = gpins_utils.discover_candidate_interfaces(
        duthost,
        primary_intf=intf,
        relay_mode=mode,
        exclude_intfs=dhcp_relay.get('uplink_interfaces', []),
    )
    existing_keys = gpins_utils.redis_keys(duthost, "CONFIG_DB", "DHCP_RELAY|*")
    if existing_keys:
        gpins_utils.redis_pipeline(duthost, "CONFIG_DB", [["DEL", k] for k in existing_keys])

    expected_deltas = {
        "Solicit": 1,
        "Advertise": 1,
        "Request": 1,
        "Reply": 1,
        "Relay-Forward": 2 * len(servers),
        "Relay-Reply": 2,
    }

    for w in range(1, RUNTIME_BURST_WAVES + 1):
        expected_configs, deleted_intfs = gpins_utils.run_single_wave_db_burst(
            duthost=duthost,
            wave_idx=w,
            total_waves=RUNTIME_BURST_WAVES,
            candidate_intfs=candidates,
            primary_db_intf=intf,
            dhcpv6_servers=servers,
            ops_per_wave=RUNTIME_BURST_OPS_PER_WAVE,
            final_active_count=RUNTIME_BURST_ACTIVE_COUNT,
            primary_interface_id=primary_iid,
        )

        # Verify dhcp6relay daemon stability and STATE_DB Applied Configuration for this wave.
        gpins_utils.wait_and_verify_state_db(
            duthost=duthost,
            expected_configs=expected_configs,
            deleted_intfs=deleted_intfs,
            timeout_sec=CONVERGE_TIMEOUT,
        )
        pytest_assert(gpins_utils.wait_dhcp6relay_bound(duthost, timeout=CONVERGE_TIMEOUT, interval=2),
                      "[Wave {}/{}] dhcp6relay not running/listening after burst churn".format(
                          w, RUNTIME_BURST_WAVES))

        # Snapshot STATE_DB counters before sending PTF packets for this wave.
        raw_before = gpins_utils.get_dhcpv6_counters(duthost, intf)
        before_counters = {k: int(v) for k, v in raw_before.items() if str(v).lstrip("-").isdigit()}

        # Verify end-to-end packet relaying (including the wave's updated remote_id_w<w>_001) and STATE_DB counters.
        run_dhcptest(ptfhost, duthost, dhcp_relay, expected_configs[intf])
        _verify_counter_deltas(duthost, intf, before_counters, expected_deltas)

        # Re-verify STATE_DB Applied Configuration remained intact after traffic.
        gpins_utils.wait_and_verify_state_db(
            duthost=duthost,
            expected_configs=expected_configs,
            deleted_intfs=deleted_intfs,
            timeout_sec=10.0,
        )
        logger.info("[Wave %d/%d] PASS: DB burst churn, STATE_DB convergence, PTF relay & counter check succeeded",
                    w, RUNTIME_BURST_WAVES)
        if w < RUNTIME_BURST_WAVES:
            time.sleep(RUNTIME_BURST_WAVE_INTERVAL)


DISABLED_SCENARIOS = ("no_config", "feature_disabled")


def _relay_inactive(duthost):
    return not gpins_utils.is_dhcp6relay_bound(duthost) and not gpins_utils.is_dhcp6relay_running(duthost)


def _describe_relay(duthost):
    return "dhcp6relay running={} bound={}, {} running={}".format(
        gpins_utils.is_dhcp6relay_running(duthost), gpins_utils.is_dhcp6relay_bound(duthost),
        gpins_utils.DHCP_RELAY_CONTAINER, gpins_utils.is_container_running(duthost))


@pytest.mark.parametrize("scenario", DISABLED_SCENARIOS)
def test_dhcpv6_relay_disabled(ptfhost, duthosts, rand_one_dut_hostname, dut_dhcp_relay_data,  # noqa F811
                               dhcp_relay_config, dhcp_relay_config_snapshot, scenario):  # noqa F811
    """No relay may run when it is disabled.

    * ``no_config``: delete all relay config (a) at runtime and (b) before the container starts.
    * ``feature_disabled``: with valid DHCP_RELAY config in place, ``FEATURE|dhcp_relay state=disabled``
      must keep dhcp6relay from ever starting in a (re)started relay container; re-enabling starts it.
    """
    duthost = duthosts[rand_one_dut_hostname]

    if scenario == "feature_disabled":
        _check_feature_disabled(
            ptfhost, duthost, dut_dhcp_relay_data[0], "CONFIG_DB",
            lambda state: gpins_utils.set_feature_state(duthost, gpins_utils.DHCP_RELAY_FEATURE, state))
        return

    keys = gpins_utils.redis_keys(duthost, "CONFIG_DB", "DHCP_RELAY|*")
    pytest_assert(keys, "expected at least one DHCP_RELAY entry to remove")

    # (a) runtime removal
    gpins_utils.redis_del(duthost, "CONFIG_DB", *keys)
    for key in keys:
        gpins_utils.assert_dhcp_relay_state_cleared(duthost, key.split("|", 1)[1])
    pytest_assert(wait_until(CONVERGE_TIMEOUT, 3, 0, _relay_inactive, duthost),
                  "relay still active {}s after removing all DHCP_RELAY config: {}".format(
                      CONVERGE_TIMEOUT, _describe_relay(duthost)))

    # (b) container (re)start with no relay config at all
    gpins_utils.restart_dhcp_relay(duthost, wait=False)
    time.sleep(DISABLED_SETTLE_SECONDS)
    pytest_assert(_relay_inactive(duthost),
                  "relay became active after a container start with no DHCP_RELAY config: {}".format(
                      _describe_relay(duthost)))
    # It must also *stay* inactive (no delayed bind / retry loop).
    time.sleep(DISABLED_SETTLE_SECONDS)
    pytest_assert(_relay_inactive(duthost),
                  "relay activated late with no DHCP_RELAY config: {}".format(_describe_relay(duthost)))
    # Teardown (dhcp_relay_config_snapshot) restores the config and asserts the relay binds again.


def _check_feature_disabled(ptfhost, duthost, dhcp_relay, writer, set_feature):
    """``FEATURE|dhcp_relay state=disabled`` keeps dhcp6relay from starting although DHCP_RELAY is configured.

    ``set_feature(state)`` writes the flag (``writer`` names how, for messages: CONFIG_DB or gNMI). Re-enabling
    the flag must bring the relay back with the same config.
    """
    intf = dhcp_relay['relay_interface']
    feature = gpins_utils.DHCP_RELAY_FEATURE
    # 1. The config itself is good: the relay is up and bound with it while the feature is enabled.
    pytest_assert(gpins_utils.feature_state(duthost, feature) == "enabled",
                  "FEATURE|{} must start out enabled (session fixture)".format(feature))
    pytest_assert(gpins_utils.get_dhcp_relay_config(duthost, intf), "DHCP_RELAY|{} is not configured".format(intf))
    pytest_assert(gpins_utils.wait_dhcp6relay_bound(duthost, timeout=CONVERGE_TIMEOUT, interval=3),
                  "dhcp6relay not running/listening with DHCP_RELAY|{} before disabling the feature".format(intf))
    passed = False
    try:
        # 2. Disable the feature, then (re)start the relay container with the DHCP_RELAY config still present.
        set_feature("disabled")
        gpins_utils.restart_dhcp_relay(duthost)
        # 3. dhcp6relay never boots up, also not after a delay. The relay manager in the container reads the flag at
        #    start (and on every change) and keeps dhcp6relay stopped; the container itself may or may not stay up
        #    (it flapped on ju3u1m1 2026-10-26 around back-to-back init-script restarts), which is not under test.
        for window in (1, 2):
            time.sleep(DISABLED_SETTLE_SECONDS)
            pytest_assert(_relay_inactive(duthost),
                          "dhcp6relay started with FEATURE|{} disabled through {} (after {}s): {}".format(
                              feature, writer, window * DISABLED_SETTLE_SECONDS, _describe_relay(duthost)))
            logger.info("FEATURE|%s disabled through %s, after %ss: %s", feature, writer,
                        window * DISABLED_SETTLE_SECONDS, _describe_relay(duthost))
        # 4. A client Solicit is not relayed.
        run_no_relay_test(ptfhost, duthost, dhcp_relay)
        passed = True
    finally:
        reenable_error = _reenable_feature(duthost, writer, set_feature)
        if not passed:
            _recover_relay_quietly(duthost)
    # 5. Re-enabling the feature (through the same writer) brings the relay up with the same config.
    pytest_assert(reenable_error is None,
                  "Re-enabling FEATURE|{} through {} failed: {}".format(feature, writer, reenable_error))
    _wait_relay_back(duthost, intf, "FEATURE|{} was re-enabled through {}".format(feature, writer))


def _reenable_feature(duthost, writer, set_feature):
    """Set FEATURE|dhcp_relay back to enabled with ``set_feature``, falling back to CONFIG_DB if that fails.

    Returns the error ``set_feature`` raised, or None.
    """
    feature = gpins_utils.DHCP_RELAY_FEATURE
    error = None
    try:
        set_feature("enabled")
    except (Exception, pytest.fail.Exception) as exc:  # never leave the feature disabled behind
        error = exc
        logger.error("Re-enabling FEATURE|%s through %s failed (%s); writing CONFIG_DB directly", feature, writer, exc)
    if gpins_utils.feature_state(duthost, feature) != "enabled":
        gpins_utils.set_feature_state(duthost, feature, "enabled")
    return error


def _restart_relay_container(duthost):
    """Restart the relay container, or start it if it is down."""
    if gpins_utils.is_container_running(duthost):
        gpins_utils.restart_dhcp_relay(duthost)
    else:
        gpins_utils.start_dhcp_relay(duthost)


def _recover_relay_quietly(duthost):
    """Best effort after a failed test: get dhcp6relay running again so later tests start clean. Never raises."""
    try:
        if not gpins_utils.wait_dhcp6relay_bound(duthost, timeout=CONVERGE_TIMEOUT, interval=3):
            _restart_relay_container(duthost)
            gpins_utils.wait_dhcp6relay_bound(duthost, timeout=CONVERGE_TIMEOUT, interval=3)
    except Exception as exc:  # never mask the test's own failure
        logger.error("Recovering dhcp6relay on %s failed: %s", duthost.hostname, exc)


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
    # enable_interface_id=true without interface_id: on an L3+GUA port the relay must fill Option 18 with the
    # port's GUA (Boyang, 2026-10-26 review).
    "interface_id_default": {"enable_interface_id": True, "enable_remote_id": False, "rfc6939_support": True},
}
# Cases that leave interface_id unset on purpose.
DEFAULT_INTERFACE_ID_CASES = frozenset({"interface_id_default"})


@pytest.mark.parametrize("option_case", sorted(OPTION_CASES))
def test_dhcpv6_relay_option_test(ptfhost, duthosts, rand_one_dut_hostname, dut_dhcp_relay_data,  # noqa F811
                                  validate_dut_routes_exist, dhcp_relay_config, dhcp_relay_config_snapshot,  # noqa F811
                                  option_case):
    """Configure Interface-ID / Remote-ID / RFC6939 combinations and check the Relay-Forward carries those."""
    duthost = duthosts[rand_one_dut_hostname]
    dhcp_relay = dut_dhcp_relay_data[0]
    intf = dhcp_relay['relay_interface']
    options = dict(OPTION_CASES[option_case])
    is_lla = dhcp_relay['relay_mode'] == gpins_utils.RELAY_MODE_L3_LLA

    if option_case in DEFAULT_INTERFACE_ID_CASES:
        if is_lla:
            pytest.skip("LLA-only relay has no GUA to default Option 18 to and requires an explicit Interface-ID "
                        "(design doc); case {!r} not applicable".format(option_case))
    elif options["enable_interface_id"]:
        options["interface_id"] = gpins_utils.design_interface_id(dhcp_relay)
    elif is_lla:
        pytest.skip("LLA-only relay requires Interface-ID (design doc); case {!r} not applicable".format(
            option_case))

    applied = gpins_utils.set_dhcp_relay_config(
        duthost, intf, gpins_utils.relay_servers(dhcp_relay), relay_mode=dhcp_relay['relay_mode'], **options)
    gpins_utils.assert_dhcp_relay_converged(duthost, intf)
    pytest_assert(gpins_utils.wait_dhcp6relay_bound(duthost, timeout=CONVERGE_TIMEOUT, interval=3),
                  "dhcp6relay not running/listening after applying option config {}".format(applied))
    # The PTF verifier derives the expected Option 18 / 37 / 79 from ``applied``
    # (see gpins_ptf_params): absent -> asserted absent (18) / skipped (37, 79); enable_interface_id
    # without interface_id -> the relay interface's GUA.
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


def _assert_openconfig_config_and_state(iface_node, intf, expected_servers, expected_iid, expected_remote_id):
    """Assert that OpenConfig config, state (with counters), and options/config + options/state match."""
    gpins_utils.validate_openconfig_dhcpv6_relay_schema(iface_node, intf)
    norm = gpins_utils.normalize_openconfig_dict(iface_node)
    # The schema check only validates counters that are present; SAMPLE / ONCE / Get must actually carry them.
    pytest_assert("counters" in norm.get("state", {}),
                  "OpenConfig state/counters missing from the interface payload: {}".format(norm.get("state")))

    for sec_name in ("config", "state"):
        sec = norm.get(sec_name, {})
        pytest_assert(sec.get("id") == intf,
                      "OpenConfig {}.id = {!r}, expected {!r}".format(sec_name, sec.get("id"), intf))
        pytest_assert(set(sec.get("helper-address", [])) == set(expected_servers),
                      "OpenConfig {}.helper-address = {}, expected {}".format(
                          sec_name, sec.get("helper-address"), expected_servers))

    opts = norm.get("options", {})
    for opt_sec_name in ("config", "state"):
        opt_sec = opts.get(opt_sec_name, {})
        pytest_assert(gpins_utils.openconfig_client_mac(opt_sec) is True,
                      "OpenConfig options.{} client-MAC ({}) = {!r}, expected True".format(
                          opt_sec_name, "/".join(gpins_utils.OPENCONFIG_CLIENT_MAC_LEAVES),
                          gpins_utils.openconfig_client_mac(opt_sec)))
        pytest_assert(opt_sec.get("enable-interface-id") is True,
                      "OpenConfig options.{}.enable-interface-id = {!r}, expected True".format(
                          opt_sec_name, opt_sec.get("enable-interface-id")))
        pytest_assert(opt_sec.get("interface-id") == expected_iid,
                      "OpenConfig options.{}.interface-id = {!r}, expected {!r}".format(
                          opt_sec_name, opt_sec.get("interface-id"), expected_iid))
        pytest_assert(opt_sec.get("enable-remote-id") is True,
                      "OpenConfig options.{}.enable-remote-id = {!r}, expected True".format(
                          opt_sec_name, opt_sec.get("enable-remote-id")))
        pytest_assert(opt_sec.get("remote-id") == expected_remote_id,
                      "OpenConfig options.{}.remote-id = {!r}, expected {!r}".format(
                          opt_sec_name, opt_sec.get("remote-id"), expected_remote_id))


# ON_CHANGE stream: how long it runs and when the CONFIG_DB change is made inside it.
ON_CHANGE_STREAM_SECONDS = 30
ON_CHANGE_CHANGE_AFTER_SECONDS = 10


def _leaf_values(obj, leaf):
    """All values of ``leaf`` anywhere in a normalized OpenConfig JSON tree."""
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == leaf and not isinstance(value, (dict, list)):
                found.append(value)
            else:
                found.extend(_leaf_values(value, leaf))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_leaf_values(item, leaf))
    return found


def _check_on_change(duthost, intf, oc_iface_path, remote_id):
    """STREAM/ON_CHANGE on options/state: initial sync carries ``remote_id``, a runtime change is streamed."""
    path = "{}/options/state".format(oc_iface_path)
    key = "DHCP_RELAY|{}".format(intf)
    changed_remote_id = "{}_changed".format(remote_id)
    result = {}

    def _subscribe():
        try:
            result["updates"] = gpins_utils.gnmi_subscribe_json(
                duthost, path, query_type="s", streaming_type="ON_CHANGE",
                streaming_duration="{}s".format(ON_CHANGE_STREAM_SECONDS))
        except Exception as exc:  # reported from the main thread
            result["error"] = exc

    subscriber = threading.Thread(target=_subscribe, daemon=True)
    subscriber.start()
    try:
        time.sleep(ON_CHANGE_CHANGE_AFTER_SECONDS)
        gpins_utils.redis_hset(duthost, "CONFIG_DB", key, {"remote_id": changed_remote_id})
        subscriber.join(ON_CHANGE_STREAM_SECONDS + 60)
    finally:
        gpins_utils.redis_hset(duthost, "CONFIG_DB", key, {"remote_id": remote_id})
        gpins_utils.assert_dhcp_relay_converged(duthost, intf)

    pytest_assert(not subscriber.is_alive(), "ON_CHANGE subscription on {} did not end".format(path))
    pytest_assert("error" not in result, "ON_CHANGE subscription on {} failed: {}".format(path, result.get("error")))
    updates = [gpins_utils.normalize_openconfig_dict(u) for u in result["updates"]]
    logger.info("ON_CHANGE updates on %s: %s", path, updates)
    pytest_assert(not any(_leaf_values(u, "dhcpv6-solicit-received") for u in updates),
                  "ON_CHANGE on {} streamed counters: {}".format(path, updates))
    remote_ids = [v for u in updates for v in _leaf_values(u, "remote-id")]
    pytest_assert(remote_ids and remote_ids[0] == remote_id,
                  "ON_CHANGE initial sync on {}: remote-id {!r}, expected {!r} first".format(
                      path, remote_ids, remote_id))
    pytest_assert(changed_remote_id in remote_ids,
                  "ON_CHANGE on {} did not stream the remote-id change to {!r} within {}s (seen: {})".format(
                      path, changed_remote_id, ON_CHANGE_STREAM_SECONDS - ON_CHANGE_CHANGE_AFTER_SECONDS,
                      remote_ids))

def test_dhcpv6_relay_openconfig_telemetry(ptfhost, duthosts, rand_one_dut_hostname, dut_dhcp_relay_data,  # noqa F811
                                           validate_dut_routes_exist, dhcp_relay_config,  # noqa F811
                                           dhcp_relay_config_snapshot):
    """Verify gNMI Subscribe & OpenConfig YANG schema conformance.

    STREAM ON_CHANGE on options/state (a runtime remote-id change must be streamed), STREAM SAMPLE / ONCE / Get
    on the whole interface (config, state with counters, options). Counters are never subscribed ON_CHANGE.
    """
    duthost = duthosts[rand_one_dut_hostname]
    dhcp_relay = dut_dhcp_relay_data[0]
    intf = dhcp_relay['relay_interface']
    mode = dhcp_relay['relay_mode']
    servers = gpins_utils.relay_servers(dhcp_relay)
    expected_iid = gpins_utils.design_interface_id(dhcp_relay)
    expected_remote_id = "telemetry_remote_id_{}".format(gpins_utils.to_kernel_ifname(intf))

    # 1. Remove DHCP_RELAY|<intf> first and verify the interface type & address state
    #    (e.g. in l3_lla mode, the interface must have an LLA and NO GUA) before enabling DHCP relay.
    gpins_utils.del_dhcp_relay_config(duthost, intf)
    gpins_utils.assert_dhcp_relay_state_cleared(duthost, intf)
    gpins_utils.verify_interface_mode(duthost, intf, mode)

    # 2. Ensure telemetry-con and the gNMI server on :9339 are running (no CONFIG_DB change).
    gpins_utils.ensure_telemetry_running(duthost)

    # 3. Enable DHCPv6 relay on the verified interface with Interface-ID, Remote-ID, and RFC 6939 enabled.
    applied = gpins_utils.set_dhcp_relay_config(
        duthost,
        intf,
        servers,
        relay_mode=mode,
        enable_interface_id=True,
        interface_id=expected_iid,
        enable_remote_id=True,
        remote_id=expected_remote_id,
        rfc6939_support=True,
    )
    gpins_utils.assert_dhcp_relay_converged(duthost, intf)
    pytest_assert(gpins_utils.wait_dhcp6relay_bound(duthost, timeout=CONVERGE_TIMEOUT, interval=3),
                  "dhcp6relay not running/listening after configuring {}".format(intf))

    oc_iface_path = "/openconfig-relay-agent:relay-agent/dhcpv6/interfaces/interface[id={}]".format(intf)
    oc_counters_path = "{}/state/counters".format(oc_iface_path)

    # Steps 4-6 talk to the gNMI server from inside telemetry-con. The lab server requires client certificates
    # (TELEMETRY|certs ca_crt) that no local client can present, so for the duration of these steps the server
    # runs TLS-without-client-auth and without authz/pathz policies (what the old ensure_telemetry_running()
    # left behind for good); the CONFIG_DB rows are restored and telemetry-con restarted afterwards.
    with gpins_utils.gnmi_cli_session(duthost, "server-tls+no-policies"):
        models = gpins_utils.gnmi_supported_models(duthost, "server-tls")
        if models and gpins_utils.OC_RELAY_AGENT_MODEL not in models:
            pytest.skip("The gNMI server on {} ({}) does not advertise {}: this image has no DHCPv6-relay OpenConfig "
                        "support, so there is nothing to subscribe to ({} models advertised).".format(
                            duthost.hostname,
                            gpins_utils.redis_hget(duthost, "CONFIG_DB", "DEVICE_METADATA|localhost", "build_version"),
                            gpins_utils.OC_RELAY_AGENT_MODEL, len(models)))

        # 4. Pre-traffic gNMI Subscribe + strict OpenConfig YANG schema conformance validation.
        #    STREAM/ON_CHANGE only on options/state: state/counters change too often for on-change and the
        #    server does not support it there (Boyang, 2026-10-26). The ON_CHANGE stream must deliver a
        #    runtime remote-id change. Counters are read with SAMPLE / ONCE / Get only.
        _check_on_change(duthost, intf, oc_iface_path, expected_remote_id)

        sample_updates = gpins_utils.gnmi_subscribe_json(
            duthost, oc_iface_path, query_type="s", streaming_type="SAMPLE",
            sample_interval=1, streaming_duration="2s")
        sample_node = gpins_utils.extract_openconfig_interface_node(sample_updates[-1], intf)
        _assert_openconfig_config_and_state(sample_node, intf, servers, expected_iid, expected_remote_id)

        once_updates = gpins_utils.gnmi_subscribe_json(duthost, oc_iface_path, query_type="once")
        once_node = gpins_utils.extract_openconfig_interface_node(once_updates[-1], intf)
        _assert_openconfig_config_and_state(once_node, intf, servers, expected_iid, expected_remote_id)

        before_oc_counters = gpins_utils.validate_openconfig_dhcpv6_counters_schema(
            sample_node["state"]["counters"])

        # 5. Run end-to-end 4-way DHCPv6 relay packet test via PTF (Solicit, Advertise, Request, Reply).
        run_dhcptest(ptfhost, duthost, dhcp_relay, applied)

        # 6. Post-traffic gNMI Subscribe (STREAM/SAMPLE) + gnmi_get verification:
        #    verify strict YANG schema conformance, expected packet counter deltas,
        #    and 1-for-1 match between all 15 OpenConfig counter leaves and STATE_DB DHCPv6_COUNTER_TABLE.
        expected_oc_deltas = {
            "dhcpv6-solicit-received": 1,
            "dhcpv6-adverstise-sent": 1,
            "dhcpv6-request-received": 1,
            "dhcpv6-reply-sent": 1,
            "dhcpv6-relay-forw-sent": 2 * len(servers),
            "dhcpv6-relay-reply-received": 2,
        }
        post_updates = gpins_utils.gnmi_subscribe_json(
            duthost, oc_counters_path, query_type="s", streaming_type="SAMPLE",
            sample_interval=1, streaming_duration="2s")
        post_node = gpins_utils.extract_openconfig_interface_node(post_updates[-1], intf)
        gpins_utils.validate_openconfig_dhcpv6_relay_schema(post_node, intf)
        after_oc_counters = gpins_utils.validate_openconfig_dhcpv6_counters_schema(
            post_node["state"]["counters"])

        get_payload = gpins_utils.gnmi_get_json(duthost, oc_iface_path)
        get_node = gpins_utils.extract_openconfig_interface_node(get_payload, intf)
        _assert_openconfig_config_and_state(get_node, intf, servers, expected_iid, expected_remote_id)

    for oc_leaf, exp_delta in expected_oc_deltas.items():
        act_delta = after_oc_counters[oc_leaf] - before_oc_counters.get(oc_leaf, 0)
        pytest_assert(act_delta >= exp_delta,
                      "OpenConfig counter {} delta={} (expected >={}, before={}, after={})".format(
                          oc_leaf, act_delta, exp_delta,
                          before_oc_counters.get(oc_leaf, 0), after_oc_counters[oc_leaf]))

    state_db_counters = gpins_utils.get_dhcpv6_counters(duthost, intf)
    for oc_leaf, state_db_field in gpins_utils.OPENCONFIG_DHCPV6_COUNTER_TO_STATE_DB.items():
        oc_val = after_oc_counters[oc_leaf]
        db_val = int(state_db_counters.get(state_db_field, "0"))
        pytest_assert(oc_val == db_val,
                      "OpenConfig counter {}={} does not match STATE_DB DHCPv6_COUNTER_TABLE|{} {}={}".format(
                          oc_leaf, oc_val, intf, state_db_field, db_val))


# ---------------------------------------------------------------------------
# Configuration through gNMI (FEATURE and DHCP_RELAY tables)
# ---------------------------------------------------------------------------

GNMI_ACCESS = "server-tls+no-policies"
GNMI_CONFIG_TIMEOUT = 30
# FEATURE|<feature> state through gNMI: SONiC YANG via translib, no origin/target (works on ju3u1m1, 2026-10-26;
# the sonic-gnmi native "sonic-db:CONFIG_DB" write is rejected with "Invalid target").
FEATURE_GNMI_PATH = "/sonic-feature:sonic-feature/FEATURE/FEATURE_LIST[name={feature}]/state"
# CONFIG_DB rows the gNMI server itself writes on every Set (UMF Set counters); the framework's post-module
# config check flags them, so the test removes / restores them.
GNMI_SIDE_EFFECT_PATTERNS = ("UMF_STATS|*",)


def _gnmi_set_ok(duthost, operation, path, payload, prefix=gpins_dpb.GNMI_PREFIX):
    result = gpins_dpb.gnmi_set_path(duthost, operation, path, payload, access=GNMI_ACCESS, prefix=prefix)
    pytest_assert(result["rc"] == 0, "gNMI {} {} failed: rc={} {}".format(
        operation, path, result["rc"], result["stderr"] or result["stdout"]))
    return result


def _oc_name(key):
    """OpenConfig node name without its module prefix (``openconfig-relay-agent:state`` -> ``state``)."""
    return str(key).split(":", 1)[-1]


def _strip_config_false(obj):
    """Copy of an OpenConfig JSON tree without ``state`` containers (config false; a Set must not carry them)."""
    if isinstance(obj, dict):
        return {k: _strip_config_false(v) for k, v in obj.items() if _oc_name(k) != "state"}
    if isinstance(obj, list):
        return [_strip_config_false(item) for item in obj]
    return obj


def _replace_leaves(obj, values):
    """Copy of an OpenConfig JSON tree with every leaf named in ``values`` (module prefix ignored) replaced."""
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            name = _oc_name(key)
            out[key] = values[name] if name in values else _replace_leaves(value, values)
        return out
    if isinstance(obj, list):
        return [_replace_leaves(item, values) for item in obj]
    return obj


def _expected_dhcp_relay_fields(servers, interface_id, remote_id):
    return {"dhcpv6_servers": ",".join(servers), "enable_interface_id": "true", "interface_id": interface_id,
            "enable_remote_id": "true", "remote_id": remote_id, "rfc6939_support": "true"}


def _wait_config_db(duthost, key, expected, what):
    """Wait until CONFIG_DB ``key`` has ``expected`` (``None`` = key absent); server lists compared as sets."""
    def _norm(fields):
        if fields and "dhcpv6_servers" in fields:
            fields = dict(fields, dhcpv6_servers=sorted(fields["dhcpv6_servers"].split(",")))
        return fields

    state = {}

    def _matches():
        state["now"] = gpins_utils.redis_hgetall(duthost, "CONFIG_DB", key)
        if expected is None:
            return not state["now"]
        now = _norm(state["now"])
        return all(now.get(field) == value for field, value in _norm(expected).items())

    pytest_assert(wait_until(GNMI_CONFIG_TIMEOUT, 2, 0, _matches),
                  "{}: CONFIG_DB {} = {}, expected {}".format(what, key, state.get("now"), expected or "absent"))
    return state["now"]


@pytest.fixture
def persisted_config_guard(duthosts, rand_one_dut_hostname):
    """save_on_set=true makes every accepted gNMI Set rewrite config_db.json: back it up, put it back after.
    Also undoes the UMF_STATS rows the server writes into CONFIG_DB on each Set."""
    duthost = duthosts[rand_one_dut_hostname]
    md5 = gpins_dpb.backup_persisted_config(duthost)
    side_effects = gpins_utils.snapshot_config_db(duthost, list(GNMI_SIDE_EFFECT_PATTERNS))
    yield
    gpins_utils.restore_config_db(duthost, side_effects)
    gpins_dpb.restore_persisted_config_if_changed(duthost, md5)
    # A config_db.json left with the test's gNMI changes would survive the next reboot.
    pytest_assert(gpins_dpb.persisted_config_md5(duthost) == md5,
                  "{} on {} could not be restored from {} (md5 {} expected)".format(
                      gpins_dpb.PERSISTED_CONFIG_DB, duthost.hostname, gpins_dpb.PERSISTED_CONFIG_DB_BACKUP, md5))


@pytest.mark.parametrize("table", ["DHCP_RELAY", "FEATURE"])
def test_dhcpv6_relay_openconfig_configuration(ptfhost, duthosts, rand_one_dut_hostname,
                                               dut_dhcp_relay_data, validate_dut_routes_exist,  # noqa F811
                                               dhcp_relay_config, dhcp_relay_config_snapshot,  # noqa F811
                                               persisted_config_guard, table):
    """Configure the relay through gNMI and verify CONFIG_DB, STATE_DB and the relay's behaviour.

    * ``DHCP_RELAY``: create / modify / read back / delete ``relay-agent/dhcpv6/interfaces/interface[id=X]`` through
      OpenConfig; CONFIG_DB ``DHCP_RELAY|X`` and STATE_DB follow and the relay forwards with the configured options.
    * ``FEATURE``: disable / enable ``FEATURE|dhcp_relay`` through gNMI; dhcp6relay stops starting / starts again.
    """
    duthost = duthosts[rand_one_dut_hostname]
    dhcp_relay = dut_dhcp_relay_data[0]
    with gpins_utils.gnmi_cli_session(duthost, GNMI_ACCESS):
        if table == "FEATURE":
            _check_feature_disabled(ptfhost, duthost, dhcp_relay, "gNMI",
                                    lambda state: _gnmi_set_feature(duthost, state))
        else:
            _check_oc_relay_config(ptfhost, duthost, dhcp_relay)


def _check_oc_relay_config(ptfhost, duthost, dhcp_relay):
    intf = dhcp_relay['relay_interface']
    mode = dhcp_relay['relay_mode']
    key = "DHCP_RELAY|{}".format(intf)
    servers = gpins_utils.relay_servers(dhcp_relay)
    iid = gpins_utils.design_interface_id(dhcp_relay)
    kernel_if = gpins_utils.to_kernel_ifname(intf)
    seed_remote_id, set_remote_id, new_remote_id = ("oc_cfg_{}_{}".format(step, kernel_if)
                                                    for step in ("seed", "set", "update"))
    oc_iface_path = "/openconfig-relay-agent:relay-agent/dhcpv6/interfaces/interface[id={}]".format(intf)
    # Set paths carry no module prefix: with origin "openconfig" the server prepends "openconfig-" itself
    # (a prefixed path was rejected: "Invalid yang module prefix ... openconfig-openconfig-relay-agent", 2026-10-26).
    oc_set_path = "/relay-agent/dhcpv6/interfaces/interface[id={}]".format(intf)

    models = gpins_utils.gnmi_supported_models(duthost, "server-tls")
    if models and gpins_utils.OC_RELAY_AGENT_MODEL not in models:
        pytest.skip("The gNMI server on {} does not advertise {}: no DHCPv6-relay OpenConfig support to configure "
                    "through".format(duthost.hostname, gpins_utils.OC_RELAY_AGENT_MODEL))

    # 1. Learn the server's own encoding of the interface subtree (module prefixes of the augments included):
    #    seed it through CONFIG_DB, Get it, keep only the config nodes.
    gpins_utils.set_dhcp_relay_config(duthost, intf, servers, relay_mode=mode, enable_interface_id=True,
                                      interface_id=iid, enable_remote_id=True, remote_id=seed_remote_id,
                                      rfc6939_support=True)
    gpins_utils.assert_dhcp_relay_converged(duthost, intf)
    template = _strip_config_false(gpins_utils.gnmi_get_json(duthost, oc_iface_path))
    logger.info("OpenConfig config template for %s: %s", intf, template)

    # 2. Remove the relay from CONFIG_DB, then create it again through gNMI with a different remote-id.
    gpins_utils.del_dhcp_relay_config(duthost, intf)
    gpins_utils.assert_dhcp_relay_state_cleared(duthost, intf)
    _gnmi_set_ok(duthost, "update", oc_set_path,
                 _replace_leaves(template, {"remote-id": set_remote_id, "helper-address": servers}))
    applied = _wait_config_db(duthost, key, _expected_dhcp_relay_fields(servers, iid, set_remote_id), "gNMI create")
    gpins_utils.assert_dhcp_relay_converged(duthost, intf)
    pytest_assert(gpins_utils.wait_dhcp6relay_bound(duthost, timeout=CONVERGE_TIMEOUT, interval=3),
                  "dhcp6relay not running/listening after the gNMI create of {}".format(intf))
    run_dhcptest(ptfhost, duthost, dhcp_relay, applied)

    # 3. Modify through gNMI: CONFIG_DB and STATE_DB follow, the relay uses the new remote-id.
    _gnmi_set_ok(duthost, "update", oc_set_path,
                 _replace_leaves(template, {"remote-id": new_remote_id, "helper-address": servers}))
    applied = _wait_config_db(duthost, key, _expected_dhcp_relay_fields(servers, iid, new_remote_id), "gNMI update")
    gpins_utils.assert_dhcp_relay_converged(duthost, intf)
    run_dhcptest(ptfhost, duthost, dhcp_relay, applied)

    # 4. Read back through gNMI.
    node = gpins_utils.extract_openconfig_interface_node(gpins_utils.gnmi_get_json(duthost, oc_iface_path), intf)
    gpins_utils.validate_openconfig_dhcpv6_relay_schema(node, intf)
    norm = gpins_utils.normalize_openconfig_dict(node)
    pytest_assert(set(norm.get("config", {}).get("helper-address", [])) == set(servers),
                  "gNMI read-back helper-address {}, expected {}".format(norm.get("config"), servers))
    opts = norm.get("options", {}).get("config", {})
    pytest_assert(opts.get("remote-id") == new_remote_id and opts.get("interface-id") == iid,
                  "gNMI read-back options/config {}, expected remote-id {!r} interface-id {!r}".format(
                      opts, new_remote_id, iid))

    # 5. Delete through gNMI: CONFIG_DB and STATE_DB are cleared and nothing is relayed.
    _gnmi_set_ok(duthost, "delete", oc_set_path, None)
    _wait_config_db(duthost, key, None, "gNMI delete")
    gpins_utils.assert_dhcp_relay_state_cleared(duthost, intf)
    run_no_relay_test(ptfhost, duthost, dhcp_relay)
    # Teardown (dhcp_relay_config_snapshot) restores DHCP_RELAY; persisted_config_guard restores config_db.json.


def _gnmi_set_feature(duthost, state):
    """Set FEATURE|dhcp_relay ``state`` through gNMI and wait until CONFIG_DB has it."""
    feature = gpins_utils.DHCP_RELAY_FEATURE
    _gnmi_set_ok(duthost, "update", FEATURE_GNMI_PATH.format(feature=feature), {"sonic-feature:state": state},
                 prefix="")
    pytest_assert(wait_until(GNMI_CONFIG_TIMEOUT, 2, 0, lambda: gpins_utils.feature_state(duthost, feature) == state),
                  "gNMI Set of FEATURE|{} state={} did not reach CONFIG_DB (state={!r})".format(
                      feature, state, gpins_utils.feature_state(duthost, feature)))


def _wait_relay_back(duthost, intf, after):
    """dhcp6relay must be up (and DHCP_RELAY|intf converged) again; (re)start the relay container if needed."""
    if not gpins_utils.wait_dhcp6relay_bound(duthost, timeout=CONVERGE_TIMEOUT, interval=3):
        logger.info("dhcp6relay not up %ss after %s (%s); (re)starting %s", CONVERGE_TIMEOUT, after,
                    _describe_relay(duthost), gpins_utils.DHCP_RELAY_CONTAINER)
        _restart_relay_container(duthost)
    pytest_assert(gpins_utils.wait_dhcp6relay_bound(duthost, timeout=CONVERGE_TIMEOUT, interval=3),
                  "dhcp6relay did not start after {}: {}".format(after, _describe_relay(duthost)))
    gpins_utils.assert_dhcp_relay_converged(duthost, intf)


def test_dhcpv6_relay_nsf(ptfhost, duthosts, rand_one_dut_hostname, dut_dhcp_relay_data,  # noqa F811
                          validate_dut_routes_exist, dhcp_relay_config,  # noqa F811
                          dhcp_relay_config_snapshot):
    """Start GPINS NSF warm reboot during active PTF DHCPv6 relay traffic and verify DHCPv6 relay after NSF."""
    duthost = duthosts[rand_one_dut_hostname]
    dhcp_relay = dut_dhcp_relay_data[0]
    intf = dhcp_relay["relay_interface"]
    mode = dhcp_relay["relay_mode"]
    servers = gpins_utils.relay_servers(dhcp_relay)
    expected_iid = gpins_utils.design_interface_id(dhcp_relay)
    expected_remote_id = "nsf_remote_id_{}".format(gpins_utils.to_kernel_ifname(intf))

    expected_deltas = {
        "Solicit": 1,
        "Advertise": 1,
        "Request": 1,
        "Reply": 1,
        "Relay-Forward": 2 * len(servers),
        "Relay-Reply": 2,
    }

    # 1. Remove DHCP_RELAY|<intf> first and verify the interface type & address state
    #    (e.g. in l3_lla mode, the interface must have an LLA and NO GUA) before enabling DHCP relay.
    gpins_utils.del_dhcp_relay_config(duthost, intf)
    gpins_utils.assert_dhcp_relay_state_cleared(duthost, intf)
    gpins_utils.verify_interface_mode(duthost, intf, mode)

    # 2. Enable DHCPv6 relay on the verified interface and verify pre-NSF relay forwarding & counters.
    applied = gpins_utils.set_dhcp_relay_config(
        duthost,
        intf,
        servers,
        relay_mode=mode,
        enable_interface_id=True,
        interface_id=expected_iid,
        enable_remote_id=True,
        remote_id=expected_remote_id,
        rfc6939_support=True,
    )
    gpins_utils.assert_dhcp_relay_converged(duthost, intf)
    pytest_assert(gpins_utils.wait_dhcp6relay_bound(duthost, timeout=CONVERGE_TIMEOUT, interval=3),
                  "dhcp6relay not running/listening before NSF on {}".format(intf))

    pre_nsf_raw = gpins_utils.get_dhcpv6_counters(duthost, intf)
    pre_nsf_before = {k: int(v) for k, v in pre_nsf_raw.items() if str(v).lstrip("-").isdigit()}
    run_dhcptest(ptfhost, duthost, dhcp_relay, applied)
    _verify_counter_deltas(duthost, intf, pre_nsf_before, expected_deltas)

    # 3. Start PTF DHCPv6 relay traffic in a background thread and trigger full GPINS NSF warm reboot during the relay.
    stop_traffic = threading.Event()

    def _background_ptf_traffic():
        while not stop_traffic.is_set():
            try:
                run_dhcptest(ptfhost, duthost, dhcp_relay, applied)
            except Exception as exc:
                logger.debug("Background PTF DHCPv6 traffic iteration during NSF reboot ended: %s", exc)
            stop_traffic.wait(2.0)

    traffic_thread = threading.Thread(target=_background_ptf_traffic, daemon=True)
    traffic_thread.start()
    try:
        boot_info = gpins_utils.trigger_gpins_nsf_reboot(duthost)
        logger.info("GPINS NSF reboot completed: BOOT_INFO=%s", boot_info)
    finally:
        stop_traffic.set()
        traffic_thread.join(timeout=30)

    # 4. Verify CONFIG_DB DHCP_RELAY|<intf> persisted across the NSF warm reboot via dump.rdb,
    #    and restore kernel-only lab state (Loopback0 IPv6, l3_gua kernel IPv6, static PTF routes, dhcprelay-con).
    persisted_cfg = gpins_utils.get_dhcp_relay_config(duthost, intf)
    pytest_assert(gpins_utils.parse_dhcpv6_servers(persisted_cfg) == servers,
                  "CONFIG_DB DHCP_RELAY|{} did not persist across NSF reboot: got {}, expected servers {}".format(
                      intf, persisted_cfg, servers))
    gpins_utils.reapply_lab_runtime_state(duthost, dut_dhcp_relay_data, validate_dut_routes_exist)
    gpins_utils.verify_interface_mode(duthost, intf, mode)

    # 5. Wait for STATE_DB DHCP_RELAY|<intf> and dhcp6relay to converge after NSF,
    #    then run PTF DHCPTest and verify end-to-end packet relay and STATE_DB counter increments.
    gpins_utils.assert_dhcp_relay_converged(duthost, intf)
    pytest_assert(gpins_utils.wait_dhcp6relay_bound(duthost, timeout=CONVERGE_TIMEOUT, interval=3),
                  "dhcp6relay not running/listening after GPINS NSF reboot on {}".format(intf))

    post_nsf_raw = gpins_utils.get_dhcpv6_counters(duthost, intf)
    post_nsf_before = {k: int(v) for k, v in post_nsf_raw.items() if str(v).lstrip("-").isdigit()}
    run_dhcptest(ptfhost, duthost, dhcp_relay, applied)
    _verify_counter_deltas(duthost, intf, post_nsf_before, expected_deltas)

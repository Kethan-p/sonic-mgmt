"""GPINS DHCPv6 relay + Dynamic Port Breakout (DPB) test.

Test plan: "Doing Dynamic Port Breakout (DPB) on a DHCP-relay enabled interface and enable
DHCP-relay on the newly created port". Design and verified mechanisms: dpb_test_implementation_plan.md.

Flow (``test_dhcpv6_relay_dpb``):
  0. preconditions: cage master exists, no test uplink / PTF downlink involved, NSF marker clear,
     gNMI client present in telemetry-con;
  1. baseline: make the cage master a relay interface for the mode under test and enable the relay
     on it (the "DPB on a relay-enabled interface" part);
  -  snapshot CONFIG_DB + kernel state (pre-test, pristine);
  2. breakout: remove ``DHCP_RELAY|<master>`` and every other CONFIG_DB reference to the affected
     ports (the server refuses ``port is in use`` otherwise), switch the gNMI server to ``--insecure``
     for the duration of the Sets (UMF runbook: delete ``TELEMETRY|certs``, restart telemetry-con; undone
     right after), send the breakout Set from inside telemetry-con (one root ``replace`` carrying the cage
     component and the interfaces of the new layout, as sonic-pins does), wait until the new ports exist
     in kernel + CONFIG_DB + APPL_DB and the removed ones are gone;
  3. enable the relay on the newly created port(s), verify STATE_DB convergence and that dhcp6relay
     is bound;
  4. finally: platform-driven recovery (``gpins-stack restart`` reloads ``config_db.json``), re-apply
     the snapshot diff and the lab runtime state, then prove the original downlink still relays.

``test_dpb_preflight`` is the read-only dry run for iterating on the settings: it builds the plan and
payload, lists the dependencies and prints the exact gNMI command without touching the switch.

Per-testbed values (cage, mode, the root-replace opt-in, recovery mode, settle time) are ``--dhcp_relay_dpb_*``
options (see conftest.py); the verified gNMI client, target, prefix and access recipe are constants in gpins_dpb.py.
"""

import json
import logging

import pytest

from tests.common.fixtures.ptfhost_utils import copy_ptftests_directory   # noqa F401
from tests.common.fixtures.ptfhost_utils import change_mac_addresses      # noqa F401
from tests.common.fixtures.ptfhost_utils import copy_arp_responder_py     # noqa F401
from tests.common.helpers.assertions import pytest_assert, pytest_require
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
)

logger = logging.getLogger(__name__)

CONVERGE_TIMEOUT = gpins_utils.DHCP_RELAY_CONVERGE_TIMEOUT

pytestmark = [
    pytest.mark.topology('t0', 'm0', 'mx', 't0-2vlans'),
    pytest.mark.device_type('vs'),
    # The module changes and restores the port layout on purpose (and test_dpb_recover legitimately starts from a
    # leftover layout), so the framework's module-level running-config diff is noise here -- and its fallback
    # `config reload -h` cannot run on GPINS (sonic-utilities lacks jsonpatch), which turns a passed recovery into
    # an ERROR. Recovery is asserted by the tests themselves (snapshot diff, layout check, switch_health).
    pytest.mark.skip_check_dut_health,
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _protected_ports(relay_data):
    """Ports the suite depends on (PTF downlink + uplinks); the breakout must not touch them."""
    protected = set()
    for entry in relay_data:
        protected.add(entry["relay_interface"])
        protected.update(entry.get("uplink_ports", []))
    return protected


def _check_preconditions(duthost, settings, relay_data):
    """Phase 0 (read-only). Returns ``(plan, dependencies)``."""
    problems = settings.validate()
    opt_in = [p for p in problems if "allow_root_replace" in p]
    pytest_require(not opt_in or len(problems) > len(opt_in), "; ".join(opt_in))
    pytest_assert(not problems, "unsafe/unusable DPB settings: {}".format("; ".join(problems)))
    protected = _protected_ports(relay_data)
    pytest_require(settings.parent not in protected,
                   "{} is a test uplink/downlink ({}); pick another cage with --dhcp_relay_dpb_parent".format(
                       settings.parent, sorted(protected)))
    leftover = gpins_dpb.leftover_breakout(duthost, settings.parent)
    pytest_assert(leftover is None,
                  "{}: a previous run did not recover. Run test_dpb_recover (restores from the newest snapshot in "
                  "{}) before re-running".format(leftover, gpins_dpb.latest_snapshot_path(duthost.hostname)))
    try:
        plan = gpins_dpb.BreakoutPlan.build(duthost, settings.parent, settings.mode)
    except (RuntimeError, ValueError) as exc:
        pytest.skip("cannot plan DPB on {}: {}".format(duthost.hostname, exc))
    logger.info("DPB plan: %s", plan.describe())
    clash = sorted(set(plan.affected_ports) & protected)
    pytest_require(not clash, "breakout {} would touch test ports {}".format(plan.mode, clash))
    pytest_require(plan.affected_ports,
                   "{} is already in mode {}; nothing to break out".format(plan.master, plan.mode))
    pytest_require(not gpins_dpb.is_nsf_boot(duthost),
                   "NSF boot marker is set on {}: gpins-stack restart would not reload config_db.json, so the "
                   "breakout could not be undone".format(duthost.hostname))
    pytest_require(gpins_dpb.gnmi_cli_available(duthost),
                   "gnmi_cli not found in {} on {}".format(gpins_utils.TELEMETRY_CONTAINER, duthost.hostname))
    deps = gpins_dpb.port_dependencies(duthost, plan.affected_ports)
    logger.info("CONFIG_DB references to the affected ports (removed before the breakout): %s", deps or "none")
    return plan, deps


NO_CARRIER_OBSERVE_SECONDS = 15


def _enable_relay(duthost, port, mode, servers):
    """Make ``port`` a relay interface for ``mode`` and enable the relay with the design options.

    Returns ``(applied_config, converged)``. STATE_DB convergence is required only when the port has
    carrier: without carrier the kernel assigns no LLA and the GPINS relay manager publishes no
    STATE_DB entry (observed on ju3u1m1 for the un-cabled Ethernet1/7/1, 2026-10-21), so on such a
    port the test asserts the CONFIG_DB row and that dhcp6relay is still healthy, and reports the
    (expected) absence of STATE_DB state instead of failing.
    """
    gpins_dpb.prepare_relay_port(duthost, port, mode)
    gpins_utils.verify_interface_mode(duthost, port, mode)
    applied = gpins_utils.set_dhcp_relay_config(duthost, port, servers, relay_mode=mode,
                                                enable_interface_id=True, interface_id=port)
    carrier = gpins_dpb.port_has_carrier(duthost, port)
    if carrier:
        gpins_utils.assert_dhcp_relay_converged(duthost, port)
        converged = True
    else:
        converged = wait_until(NO_CARRIER_OBSERVE_SECONDS, 3, 0, gpins_utils.dhcp_relay_config_converged, duthost,
                               port)
        logger.warning("%s has %s carrier: STATE_DB convergence %s (not required without carrier)", port,
                       "no" if carrier is False else "unknown", "observed" if converged else "not observed")
    pytest_assert(gpins_utils.get_dhcp_relay_config(duthost, port), "CONFIG_DB DHCP_RELAY|{} missing".format(port))
    pytest_assert(gpins_utils.wait_dhcp6relay_bound(duthost, timeout=CONVERGE_TIMEOUT, interval=3),
                  "dhcp6relay not running/listening after enabling the relay on {}".format(port))
    logger.info("DHCP_RELAY|%s: CONFIG_DB=%s STATE_DB=%s carrier=%s converged=%s", port, applied,
                gpins_utils.get_dhcp_relay_state(duthost, port), carrier, converged)
    return applied, converged


def _recover(duthost, settings, plan, snapshot, set_result, applied, relay_data, ensure_routes_fn):
    """Phase 4 (always runs; never raises mid-way). Returns the list of recovery problems (empty = clean).

    Cheap path when the Set was rejected and the cage is untouched: CONFIG_DB/kernel diff only.
    Otherwise: platform restart + full snapshot re-apply. Both end with the lab runtime state and a
    health report of the switch.
    """
    errors = []
    set_ok = bool(set_result) and set_result.get("rc") == 0
    untouched = gpins_dpb.recovery_step(errors, "layout check", gpins_dpb.layout_unchanged, duthost, plan)
    if not set_ok and not applied and untouched:
        logger.info("Breakout not applied (Set rc=%s) and port layout unchanged on %s: restoring CONFIG_DB/kernel "
                    "from the snapshot without a stack restart", set_result and set_result.get("rc"),
                    duthost.hostname)
        # save-on-set may still have rewritten config_db.json (e.g. a partially accepted Set) while
        # TELEMETRY|certs was deleted by the gNMI session; the backup is the pre-DPB truth.
        gpins_dpb.recovery_step(errors, "persisted config check", gpins_dpb.restore_persisted_config_if_changed,
                                duthost, snapshot.get("persisted_config_md5"))
        gpins_dpb.recovery_step(errors, "CONFIG_DB restore", gpins_dpb.restore_config_db_from_snapshot, duthost,
                                snapshot["config_db"])
        gpins_dpb.recovery_step(errors, "kernel state restore", gpins_dpb.restore_kernel_state, duthost,
                                snapshot["kernel"])
    else:
        diff, recovery_errors = gpins_dpb.recover_switch_state(duthost, snapshot, settings)
        errors.extend(recovery_errors)
        if diff and settings.recovery == "stack-restart":
            errors.append("port layout differs from the pre-DPB snapshot: {}".format(sorted(diff)))
    # Loopback0 address, downlink GUA / no-GUA, FEATURE flag, static server paths, dhcprelay-con.
    gpins_dpb.recovery_step(errors, "lab runtime state", gpins_utils.reapply_lab_runtime_state, duthost, relay_data,
                            ensure_routes_fn)
    health = gpins_dpb.recovery_step(errors, "health check", gpins_dpb.switch_health, duthost)
    errors.extend(health or [])
    if errors:
        logger.error("DPB recovery on %s finished with problems: %s", duthost.hostname, errors)
    else:
        logger.info("DPB recovery on %s complete: port layout, CONFIG_DB, kernel and lab state restored",
                    duthost.hostname)
    return errors


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_dpb_recover(request, duthosts, rand_one_dut_hostname, dut_dhcp_relay_data,  # noqa F811
                     validate_dut_routes_exist):  # noqa F811
    """Manual recovery entry point: restore the newest pre-DPB snapshot of this DUT (stack restart, CONFIG_DB,
    kernel and lab state) and assert the switch is healthy. Use after a crashed/failed run; skips when there is
    no snapshot or nothing to recover."""
    duthost = duthosts[rand_one_dut_hostname]
    settings = gpins_dpb.DpbSettings.from_request(request)
    path = gpins_dpb.latest_snapshot_path(duthost.hostname)
    pytest_require(path, "no DPB snapshot for {} on this server; nothing to restore from".format(duthost.hostname))
    snapshot = gpins_dpb.load_snapshot(path)
    leftover = gpins_dpb.leftover_breakout(duthost, settings.parent)
    access = gpins_utils.gnmi_server_access_state(duthost)
    mtls_lost = (not access[gpins_utils.TELEMETRY_CERTS_ROW]
                 and bool(snapshot["config_db"].get(gpins_utils.TELEMETRY_CERTS_ROW)))
    pytest_require(leftover is not None or mtls_lost,
                   "cage {} on {} is in its persisted layout and gNMI mTLS is intact; nothing to recover".format(
                       settings.parent, duthost.hostname))
    logger.info("Recovering %s from %s (taken %s); leftover breakout: %s; gNMI access: %s", duthost.hostname, path,
                snapshot.get("taken_at"), leftover, access)
    errors = []
    if leftover is not None:
        diff, errors = gpins_dpb.recover_switch_state(duthost, snapshot, settings)
        if diff and settings.recovery == "stack-restart":
            errors.append("port layout differs from the snapshot: {}".format(sorted(diff)))
    elif mtls_lost:
        # Only the interrupted gNMI session needs undoing: put the row back and restart telemetry-con.
        gpins_dpb.recovery_step(errors, "CONFIG_DB restore", gpins_dpb.restore_config_db_from_snapshot, duthost,
                                snapshot["config_db"])
        gpins_dpb.recovery_step(errors, "telemetry restart", gpins_utils.restart_telemetry, duthost, expect_mtls=True)
    gpins_dpb.recovery_step(errors, "lab runtime state", gpins_utils.reapply_lab_runtime_state, duthost,
                            dut_dhcp_relay_data, validate_dut_routes_exist)
    errors.extend(gpins_dpb.switch_health(duthost))
    pytest_assert(not errors, "recovery of {} from {} left problems: {}".format(duthost.hostname, path, errors))
    pytest_assert(gpins_dpb.leftover_breakout(duthost, settings.parent) is None,
                  "cage {} still differs from config_db.json after recovery".format(settings.parent))


def _log_steps(settings, plan):
    """Log every Set the trigger will issue (op, path, payload, textproto, command); returns the steps."""
    steps = settings.set_steps(plan.port_index, mtu=plan.oc_mtu)
    for number, step in enumerate(steps, 1):
        request_text = gpins_dpb.set_request_textproto(step.operation, step.path, step.payload)
        logger.info("Set step %d/%d %r: %s %s\npayload:\n%s\ntextproto (%d bytes): %s\ncommand inside %s: %s",
                    number, len(steps), step.name, step.operation, step.path,
                    json.dumps(step.payload, indent=1, sort_keys=True), len(request_text), request_text[:3000],
                    gpins_utils.TELEMETRY_CONTAINER, settings.gnmi_command())
    return steps


def test_dpb_preflight(request, duthosts, rand_one_dut_hostname, dut_dhcp_relay_data):  # noqa F811
    """Read-only dry run: plan, payloads, dependencies and the exact gNMI commands for the current settings."""
    duthost = duthosts[rand_one_dut_hostname]
    settings = gpins_dpb.DpbSettings.from_request(request)
    logger.info("DPB settings: %s", settings.describe())
    plan, deps = _check_preconditions(duthost, settings, dut_dhcp_relay_data)
    _log_steps(settings, plan)
    logger.info("Serving probe once the server runs in %s mode: %s", gpins_dpb.GNMI_ACCESS,
                gpins_utils.gnmi_serving_probe(gpins_dpb.GNMI_ACCESS))
    logger.info("Relay would be enabled on: %s; gNMI access recipe: %s; gNMI server now: %s; persisted config: %s "
                "(md5 %s)", plan.relay_targets, gpins_dpb.GNMI_ACCESS, gpins_utils.gnmi_server_access_state(duthost),
                gpins_dpb.PERSISTED_CONFIG_DB, gpins_dpb.persisted_config_md5(duthost))
    logger.info("Current cage state: %s", gpins_dpb.describe_breakout_state(duthost, plan))
    pytest_assert(plan.relay_targets, "mode {} creates no new or re-created port on {}".format(plan.mode, plan.master))


def test_dhcpv6_relay_dpb(request, ptfhost, duthosts, rand_one_dut_hostname, dut_dhcp_relay_data,  # noqa F811
                          validate_dut_routes_exist, dhcp_relay_config, dhcp_relay_config_snapshot):  # noqa F811
    """DPB on a relay-enabled cage master, relay on the new port(s), platform-driven recovery."""
    duthost = duthosts[rand_one_dut_hostname]
    settings = gpins_dpb.DpbSettings.from_request(request)
    relay = dut_dhcp_relay_data[0]
    mode = relay["relay_mode"]
    servers = gpins_utils.relay_servers(relay)
    downlink = relay["relay_interface"]
    logger.info("DPB settings: %s", settings.describe())

    # ---- Phase 0: preconditions (read-only) ----
    plan, _ = _check_preconditions(duthost, settings, dut_dhcp_relay_data)
    pytest_require(plan.relay_targets, "mode {} creates no new or re-created port on {}".format(plan.mode, plan.master))
    master = plan.master

    # ---- Snapshot the pristine state first, so recovery returns to pre-test, not to phase 1 ----
    devices = sorted({master, downlink, gpins_utils.DEFAULT_LOOPBACK} | set(relay.get("uplink_ports", [])))
    snapshot = gpins_dpb.snapshot_switch_state(duthost, devices)
    snapshot_path = gpins_dpb.save_snapshot(snapshot, "dpb_{}".format(mode))
    logger.info("Pre-DPB snapshot: %d CONFIG_DB keys, kernel addrs %s, saved to %s", len(snapshot["config_db"]),
                snapshot["kernel"]["addrs"], snapshot_path)

    set_result = None
    applied = False
    recovery_errors = None
    try:
        # ---- Phase 1: relay enabled on the cage master ----
        _, master_converged = _enable_relay(duthost, master, mode, servers)

        # ---- Phase 2: breakout ----
        gpins_utils.del_dhcp_relay_config(duthost, master)
        gpins_utils.assert_dhcp_relay_state_cleared(duthost, master)
        gpins_dpb.remove_port_dependencies(duthost, plan.affected_ports)
        steps = _log_steps(settings, plan)
        # Raises (after restoring TELEMETRY|certs) if the plaintext server does not answer within the timeout.
        with gpins_utils.gnmi_cli_session(duthost, gpins_dpb.GNMI_ACCESS,
                                          serving_timeout=gpins_dpb.GNMI_SERVING_TIMEOUT):
            for step in steps:
                set_result = gpins_dpb.gnmi_set(duthost, settings, step.operation, step.path, step.payload)
                pytest_assert(set_result["rc"] == 0,
                              "gNMI Set {!r} ({} {}) rejected on {}: rc={} stdout={!r} stderr={!r}\ncmd: {}".format(
                                  step.name, step.operation, step.path, duthost.hostname, set_result["rc"],
                                  set_result["stdout"], set_result["stderr"], set_result["cmd"]))
        applied = gpins_dpb.wait_breakout_applied(duthost, plan, timeout=settings.settle_timeout)
        pytest_assert(applied, "Breakout {} on {} not visible in kernel/CONFIG_DB/APPL_DB within {}s: {}".format(
            plan.mode, master, settings.settle_timeout, gpins_dpb.describe_breakout_state(duthost, plan)))
        logger.info("Breakout applied on %s: %s", duthost.hostname, gpins_dpb.describe_breakout_state(duthost, plan))

        # ---- Phase 3: relay on the newly created port(s) ----
        converged = {}
        for child in plan.relay_targets:
            _, converged[child] = _enable_relay(duthost, child, mode, servers)
        for child in plan.relay_targets:
            counters = gpins_utils.get_dhcpv6_counters(duthost, child)
            logger.info("DHCPv6 counters present for %s after DPB: %s", child, bool(counters))
        logger.info("Relay after DPB on %s: master converged before breakout=%s, children converged=%s",
                    duthost.hostname, master_converged, converged)
    finally:
        # ---- Phase 4: recovery, always; never raises so the original failure is what pytest reports ----
        recovery_errors = _recover(duthost, settings, plan, snapshot, set_result, applied, dut_dhcp_relay_data,
                                   validate_dut_routes_exist)

    # ---- Post-recovery: switch healthy, original layout back, relay still works on the untouched downlink ----
    pytest_assert(not recovery_errors, "DPB recovery left problems on {} (snapshot: {}): {}".format(
        duthost.hostname, snapshot_path, recovery_errors))
    gpins_utils.assert_dhcp_relay_converged(duthost, downlink)
    pytest_assert(gpins_utils.wait_dhcp6relay_bound(duthost, timeout=CONVERGE_TIMEOUT, interval=3),
                  "dhcp6relay not running/listening after DPB recovery")
    run_dhcptest(ptfhost, duthost, relay, dhcp_relay_config[downlink])

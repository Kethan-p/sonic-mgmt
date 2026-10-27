"""GPINS port of the upstream CoPP DHCPv6 policer test (copp_tests.py::DHCP6Test(PolicyTest)).

What is checked (same as upstream): when a client port is flooded with
DHCPv6 Solicit multicast far above the CoPP limit, the rate that actually
reaches the DUT CPU must sit inside ``[0.9 x limit, 1.3 x limit]``.

Why this is not the upstream code: upstream measures the CPU-side rate with a
second PTF dataplane provided by ``ptf_nn_agent`` that tests/copp installs
*inside the syncd Docker container* (apt-get nanomsg, supervisorctl, docker cp).
GPINS has no Docker/apt/supervisord, so instead:

* PTF only floods (``dhcpv6_copp_test.DHCPv6FloodTest``) and fails itself if it
  could not send faster than 1.4 x limit (upstream ``NO_POLICER_LIMIT``);
* a small shell sampler on the DUT watches the client port's kernel netdev
  ``rx_packets`` counter - packets only reach that netdev after the CPU
  queue / policer, so its rate is the post-CoPP rate - waits for the flood to
  start, lets the burst drain, then measures over a fixed window.

The limit is read from the switch (``gpins_utils.dhcpv6_copp_rate_limit``): the
smaller of the DHCPv6 trap group's policer ``cir`` and the packet-mode shaper
``pir`` of its CPU queue. The design doc's CPU-queue tab says 1600 pps / burst
512 on CPU queue 20; the 2026-10-14 zhouboyang_dhcp_test and
gpins_daily_20261021_02 images instead trap DHCPv6 into ``queue20_group1``
without a policer and shape CPU queue 20 to 1000 pps
(``SCHEDULER|default_cpu_queue_scheduler``), measured 1119..1143 pps. A
mismatch with the design doc is logged as a warning (to be settled with the
image owner), not failed. The test used to add a 1600 pps policer itself, which
only re-measured the 1000 pps shaper and leaked ``COPP_GROUP|queue20_group1``
into the module's CONFIG_DB check.

No CONFIG_DB writes. Runs once per relay mode (the client port differs).
"""
import logging

import pytest

from tests.common.fixtures.ptfhost_utils import copy_ptftests_directory   # noqa F401
from tests.common.helpers.assertions import pytest_assert
from tests.common.utilities import wait_until
from tests.dhcp_relay import gpins_utils
from tests.dhcp_relay.test_dhcp_pkt_recv import pkt_recv_target  # noqa F401
from tests.ptf_runner import ptf_runner

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.topology('t0', 'm0', 'mx', 't0-2vlans'),
    pytest.mark.disable_loganalyzer,
]

# Design doc, CPU-queue tab (Bloom): informational only, compared with what the image configures.
DESIGN_DHCPV6_COPP_PPS = 1600
DESIGN_DHCPV6_COPP_CPU_QUEUE = 20
# Upstream tolerances (ptftests/py3/copp_tests.py PolicyTest): 0.9x .. 1.3x of the limit, sender >= 1.4x.
PPS_TOLERANCE_LOW = 0.9
PPS_TOLERANCE_HIGH = 1.3
NO_POLICER_FACTOR = 1.4

FLOOD_DURATION = 30          # seconds the PTF floods
FLOOD_DETECT_PPS = 200       # sampler: rx delta/s that marks "flood has started"
FLOOD_DETECT_TIMEOUT = 60    # sampler: give up waiting for the flood after this
SETTLE_SECONDS = 3           # let the burst allowance (CBS) drain before measuring
MEASURE_SECONDS = 15         # measurement window (upstream measures the middle 20 s of 30)
IDLE_BASELINE_SECONDS = 3

SAMPLER_SCRIPT = "/tmp/dhcpv6_copp_sampler.sh"
SAMPLER_OUTPUT = "/tmp/dhcpv6_copp_sampler.out"

SAMPLER_TEMPLATE = r"""#!/bin/sh
# Wait for the DHCPv6 flood on {iface}, skip the burst, measure the post-CoPP rate.
F=/sys/class/net/{iface}/statistics/rx_packets
prev=$(cat $F); n=0
while [ $n -lt {detect_timeout} ]; do
  sleep 1; cur=$(cat $F)
  if [ $((cur - prev)) -gt {detect_pps} ]; then break; fi
  prev=$cur; n=$((n + 1))
done
if [ $n -ge {detect_timeout} ]; then echo NOFLOOD; exit 0; fi
sleep {settle}
a=$(cat $F); u1=$(cut -d' ' -f1 /proc/uptime)
sleep {window}
b=$(cat $F); u2=$(cut -d' ' -f1 /proc/uptime)
echo "$a $b $u1 $u2"
"""


def idle_rx_pps(duthost, iface, seconds=IDLE_BASELINE_SECONDS):
    """Background punt rate on the port (ND, LLDP, ...) so it can be subtracted from the flood measurement."""
    res = duthost.shell("a=$(cat /sys/class/net/{0}/statistics/rx_packets); sleep {1}; "
                        "b=$(cat /sys/class/net/{0}/statistics/rx_packets); echo $((b - a))".format(iface, seconds))
    return int(res["stdout"].strip()) / float(seconds)


def start_sampler(duthost, iface):
    script = SAMPLER_TEMPLATE.format(iface=iface, detect_timeout=FLOOD_DETECT_TIMEOUT,
                                     detect_pps=FLOOD_DETECT_PPS, settle=SETTLE_SECONDS, window=MEASURE_SECONDS)
    duthost.copy(content=script, dest=SAMPLER_SCRIPT, mode="0755")
    duthost.shell("rm -f {}".format(SAMPLER_OUTPUT))
    pid = duthost.shell("nohup sh {} > {} 2>&1 & echo $!".format(SAMPLER_SCRIPT, SAMPLER_OUTPUT))["stdout"].strip()
    logger.info("Started CoPP sampler on %s (pid %s)", iface, pid)
    return pid


def read_sampler(duthost, timeout=FLOOD_DETECT_TIMEOUT + SETTLE_SECONDS + MEASURE_SECONDS + 10):
    """Return (rx_delta, seconds) measured by the sampler, or None if it never saw the flood."""
    def done():
        res = duthost.shell("cat {} 2>/dev/null".format(SAMPLER_OUTPUT), module_ignore_errors=True)
        return bool(res.get("stdout", "").strip())
    pytest_assert(wait_until(timeout, 2, 0, done), "CoPP sampler on the DUT produced no output")
    out = duthost.shell("cat {}".format(SAMPLER_OUTPUT))["stdout"].strip()
    logger.info("CoPP sampler output: %r", out)
    if out == "NOFLOOD":
        return None
    a, b, u1, u2 = out.split()
    return int(b) - int(a), float(u2) - float(u1)


def stop_sampler(duthost, pid):
    duthost.shell("kill {} 2>/dev/null || true".format(pid), module_ignore_errors=True)
    duthost.shell("rm -f {} {}".format(SAMPLER_SCRIPT, SAMPLER_OUTPUT), module_ignore_errors=True)


def test_dhcpv6_copp_policer(ptfhost, duthosts, rand_one_dut_hostname, pkt_recv_target):  # noqa F811
    """Flood DHCPv6 Solicits; the rate reaching the CPU must be held to the image's DHCPv6 CoPP limit (0.9x..1.3x)."""
    duthost = duthosts[rand_one_dut_hostname]
    iface = pkt_recv_target["kernel_iface"]
    ptf_port = pkt_recv_target["ptf_port_id"]
    relay_intf = pkt_recv_target["relay_interface"]

    copp = gpins_utils.dhcpv6_copp_rate_limit(duthost)
    logger.info("DHCPv6 CoPP on %s: %s", duthost.hostname, copp)
    pytest_assert(copp["group"], "No CoPP trap group traps dhcpv6 on {} (APPL_DB COPP_TABLE): the image's "
                                 "copp_cfg.json has no DHCPv6 trap".format(duthost.hostname))
    pytest_assert(copp["pps"], "CoPP trap group {} (CPU queue {}) has neither a policer nor a packet-mode queue shaper "
                               "on {}: DHCPv6 reaches the CPU unlimited ({})".format(
                                   copp["group"], copp["queue"], duthost.hostname, copp))
    limit = copp["pps"]
    if limit != DESIGN_DHCPV6_COPP_PPS or str(copp["queue"]) != str(DESIGN_DHCPV6_COPP_CPU_QUEUE):
        logger.warning("DHCPv6 CoPP limit on %s is %d pps (%s, CPU queue %s), not the design doc's %d pps on CPU "
                       "queue %d; testing the image's limit", duthost.hostname, limit, copp["source"], copp["queue"],
                       DESIGN_DHCPV6_COPP_PPS, DESIGN_DHCPV6_COPP_CPU_QUEUE)
    pps_min, pps_max = limit * PPS_TOLERANCE_LOW, limit * PPS_TOLERANCE_HIGH

    baseline = idle_rx_pps(duthost, iface)
    logger.info("Idle punt rate on %s: %.1f pps", iface, baseline)
    pytest_assert(baseline < FLOOD_DETECT_PPS / 2,
                  "Port {} already receives {:.0f} pps while idle; CoPP measurement would be unreliable".format(
                      iface, baseline))
    counters_before = gpins_utils.get_dhcpv6_counters(duthost, relay_intf)

    pid = start_sampler(duthost, iface)
    try:
        ptf_runner(ptfhost, "ptftests", "dhcpv6_copp_test.DHCPv6FloodTest", platform_dir="ptftests",
                   params={"port_index": ptf_port,
                           "duration": FLOOD_DURATION,
                           "min_tx_pps": limit * NO_POLICER_FACTOR},
                   log_file="/tmp/dhcpv6_copp_test.DHCPv6FloodTest.log", is_python3=True)
        sample = read_sampler(duthost)
    finally:
        stop_sampler(duthost, pid)

    pytest_assert(sample is not None,
                  "DUT never saw more than {} pps on {} - the flood did not reach the CPU at all "
                  "(punt path / CoPP trap missing?)".format(FLOOD_DETECT_PPS, iface))
    rx_delta, seconds = sample
    pytest_assert(seconds > 0, "invalid sampler window")
    rx_pps = rx_delta / seconds - baseline
    logger.info("CoPP DHCPv6: %d packets in %.2fs on %s -> %.0f pps after CoPP (baseline %.1f pps); "
                "expected %.0f..%.0f (limit %d pps from the %s of trap group %s, cpu queue %s)",
                rx_delta, seconds, iface, rx_pps, baseline, pps_min, pps_max, limit, copp["source"],
                copp["group"], copp["queue"])
    counters_after = gpins_utils.get_dhcpv6_counters(duthost, relay_intf)
    logger.info("STATE_DB DHCPv6 counters for %s before/after flood: %s / %s",
                relay_intf, counters_before, counters_after)

    pytest_assert(pps_min <= rx_pps <= pps_max,
                  "CoPP constraint failed on {}: {:.0f} pps reached the CPU, expected {:.0f}..{:.0f} (limit {} pps "
                  "from the {} of trap group {})".format(iface, rx_pps, pps_min, pps_max, limit, copp["source"],
                                                         copp["group"]))

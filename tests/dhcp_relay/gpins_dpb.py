"""GPINS Dynamic Port Breakout (DPB) helpers for the DHCPv6-relay tests.

What is known about DPB on GPINS (Brixia, verified 2026-10-21 unless noted):

* The breakout unit is the **8-lane cage**. CONFIG_DB describes it with
  ``BREAKOUT_CFG|<master> {lanes, brkout_mode}`` plus one ``BREAKOUT_PORTS|<sub-port> {master}``
  row per possible sub-port. On ju3u1m1: ``BREAKOUT_CFG|Ethernet1/7/1 lanes=57..64
  brkout_mode=2x400G`` -> ports ``Ethernet1/7/1`` (lanes 57-60) and ``Ethernet1/7/5`` (61-64).
* The breakout is requested over **gNMI**, never by editing ``PORT|*`` (Boyang). The reference
  implementation is sonic-pins ``tests/thinkit_gnmi_interface_util.cc``
  (``GetBreakoutModeConfigJson``): one ``SetRequest`` ``replace`` at the root path whose payload
  carries ``openconfig-platform:components`` (component ``1/<port index>`` with
  ``port/breakout-mode/groups``) **and** the resulting ``openconfig-interfaces:interfaces``
  entries. Modes are strings such as ``2x400G`` or ``2x200G(4)+1x400G(4)`` (``(n)`` = lanes of
  that group); child names are ``Ethernet<slot>/<port>/<first lane of the child>``.
* Server side the mgmt-framework transformer ``YangToDb_port_breakout_subtree_xfmr`` rejects the
  Set with ``port <X> is in use`` while configuration references a port the breakout would
  remove or re-create, so dependencies (``INTERFACE``, ``DHCP_RELAY``, VLAN/LAG membership)
  must be removed first.
* Recovery: ``/etc/init.d/gpins-stack restart`` rebuilds CONFIG_DB from the persisted
  ``config_db.json`` (Redis has no persistence; verified on the daily image, scripts identical on
  ``zhouboyang_dhcp_test``), so the platform restores the default port layout itself. Everything
  the session had written to Redis/kernel must then be re-applied from a pre-DPB snapshot.
* Local gNMI access (telemetry-con, port 9339): the server runs with ``--ca_crt`` and therefore demands
  a verifiable client certificate; the lab's ``/keys/ca_crt.pem`` is the server's own self-signed cert,
  so no local client can satisfy it. The UMF runbook (go/gpins-umf-kb, "Running gNMI CLI", option 1)
  deletes CONFIG_DB ``TELEMETRY|certs`` and restarts telemetry-con, after which the server starts
  ``--insecure`` -- **plain gRPC, no TLS at all** -- and ``gnmi_cli -insecure`` talks to it (verified
  2026-10-21: capabilities + Gets with prefix ``openconfig:chassis`` answered; the empty pathz policy and
  the allow-all authz policy let Gets through; whole delete/restart/restore cycle 66 s, ``config_db.json``
  untouched). See ``gpins_utils.gnmi_cli_session``. Moving the key files aside instead does NOT work with the
  current ``telemetry.sh`` (it keeps emitting ``--server_crt/--ca_crt`` and the server fails to start).
  ``TELEMETRY|gnmi save_on_set=true`` makes every accepted Set rewrite ``config_db.json``.
* Port shape on ju3u1m1: ``interface[...]/config`` is ``{enabled, mtu 9122, name, type}`` (CONFIG_DB
  ``PORT`` mtu 9100 + 22), no ``PORT|*`` row has a P4RT ``id``; component ``1/7`` exposes
  ``port/breakout-mode``. The post-breakout interface config mirrors that shape.

* Set semantics (runs of 2026-10-21/22, see :data:`GNMI_SET_CMD`): a root ``update`` with components
  + interfaces is rejected (``Invalid Interface``: the child does not exist yet); a ``replace`` of the
  cage's ``breakout-mode`` alone is accepted but only rewrites ``BREAKOUT_CFG`` and the port aliases; the
  UMF binary does entry-level partial replace and locks the cage's ports during DPB. The breakout is
  therefore one root ``replace`` carrying the component and the interfaces, exactly as sonic-pins does.
  Verified end to end (breakout, relay on the new ports, recovery) on ju3u1m1, 2026-10-22.
* Recovery pitfall: the GPINS host has no ``setsid``; a restart launched with it never runs (run 3), so
  :func:`gpins_stack_restart` uses a nohup'ed subshell and fails hard if the stack does not go down.

Only what legitimately varies per testbed is a pytest option (cage, mode, the root-replace opt-in, recovery
mode, settle time); the gNMI client, target, prefix and access recipe that worked are constants below.
"""

import base64
import collections
import json
import logging
import os
import re
import shlex
import tempfile
import time


from tests.common.utilities import wait_until
from tests.dhcp_relay import gpins_utils as gu

logger = logging.getLogger(__name__)

CAGE_LANES = 8
GPINS_STACK_SCRIPT = "/etc/init.d/gpins-stack"
STACK_RESTART_LOG = "/tmp/sonic-mgmt-gpins-stack-restart.log"
CONTAINER_REQUEST_PATH = "/tmp/sonic-mgmt-dpb-request.textproto"
# Tables the platform owns; recovery asserts them against the snapshot instead of rewriting them.
PLATFORM_PORT_TABLES = ("PORT|", "BREAKOUT_CFG|", "BREAKOUT_PORTS|")

DEFAULT_PARENT = "Ethernet1/7/1"
DEFAULT_MODE = "2x200G(4)+1x400G(4)"
# How the breakout is requested. Findings on ju3u1m1 (2026-10-21/22):
#   * root *update* with components + interfaces -> "Interface Ethernet1/7/3 cannot be configured; err = Invalid
#     Interface" (the interface transformer validates the child against the live PORT table);
#   * *replace* of components/component[name=1/7]/port/breakout-mode alone -> accepted, but it only rewrote
#     BREAKOUT_CFG (mode, "port 1/7") and the PORT aliases of the cage; no port was re-laned or created in 3 min;
#   * the UMF binary implements replace as entry-level *partial* replace ("Since its partial replace delete aux
#     fields" / "... hence its partial replace") and locks the cage's ports during DPB ("Locked port %v during DPB"),
#     i.e. the new ports are created from the interface entries sent in the SAME transaction as the component.
# Hence one root replace, as sonic-pins' GetBreakoutModeConfigJson/BuildGnmiSetRequest("", kReplace) does, with
# the interface entries in the shape this switch reports for its own ports (breakout_root_payload).
#
# The request goes out from inside telemetry-con with the openconfig gnmi_cli (full SetRequest textproto, so the
# prefix/origin are under our control). The server is switched to plain gRPC for the Set with the UMF runbook
# recipe (go/gpins-umf-kb "Running gNMI CLI", option 1: CONFIG_DB TELEMETRY|certs deleted, telemetry-con
# restarted; gpins_utils.gnmi_cli_session "plaintext" mode), where gnmi_cli needs -insecure (it has no -notls).
GNMI_TARGET = "127.0.0.1:9339"
GNMI_PREFIX = "openconfig:chassis"      # <origin>:<target> as in sonic-pins BuildGnmiSetRequest
GNMI_ACCESS = "plaintext"
GNMI_SERVING_TIMEOUT = 90
GNMI_SET_CMD = "gnmi_cli -address {target} -insecure -set -proto \"$(cat {proto_file})\""
DEFAULT_TEST_GUA_TEMPLATE = "2001:db8:{port}:{lane}::1/64"
# What a cold db-con start loads (redis-container post_container_run); backed up before the breakout because the
# server runs --with-save-on-set=true (TELEMETRY|gnmi save_on_set): every accepted Set rewrites this file, so a
# broken-out layout (and a missing TELEMETRY|certs row) would otherwise survive gpins-stack restart.
PERSISTED_CONFIG_DB = "/mnt/region_config/container_files/etc/sonic/config_db.json"
PERSISTED_CONFIG_DB_BACKUP = PERSISTED_CONFIG_DB + ".sonic-mgmt-dpb.bak"


# ---------------------------------------------------------------------------
# Pure breakout-mode logic (no switch access; mirrors sonic-pins GetExpectedPortInfoForBreakoutMode)
# ---------------------------------------------------------------------------

BreakoutGroup = collections.namedtuple("BreakoutGroup", "num_breakouts speed_gbps lanes")
_GROUP_RE = re.compile(r"^(\d+)x(\d+)G(?:\[[^\]]*\])?(?:\((\d+)\))?$")
_PORT_RE = re.compile(r"^Ethernet(\d+)/(\d+)(?:/(\d+))?$")


def parse_breakout_mode(mode, cage_lanes=CAGE_LANES):
    """Parse ``"2x400G"`` / ``"2x200G(4)+1x400G(4)"`` / ``"4x100G[50G]"`` into ``[BreakoutGroup]``.

    ``(n)`` is the number of lanes of that group (platform.json notation); without it the cage is
    split equally between the groups (sonic-pins behaviour). ``[alt speeds]`` are ignored.
    """
    text = re.sub(r"\s+", "", str(mode))
    parts = [p for p in text.split("+") if p]
    if not parts:
        raise ValueError("empty breakout mode {!r}".format(mode))
    groups = []
    for part in parts:
        match = _GROUP_RE.match(part)
        if not match:
            raise ValueError("invalid breakout group {!r} in mode {!r} (expected e.g. 2x200G or 2x200G(4))".format(
                part, mode))
        groups.append(BreakoutGroup(int(match.group(1)), int(match.group(2)),
                                    int(match.group(3)) if match.group(3) else None))
    if any(g.lanes is None for g in groups):
        if cage_lanes % len(groups):
            raise ValueError("mode {!r}: {} lanes cannot be split equally over {} groups".format(
                mode, cage_lanes, len(groups)))
        share = cage_lanes // len(groups)
        groups = [g._replace(lanes=g.lanes or share) for g in groups]
    if sum(g.lanes for g in groups) != cage_lanes:
        raise ValueError("mode {!r} uses {} lanes, the cage has {}".format(
            mode, sum(g.lanes for g in groups), cage_lanes))
    for g in groups:
        if g.lanes % g.num_breakouts:
            raise ValueError("mode {!r}: {} lanes cannot host {} breakouts".format(mode, g.lanes, g.num_breakouts))
    return groups


def slot_port_lane(port):
    """``"Ethernet1/7/3"`` -> ``(1, 7, 3)``; a port without a lane part is lane 1."""
    match = _PORT_RE.match(str(port))
    if not match:
        raise ValueError("{!r} is not an Ethernet<slot>/<port>[/<lane>] name".format(port))
    return int(match.group(1)), int(match.group(2)), int(match.group(3) or 1)


def expected_breakout_ports(master, mode, cage_lanes=CAGE_LANES):
    """Ports that exist after applying ``mode`` to the cage of ``master`` (lane 1 of the cage).

    Returns a list of dicts ``{name, speed_gbps, num_lanes, physical_channels, group}`` in lane
    order, e.g. for ``Ethernet1/7/1`` + ``2x200G(4)+1x400G(4)``:
    ``Ethernet1/7/1`` [0,1], ``Ethernet1/7/3`` [2,3], ``Ethernet1/7/5`` [4..7].
    """
    slot, port, lane = slot_port_lane(master)
    if lane != 1:
        raise ValueError("{} is not a cage master (lane 1)".format(master))
    ports = []
    channel = 0
    for group_index, group in enumerate(parse_breakout_mode(mode, cage_lanes)):
        per_port = group.lanes // group.num_breakouts
        for _ in range(group.num_breakouts):
            ports.append({
                "name": "Ethernet{}/{}/{}".format(slot, port, channel + 1),
                "speed_gbps": group.speed_gbps,
                "num_lanes": per_port,
                "physical_channels": list(range(channel, channel + per_port)),
                "group": group_index,
            })
            channel += per_port
    return ports


def breakout_component_name(port_index):
    """OpenConfig component name of a cage (sonic-pins convention ``1/<port index>``)."""
    return "1/{}".format(port_index)


# OpenConfig mtu includes the 22-byte L2 header: CONFIG_DB PORT mtu 9100 (GPINS default) is reported as 9122 by the
# switch's own gNMI Get of interface[...]/config (ju3u1m1 2026-10-21).
OC_MTU_OVERHEAD = 22
DEFAULT_PORT_MTU = 9100
DEFAULT_OC_MTU = DEFAULT_PORT_MTU + OC_MTU_OVERHEAD


def breakout_groups_json(mode, cage_lanes=CAGE_LANES):
    """``groups/group`` list of the OpenConfig ``breakout-mode`` container for ``mode``."""
    group_json = []
    for index, group in enumerate(parse_breakout_mode(mode, cage_lanes)):
        group_json.append({
            "index": index,
            "config": {
                "index": index,
                "num-breakouts": group.num_breakouts,
                "breakout-speed": "openconfig-if-ethernet:SPEED_{}GB".format(group.speed_gbps),
                "num-physical-channels": group.lanes // group.num_breakouts,
            },
        })
    return group_json


def breakout_root_payload(master, mode, port_index, mtu=DEFAULT_OC_MTU, cage_lanes=CAGE_LANES):
    """Root-path ``replace`` payload for the breakout: the cage component (breakout groups + port-id) **and** the
    interfaces of the new layout in one transaction -- the layout of sonic-pins' ``GetBreakoutModeConfigJson``,
    which UMF needs because the new ports are created from these interface entries while the component locks the
    cage. The interface entries have the shape this switch reports for its own ports (``{enabled, mtu, name,
    type}`` + ethernet port-speed; no P4RT id, which no ``PORT|*`` row has on GPINS).
    """
    component_name = breakout_component_name(port_index)
    interfaces = []
    for port in expected_breakout_ports(master, mode, cage_lanes):
        interfaces.append({
            "name": port["name"],
            "config": {
                "name": port["name"],
                "type": "iana-if-type:ethernetCsmacd",
                "enabled": True,
                "mtu": int(mtu),
            },
            "openconfig-if-ethernet:ethernet": {
                "config": {"port-speed": "openconfig-if-ethernet:SPEED_{}GB".format(port["speed_gbps"])},
            },
        })
    return {
        "openconfig-interfaces:interfaces": {"interface": interfaces},
        "openconfig-platform:components": {
            "component": [{
                "name": component_name,
                "config": {"name": component_name},
                "port": {
                    "config": {"port-id": int(port_index)},
                    "breakout-mode": {"groups": {"group": breakout_groups_json(mode, cage_lanes)}},
                },
            }],
        },
    }


# One gNMI Set of the configured trigger (name, operation, path, payload).
SetStep = collections.namedtuple("SetStep", "name operation path payload")


# ---------------------------------------------------------------------------
# gNMI request encoding (textproto for gnmi_cli)
# ---------------------------------------------------------------------------

def split_gnmi_path(path):
    """``/a/b[name=Ethernet1/7/1][x=y]/c`` -> ``[("a", {}), ("b", {"name": "Ethernet1/7/1", "x": "y"}), ("c", {})]``.

    Slashes inside ``[...]`` keys do not split the path. ``/`` (root) -> ``[]``.
    """
    elems = []
    buf = ""
    depth = 0
    for ch in str(path).strip():
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "/" and depth == 0:
            if buf:
                elems.append(buf)
            buf = ""
        else:
            buf += ch
    if buf:
        elems.append(buf)
    parsed = []
    for elem in elems:
        name = elem.split("[", 1)[0]
        keys = dict(re.findall(r"\[([^=\]]+)=([^\]]*)\]", elem))
        parsed.append((name, keys))
    return parsed


def _tp_escape(text):
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def path_textproto(path):
    """gNMI ``Path`` textproto body (``elem { name: "..." key { key: "..." value: "..." } }`` ...)."""
    parts = []
    for name, keys in split_gnmi_path(path):
        key_text = "".join(' key { key: "%s" value: "%s" }' % (_tp_escape(k), _tp_escape(v)) for k, v in keys.items())
        parts.append('elem { name: "%s"%s }' % (_tp_escape(name), key_text))
    return " ".join(parts)


def _prefix_textproto(prefix):
    """``"openconfig:chassis"`` -> ``prefix { origin: "openconfig" target: "chassis" }`` ('' / 'none' -> '')."""
    if not prefix or str(prefix).lower() == "none":
        return ""
    origin, _, target = str(prefix).partition(":")
    fields = []
    if origin:
        fields.append('origin: "%s"' % _tp_escape(origin))
    if target:
        fields.append('target: "%s"' % _tp_escape(target))
    return "prefix { %s }" % " ".join(fields) if fields else ""


def set_request_textproto(operation, path, payload, prefix=GNMI_PREFIX):
    """gNMI ``SetRequest`` textproto with one ``replace``/``update``/``delete`` entry."""
    if operation not in ("replace", "update", "delete"):
        raise ValueError("unsupported gNMI set operation {!r}".format(operation))
    path_body = path_textproto(path)
    if operation == "delete":
        entry = "delete { %s }" % path_body
    else:
        json_text = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        entry = '%s { path { %s } val { json_ietf_val: "%s" } }' % (operation, path_body, _tp_escape(json_text))
    return " ".join(part for part in (_prefix_textproto(prefix), entry) if part)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class DpbSettings(object):
    """The per-testbed DPB knobs (pytest ``--dhcp_relay_dpb_*`` options); the verified gNMI client, target, prefix
    and access recipe are module constants (:data:`GNMI_SET_CMD`, :data:`GNMI_ACCESS`)."""

    def __init__(self, parent=DEFAULT_PARENT, mode=DEFAULT_MODE, allow_root_replace=False, recovery="stack-restart",
                 settle_timeout=180):
        self.parent = parent
        self.mode = re.sub(r"\s+", "", mode)
        self.allow_root_replace = allow_root_replace
        self.recovery = recovery
        self.settle_timeout = settle_timeout

    @classmethod
    def from_request(cls, request):
        opt = request.config.getoption
        return cls(
            parent=opt("--dhcp_relay_dpb_parent"),
            mode=opt("--dhcp_relay_dpb_mode"),
            allow_root_replace=opt("--dhcp_relay_dpb_allow_root_replace"),
            recovery=opt("--dhcp_relay_dpb_recovery"),
            settle_timeout=opt("--dhcp_relay_dpb_settle_timeout"),
        )

    def validate(self):
        """Return a list of reasons the settings are unsafe/unusable (empty when fine).

        The breakout is a ``replace`` at the root path and stays behind an explicit opt-in even though the
        evidence says it is safe (the UMF binary implements entry-level *partial* replace, sonic-pins sends exactly
        this payload as a root replace on production-like switches, and it passed end to end on ju3u1m1): if the
        semantics were "replace the whole configuration", the management path could go with it, which is not
        recoverable from the test server. The opt-in is the operator's acknowledgement of that residual risk.
        """
        problems = []
        if not self.allow_root_replace:
            problems.append("the breakout is a root-path gNMI replace; pass --dhcp_relay_dpb_allow_root_replace to "
                            "acknowledge it (UMF replace is entry-level partial replace, verified from the binary; "
                            "sonic-pins uses the same request)")
        return problems

    def set_steps(self, port_index, mtu=DEFAULT_OC_MTU):
        """The breakout as an ordered :class:`SetStep` list (one root ``replace``). ``mtu`` is the OpenConfig value
        (CONFIG_DB mtu + 22) for the interface entries."""
        return [SetStep("breakout (components + interfaces, root replace)", "replace", "/",
                        breakout_root_payload(self.parent, self.mode, port_index, mtu=mtu))]

    @staticmethod
    def gnmi_command():
        """The ``gnmi_cli`` Set command run inside telemetry-con (reads the request textproto written there)."""
        return GNMI_SET_CMD.format(target=GNMI_TARGET, proto_file=CONTAINER_REQUEST_PATH)

    def describe(self):
        return ("parent={} mode={} allow_root_replace={} recovery={} settle_timeout={}s (gNMI: {} prefix={} access={})"
                .format(self.parent, self.mode, self.allow_root_replace, self.recovery, self.settle_timeout,
                        GNMI_TARGET, GNMI_PREFIX, GNMI_ACCESS))


# ---------------------------------------------------------------------------
# Switch facts: breakout tables, dependencies
# ---------------------------------------------------------------------------

def get_breakout_cfg(duthost, master):
    """``BREAKOUT_CFG|<master>`` fields (``{}`` if the port is not a cage master)."""
    return gu.redis_hgetall(duthost, "CONFIG_DB", "BREAKOUT_CFG|{}".format(master))


def get_port_index(duthost, port):
    """``PORT|<port> index`` as int (first value of a ``7,7,7,7`` style list)."""
    raw = gu.redis_hget(duthost, "CONFIG_DB", "PORT|{}".format(port), "index")
    if raw is None:
        raise RuntimeError("PORT|{} has no index field on {}".format(port, duthost.hostname))
    return int(str(raw).split(",")[0])


def current_cage_ports(duthost, master):
    """``{port: PORT fields}`` for every CONFIG_DB PORT whose BREAKOUT_PORTS master is ``master``."""
    members = {p for p, f in gu.get_table(duthost, "CONFIG_DB", "BREAKOUT_PORTS").items() if f.get("master") == master}
    ports = gu.get_table(duthost, "CONFIG_DB", "PORT")
    return {p: ports[p] for p in sorted(members) if p in ports}


# CONFIG_DB rows that reference a port. The L3/relay set is what this test creates itself and removes before
# the breakout; the QoS/buffer rows come from the platform config and are only reported (the verified breakout,
# ju3u1m1 2026-10-22, did not need them removed).
L3_DEPENDENCY_PATTERNS = ("INTERFACE|{p}", "INTERFACE|{p}|*", "DHCP_RELAY|{p}", "VLAN_MEMBER|*|{p}",
                          "PORTCHANNEL_MEMBER|*|{p}")
QOS_DEPENDENCY_PATTERNS = ("PORT_QOS_MAP|{p}", "QUEUE|{p}|*", "BUFFER_PG|{p}|*", "BUFFER_QUEUE|{p}|*")
DEPENDENCY_PATTERNS = L3_DEPENDENCY_PATTERNS + QOS_DEPENDENCY_PATTERNS


def port_dependencies(duthost, ports):
    """CONFIG_DB keys that reference ``ports`` and may make the breakout fail with ``port is in use``.

    ``CABLE_LENGTH`` references are reported too (as ``CABLE_LENGTH|<key>[<port>]``) but are not removable.
    """
    deps = {}
    cable_lengths = gu.get_table(duthost, "CONFIG_DB", "CABLE_LENGTH")
    for port in ports:
        keys = []
        for pattern in DEPENDENCY_PATTERNS:
            keys.extend(gu.redis_keys(duthost, "CONFIG_DB", pattern.format(p=port)))
        keys.extend("CABLE_LENGTH|{}[{}]".format(key, port) for key, fields in cable_lengths.items() if port in fields)
        if keys:
            deps[port] = sorted(set(keys))
    return deps


def remove_port_dependencies(duthost, ports):
    """Delete the L3/relay dependencies of ``ports`` (the rows this test may own); returns the CONFIG_DB snapshot
    of what was removed (``gpins_utils.restore_config_db`` format) so a cheap recovery can put them back."""
    patterns = [pattern.format(p=port) for port in ports for pattern in L3_DEPENDENCY_PATTERNS]
    snapshot = gu.snapshot_config_db(duthost, patterns)
    keys = sorted(snapshot["entries"])
    if keys:
        logger.info("Removing %d CONFIG_DB dependencies of %s before DPB: %s", len(keys), ports, keys)
        gu.redis_del(duthost, "CONFIG_DB", *keys)
    return snapshot


class BreakoutPlan(object):
    """What a breakout of ``master`` to ``mode`` means on this switch (built from live CONFIG_DB)."""

    def __init__(self, master, mode, port_index, current_mode, current_ports, expected_ports):
        self.master = master
        self.mode = mode
        self.port_index = port_index
        self.component = breakout_component_name(port_index)
        self.current_mode = current_mode
        self.current_ports = current_ports                      # {name: PORT fields}
        self.expected_ports = expected_ports                    # [dict]
        self.expected_names = [p["name"] for p in expected_ports]
        self.removed_ports = sorted(set(current_ports) - set(self.expected_names))
        self.new_ports = [n for n in self.expected_names if n not in current_ports]
        self.changed_ports = []
        for port in expected_ports:
            before = current_ports.get(port["name"])
            if before is None:
                continue
            lanes = [x for x in before.get("lanes", "").split(",") if x]
            if len(lanes) != port["num_lanes"] or str(before.get("speed")) != str(port["speed_gbps"] * 1000):
                self.changed_ports.append(port["name"])
        self.unchanged_ports = [n for n in self.expected_names
                                if n in current_ports and n not in self.changed_ports]
        self.affected_ports = sorted(set(self.removed_ports) | set(self.new_ports) | set(self.changed_ports))

    @property
    def oc_mtu(self):
        """OpenConfig MTU for the new ports: the master's CONFIG_DB mtu (+22 L2 bytes), GPINS default if unset."""
        raw = (self.current_ports.get(self.master) or {}).get("mtu")
        return (int(raw) if str(raw or "").isdigit() else DEFAULT_PORT_MTU) + OC_MTU_OVERHEAD

    @classmethod
    def build(cls, duthost, master, mode, cage_lanes=CAGE_LANES):
        cfg = get_breakout_cfg(duthost, master)
        if not cfg:
            raise RuntimeError("BREAKOUT_CFG|{} does not exist on {}: not a cage master".format(
                master, duthost.hostname))
        lanes = [x for x in cfg.get("lanes", "").split(",") if x]
        if lanes and len(lanes) != cage_lanes:
            logger.warning("BREAKOUT_CFG|%s has %d lanes, expected %d; using %d", master, len(lanes), cage_lanes,
                           len(lanes))
            cage_lanes = len(lanes)
        return cls(master, mode, get_port_index(duthost, master), cfg.get("brkout_mode", ""),
                   current_cage_ports(duthost, master), expected_breakout_ports(master, mode, cage_lanes))

    @property
    def relay_targets(self):
        """Ports the relay gets enabled on after the breakout: the newly created ones (plus the
        master if it was re-created with a different lane count/speed)."""
        return [n for n in self.expected_names if n in self.new_ports or n in self.changed_ports]

    def describe(self):
        return ("{} (component {}, current mode {}) -> {}: expected {}, new {}, changed {}, removed {}, "
                "unchanged {}".format(self.master, self.component, self.current_mode or "?", self.mode,
                                      self.expected_names, self.new_ports, self.changed_ports, self.removed_ports,
                                      self.unchanged_ports))


# ---------------------------------------------------------------------------
# gNMI access from inside telemetry-con
# ---------------------------------------------------------------------------

def _ctr_sh(duthost, script):
    """Run a shell snippet inside telemetry-con; returns the (duck-typed) Ansible result."""
    return gu.ctr_exec(duthost, "sh -c {}".format(shlex.quote(script)), container=gu.TELEMETRY_CONTAINER)


def gnmi_cli_available(duthost):
    """True if the openconfig ``gnmi_cli`` of :data:`GNMI_SET_CMD` exists inside telemetry-con."""
    res = _ctr_sh(duthost, "command -v gnmi_cli")
    return res.get("rc") == 0 and bool((res.get("stdout") or "").strip())


def write_container_file(duthost, path, content):
    """Write ``content`` to ``path`` inside telemetry-con (base64 over the exec channel)."""
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    res = _ctr_sh(duthost, "printf %s {} | base64 -d > {} && wc -c < {}".format(
        shlex.quote(encoded), shlex.quote(path), shlex.quote(path)))
    if res.get("rc") != 0:
        raise RuntimeError("Cannot write {} in {} on {}: rc={} {}".format(
            path, gu.TELEMETRY_CONTAINER, duthost.hostname, res.get("rc"), (res.get("stderr") or "").strip()))
    return int((res.get("stdout") or "0").strip() or 0)


def gnmi_set(duthost, settings, operation, path, payload):
    """``replace``/``update``/``delete`` ``payload`` at ``path`` with :data:`GNMI_SET_CMD` from inside telemetry-con
    (the server must already accept the plaintext client, see ``gpins_utils.gnmi_cli_session``).

    Returns ``{cmd, rc, stdout, stderr, request}``.
    """
    return gnmi_set_path(duthost, operation, path, payload, cmd=settings.gnmi_command())


# gnmi_cli transport flag per gnmi_cli_session mode: -insecure = plain gRPC, -tls_skip_verify = TLS, no verification.
GNMI_SET_CMD_BY_ACCESS = {
    "plaintext": GNMI_SET_CMD,
    "server-tls": "gnmi_cli -address {target} -tls_skip_verify -set -proto \"$(cat {proto_file})\"",
}


def gnmi_set_path(duthost, operation, path, payload, access=GNMI_ACCESS, prefix=GNMI_PREFIX, cmd=None):
    """One gNMI Set (``replace``/``update``/``delete``) from inside telemetry-con, for a server already running in
    the ``gpins_utils.gnmi_cli_session`` mode ``access``. Returns ``{cmd, rc, stdout, stderr, request}``."""
    request = set_request_textproto(operation, path, payload, prefix)
    write_container_file(duthost, CONTAINER_REQUEST_PATH, request)
    if cmd is None:
        key = "plaintext" if access.startswith("plaintext") else "server-tls"
        cmd = GNMI_SET_CMD_BY_ACCESS[key].format(target=GNMI_TARGET, proto_file=CONTAINER_REQUEST_PATH)
    logger.info("gNMI %s on %s via %s: %s", operation, duthost.hostname, gu.TELEMETRY_CONTAINER, cmd)
    logger.info("gNMI request textproto (%d bytes): %s", len(request), request[:2000])
    res = _ctr_sh(duthost, cmd)
    result = {"cmd": cmd, "rc": res.get("rc"), "stdout": (res.get("stdout") or "").strip(),
              "stderr": (res.get("stderr") or "").strip(), "request": request}
    logger.info("gNMI %s result on %s: rc=%s stdout=%r stderr=%r", operation, duthost.hostname,
                result["rc"], result["stdout"][:1500], result["stderr"][:1500])
    return result


# ---------------------------------------------------------------------------
# Snapshot / restore of everything the stack restart will wipe
# ---------------------------------------------------------------------------

def port_has_carrier(duthost, port):
    """True/False from ``/sys/class/net/<dev>/carrier``; None if the netdev does not exist.

    A port without carrier (un-cabled spare, or a freshly broken-out child whose peer is not
    broken out) never gets an LLA from the kernel and the GPINS relay manager does not publish a
    STATE_DB ``DHCP_RELAY`` entry for it (observed on ju3u1m1, 2026-10-21), so convergence can only
    be required on ports that report carrier.
    """
    out = gu.shell_stdout(duthost, "cat /sys/class/net/{}/carrier 2>/dev/null || echo missing".format(
        shlex.quote(_kernel_dev(port))))
    value = (out or "").strip()
    if value in ("", "missing"):
        return None
    return value == "1"


def _kernel_dev(name):
    return gu.to_kernel_ifname(name) if str(name).startswith("Ethernet") else str(name)


def kernel_state(duthost, devices):
    """Global IPv6 addresses of ``devices``, static (``proto boot``/``static``) routes and permanent neighbours."""
    state = {"addrs": {}, "routes": [], "neighs": []}
    for dev in devices:
        kdev = _kernel_dev(dev)
        state["addrs"][kdev] = gu.kernel_ipv6_addrs(duthost, kdev, "global")
    for proto in ("boot", "static"):
        out = gu.shell_stdout(duthost, "ip -6 route show proto {} 2>/dev/null".format(proto))
        state["routes"].extend(line.strip() for line in (out or "").splitlines() if line.strip())
    out = gu.shell_stdout(duthost, "ip -6 neigh show nud permanent 2>/dev/null")
    state["neighs"] = [line.strip() for line in (out or "").splitlines() if line.strip()]
    return state


def persisted_config_md5(duthost):
    out = gu.shell_stdout(duthost, "md5sum {} 2>/dev/null | cut -d' ' -f1".format(PERSISTED_CONFIG_DB))
    return (out or "").strip() or None


def backup_persisted_config(duthost):
    """Copy ``config_db.json`` aside and return its md5 (None when the file does not exist on this image)."""
    md5 = persisted_config_md5(duthost)
    if md5:
        duthost.shell("cp -p {} {}".format(PERSISTED_CONFIG_DB, PERSISTED_CONFIG_DB_BACKUP), module_ignore_errors=True)
        logger.info("Backed up %s (md5 %s) to %s", PERSISTED_CONFIG_DB, md5, PERSISTED_CONFIG_DB_BACKUP)
    return md5


def restore_persisted_config_if_changed(duthost, md5_before):
    """Put the backup back if ``config_db.json`` changed since the snapshot (i.e. the gNMI server persists
    its changes, which the recovery assumptions say it must not). Returns True if a restore happened."""
    if not md5_before:
        return False
    md5_now = persisted_config_md5(duthost)
    if md5_now == md5_before:
        return False
    logger.error("%s on %s changed during the test (md5 %s -> %s): the gNMI server persists its changes; restoring "
                 "the backup so a gpins-stack restart reloads the pre-test config", PERSISTED_CONFIG_DB,
                 duthost.hostname, md5_before, md5_now)
    duthost.shell("cp -p {} {}".format(PERSISTED_CONFIG_DB_BACKUP, PERSISTED_CONFIG_DB), module_ignore_errors=True)
    if persisted_config_md5(duthost) != md5_before:
        logger.error("Could not restore %s from %s on %s", PERSISTED_CONFIG_DB, PERSISTED_CONFIG_DB_BACKUP,
                     duthost.hostname)
    return True


def snapshot_switch_state(duthost, devices):
    """Everything recovery needs: full CONFIG_DB, kernel addresses/routes/neighbours, persisted config md5."""
    return {"hostname": duthost.hostname, "taken_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config_db": gu.dump_config_db(duthost), "kernel": kernel_state(duthost, devices),
            "persisted_config_md5": backup_persisted_config(duthost)}


def save_snapshot(snapshot, label):
    """Persist the snapshot on the test server (survives a pytest crash); returns the path."""
    directory = os.path.join(tempfile.gettempdir(), "sonic-mgmt-dpb-snapshots")
    if not os.path.isdir(directory):
        os.makedirs(directory)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(directory, "{}_{}_{}.json".format(snapshot.get("hostname", "dut"), label, stamp))
    with open(path, "w") as handle:
        json.dump(snapshot, handle, indent=1, sort_keys=True)
    return path


def latest_snapshot_path(hostname):
    """Newest snapshot file saved by :func:`save_snapshot` for ``hostname`` (None if there is none)."""
    directory = os.path.join(tempfile.gettempdir(), "sonic-mgmt-dpb-snapshots")
    if not os.path.isdir(directory):
        return None
    candidates = sorted(f for f in os.listdir(directory) if f.startswith(hostname + "_") and f.endswith(".json"))
    return os.path.join(directory, candidates[-1]) if candidates else None


def load_snapshot(path):
    with open(path) as handle:
        return json.load(handle)


def persisted_cage_layout(duthost, master):
    """``(brkout_mode, [ports])`` of ``master``'s cage in the persisted ``config_db.json`` (what a stack
    restart restores), or ``(None, [])`` if the file cannot be read."""
    script = ("python3 -c \"import json,sys; d=json.load(open(sys.argv[1])); m=sys.argv[2]; "
              "print(d.get('BREAKOUT_CFG',{{}}).get(m,{{}}).get('brkout_mode','')); "
              "print(','.join(sorted(p for p,f in d.get('BREAKOUT_PORTS',{{}}).items() "
              "if f.get('master')==m and p in d.get('PORT',{{}}))))\" {} {}").format(
                  shlex.quote(PERSISTED_CONFIG_DB), shlex.quote(master))
    out = gu.shell_stdout(duthost, script)
    if out is None:
        return None, []
    lines = out.splitlines() + ["", ""]
    return (lines[0].strip() or None), [p for p in lines[1].strip().split(",") if p]


def leftover_breakout(duthost, master):
    """Describe a cage that is NOT in its persisted layout (a previous run did not recover), else None."""
    persisted_mode, persisted_ports = persisted_cage_layout(duthost, master)
    if persisted_mode is None:
        return None
    live_mode = get_breakout_cfg(duthost, master).get("brkout_mode")
    live_ports = sorted(current_cage_ports(duthost, master))
    if live_mode == persisted_mode and live_ports == sorted(persisted_ports):
        return None
    return "cage {} is {} with ports {} but config_db.json says {} with {}".format(
        master, live_mode, live_ports, persisted_mode, persisted_ports)


def platform_port_tables(config_db):
    return {k: v for k, v in config_db.items() if k.startswith(PLATFORM_PORT_TABLES)}


def port_layout_diff(snapshot_cfg, current_cfg):
    """``{key: (saved, current)}`` for PORT/BREAKOUT_* keys that differ (``{}`` when the layout is back)."""
    saved = platform_port_tables(snapshot_cfg)
    current = platform_port_tables(current_cfg)
    return {k: (saved.get(k), current.get(k)) for k in set(saved) | set(current) if saved.get(k) != current.get(k)}


def restore_config_db_from_snapshot(duthost, saved_cfg):
    """Make CONFIG_DB equal to ``saved_cfg`` except for the platform port tables; returns ``(rewritten, deleted)``."""
    current = gu.dump_config_db(duthost)
    commands, rewritten, deleted = [], [], []
    for key in sorted(saved_cfg):
        if key.startswith(PLATFORM_PORT_TABLES):
            continue
        fields = saved_cfg[key] or {}
        if current.get(key) == fields:
            continue
        flat = []
        for field, value in (fields or {"NULL": "NULL"}).items():
            flat.extend([field, value])
        commands.append(["DEL", key])
        commands.append(["HSET", key] + flat)
        rewritten.append(key)
    for key in sorted(current):
        if key.startswith(PLATFORM_PORT_TABLES) or key in saved_cfg:
            continue
        commands.append(["DEL", key])
        deleted.append(key)
    for start in range(0, len(commands), 200):
        gu.redis_pipeline(duthost, "CONFIG_DB", commands[start:start + 200])
    logger.info("CONFIG_DB restored on %s: %d keys rewritten %s, %d deleted %s", duthost.hostname, len(rewritten),
                rewritten[:20], len(deleted), deleted[:20])
    return rewritten, deleted


def restore_kernel_state(duthost, kernel):
    for dev, cidrs in (kernel.get("addrs") or {}).items():
        for cidr in cidrs:
            duthost.shell("ip -6 addr replace {} dev {}".format(shlex.quote(cidr), shlex.quote(dev)),
                          module_ignore_errors=True, verbose=False)
    for line in kernel.get("neighs") or []:
        match = re.match(r"^(\S+) dev (\S+) lladdr (\S+)", line)
        if match:
            duthost.shell("ip -6 neigh replace {} lladdr {} dev {} nud permanent".format(
                shlex.quote(match.group(1)), shlex.quote(match.group(3)), shlex.quote(match.group(2))),
                module_ignore_errors=True, verbose=False)
    for line in kernel.get("routes") or []:
        duthost.shell("ip -6 route replace {}".format(line), module_ignore_errors=True, verbose=False)


# ---------------------------------------------------------------------------
# Recovery: gpins-stack restart
# ---------------------------------------------------------------------------

def is_nsf_boot(duthost):
    """True if the NSF marker is set (``redis-container`` would then skip loading ``config_db.json``)."""
    res = duthost.shell("{} is-nsf-boot".format(gu.GPINS_NSF_SCRIPT), module_ignore_errors=True, verbose=False)
    return res.get("rc") == 0


def stack_core_ready(duthost):
    """``db-con``/``or-con``/``sy-con`` RUNNING and ``APPL_DB PORT_TABLE:PortInitDone`` present."""
    out = gu.shell_stdout(duthost, "{0} -n {1} --raw EXISTS PORT_TABLE:PortInitDone && {2} tasks ls".format(
        gu.REDIS_CLI, gu.db_id("APPL_DB"), gu.CTR))
    if out is None:
        return False
    lines = out.splitlines()
    if not lines or lines[0].strip() != "1":
        return False
    tasks = gu.parse_container_tasks("\n".join(lines[1:]))
    return all(tasks.get(c) == "RUNNING" for c in gu.NSF_CORE_CONTAINERS)


def gpins_stack_restart(duthost, timeout=480, stop_timeout=180):
    """``/etc/init.d/gpins-stack restart`` (detached) and wait until the core stack is back.

    The script returns after ``stop_all`` with ``start_all`` running in the background, so it is
    launched detached and the stack is polled: first until it goes down, then until ``PortInitDone``
    reappears (measured ~90 s on ju4u1m1). The GPINS host has no ``setsid`` (run 3 on ju3u1m1: the
    restart never ran and the "recovery" passed on the untouched stack), so the launch uses a nohup'ed
    subshell, the launch log is checked, and a stack that does not go down is a hard failure.
    """
    gu.ensure_lab_ca_cert(duthost)
    logger.warning("Restarting the GPINS stack on %s via %s restart (log: %s)", duthost.hostname,
                   GPINS_STACK_SCRIPT, STACK_RESTART_LOG)
    started = time.time()
    res = duthost.shell("rm -f {log}; (nohup {script} restart > {log} 2>&1 < /dev/null &); sleep 3; "
                        "head -c 2000 {log}".format(script=GPINS_STACK_SCRIPT, log=STACK_RESTART_LOG),
                        module_ignore_errors=True)
    launch_log = (res.get("stdout") or "").strip()
    if "command not found" in launch_log or "No such file" in launch_log:
        raise RuntimeError("'{} restart' could not be launched on {}: {!r}".format(
            GPINS_STACK_SCRIPT, duthost.hostname, launch_log[:300]))
    if not wait_until(stop_timeout, 5, 0, lambda: not stack_core_ready(duthost)):
        tail = gu.shell_stdout(duthost, "tail -n 15 {}".format(STACK_RESTART_LOG)) or ""
        raise RuntimeError("GPINS stack on {} never went down within {}s after '{} restart'; the port layout was NOT "
                           "reloaded. Restart log tail: {!r}".format(duthost.hostname, stop_timeout, GPINS_STACK_SCRIPT,
                                                                     tail[-800:]))
    logger.info("GPINS stack on %s is down after %.0f s; waiting for it to come back", duthost.hostname,
                time.time() - started)
    ok = wait_until(timeout, 10, 0, stack_core_ready, duthost)
    logger.info("GPINS stack on %s %s after %.0f s (tasks: %s)", duthost.hostname, "ready" if ok else "NOT ready",
                time.time() - started, gu.shell_stdout(duthost, "{} tasks ls".format(gu.CTR)))
    if not ok:
        raise RuntimeError("GPINS stack on {} not ready {}s after restart; see {} on the switch".format(
            duthost.hostname, timeout, STACK_RESTART_LOG))


def recovery_step(errors, label, func, *args, **kwargs):
    """Run one recovery step; record (not raise) its failure so the following steps still run."""
    try:
        return func(*args, **kwargs)
    except Exception as exc:  # recovery must try everything
        logger.exception("DPB recovery step %r failed", label)
        errors.append("{}: {}".format(label, exc))
        return None


def recover_switch_state(duthost, snapshot, settings):
    """Bring the switch back to ``snapshot``: platform restart (port layout), CONFIG_DB diff, kernel state.

    Every step is attempted even if an earlier one failed. Returns ``(port_layout_diff, errors)``:
    both empty on success.
    """
    recovery = settings.recovery
    errors = []
    if recovery == "stack-restart":
        if recovery_step(errors, "nsf marker check", is_nsf_boot, duthost):
            errors.append("NSF boot marker is set: gpins-stack restart will NOT reload config_db.json")
        recovery_step(errors, "persisted config check", restore_persisted_config_if_changed, duthost,
                      snapshot.get("persisted_config_md5"))
        recovery_step(errors, "gpins-stack restart", gpins_stack_restart, duthost)
    else:
        logger.warning("DPB recovery mode %r: leaving the port layout as is on %s", recovery, duthost.hostname)
    recovery_step(errors, "CONFIG_DB restore", restore_config_db_from_snapshot, duthost, snapshot["config_db"])
    recovery_step(errors, "kernel state restore", restore_kernel_state, duthost, snapshot["kernel"])
    diff = recovery_step(errors, "port layout check", lambda: port_layout_diff(snapshot["config_db"],
                                                                               gu.dump_config_db(duthost)))
    if diff is None:
        diff = {"<unknown>": None}
    elif diff and recovery == "stack-restart":
        logger.error("Port layout on %s still differs from the pre-DPB snapshot after recovery: %s", duthost.hostname,
                     sorted(diff))
    return diff, errors


def switch_health(duthost):
    """Post-recovery sanity: core + relay + telemetry containers RUNNING, ``PortInitDone`` back, and the gNMI
    server back behind mTLS, i.e. ``TELEMETRY|certs`` present and the server started with ``--ca_crt``.
    Returns a list of problems (empty = healthy enough for the next test)."""
    problems = []
    try:
        tasks = gu.get_container_tasks(duthost)
    except Exception as exc:
        return ["cannot list containers: {}".format(exc)]
    for name in list(gu.NSF_CORE_CONTAINERS) + [gu.TELEMETRY_CONTAINER, gu.DHCP_RELAY_CONTAINER]:
        if tasks.get(name) != "RUNNING":
            problems.append("{} is {}".format(name, tasks.get(name, "absent")))
    state = gu.gnmi_server_access_state(duthost)
    if not state[gu.TELEMETRY_CERTS_ROW]:
        problems.append("CONFIG_DB {} is missing (gNMI mTLS not restored)".format(gu.TELEMETRY_CERTS_ROW))
    elif state["cmdline"] and not gu.telemetry_mtls_enabled(state["cmdline"]):
        problems.append("gNMI server still runs without mTLS: {}".format(state["cmdline"]))
    if not stack_core_ready(duthost):
        problems.append("APPL_DB PORT_TABLE:PortInitDone missing or core containers not RUNNING")
    return problems


# ---------------------------------------------------------------------------
# Breakout verification
# ---------------------------------------------------------------------------

def breakout_state(duthost, names):
    """Per port: kernel netdev present, CONFIG_DB lanes/speed/admin_status/mtu, APPL_DB/STATE_DB PORT_TABLE
    presence and kernel operstate (one round trip)."""
    script = []
    for name in names:
        kdev = _kernel_dev(name)
        script.append(
            "echo '==PORT {n}'; test -e /sys/class/net/{k} && echo kernel=1 || echo kernel=0; "
            "echo lanes=$({r} -n {cfg} --raw HGET 'PORT|{n}' lanes); "
            "echo speed=$({r} -n {cfg} --raw HGET 'PORT|{n}' speed); "
            "echo admin=$({r} -n {cfg} --raw HGET 'PORT|{n}' admin_status); "
            "echo mtu=$({r} -n {cfg} --raw HGET 'PORT|{n}' mtu); "
            "echo appl=$({r} -n {appl} --raw EXISTS 'PORT_TABLE:{n}'); "
            "echo state=$({r} -n {state} --raw EXISTS 'PORT_TABLE|{n}'); "
            "echo oper=$(cat /sys/class/net/{k}/operstate 2>/dev/null)".format(
                n=name, k=kdev, r=gu.REDIS_CLI, cfg=gu.db_id("CONFIG_DB"), appl=gu.db_id("APPL_DB"),
                state=gu.db_id("STATE_DB")))
    out = gu.shell_stdout(duthost, "; ".join(script)) or ""
    result, current = {}, None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("==PORT "):
            current = line[len("==PORT "):]
            result[current] = {"kernel": False, "lanes": "", "speed": "", "admin": "", "mtu": "", "appl": False,
                               "state": False, "oper": ""}
        elif current and "=" in line:
            field, _, value = line.partition("=")
            value = value.strip()
            if field in ("kernel", "appl", "state"):
                result[current][field] = value == "1"
            elif field in ("lanes", "speed", "admin", "mtu", "oper"):
                result[current][field] = value
    return result


def breakout_applied(duthost, plan):
    """True once every expected port exists with the expected lane count/speed in kernel + CONFIG_DB + APPL_DB
    and every removed port is gone."""
    state = breakout_state(duthost, plan.expected_names + plan.removed_ports)
    for port in plan.expected_ports:
        seen = state.get(port["name"], {})
        lanes = [x for x in seen.get("lanes", "").split(",") if x]
        if not (seen.get("kernel") and seen.get("appl") and len(lanes) == port["num_lanes"]
                and seen.get("speed") == str(port["speed_gbps"] * 1000)):
            return False
    for name in plan.removed_ports:
        seen = state.get(name, {})
        if seen.get("kernel") or seen.get("appl") or seen.get("lanes"):
            return False
    return True


def describe_breakout_state(duthost, plan):
    cfg = get_breakout_cfg(duthost, plan.master)
    return "BREAKOUT_CFG|{}={} ports={}".format(plan.master, cfg, breakout_state(
        duthost, plan.expected_names + plan.removed_ports))


def wait_breakout_applied(duthost, plan, timeout=180, interval=10):
    return wait_until(timeout, interval, 5, breakout_applied, duthost, plan)


def layout_unchanged(duthost, plan):
    """True if the cage still has exactly its pre-DPB ports (used to skip the heavy recovery)."""
    state = breakout_state(duthost, sorted(set(plan.current_ports) | set(plan.expected_names)))
    for name, before in plan.current_ports.items():
        seen = state.get(name, {})
        if (not seen.get("kernel") or seen.get("lanes") != before.get("lanes", "")
                or seen.get("speed") != str(before.get("speed", ""))):
            return False
    for name in plan.new_ports:
        if state.get(name, {}).get("kernel") or state.get(name, {}).get("lanes"):
            return False
    return get_breakout_cfg(duthost, plan.master).get("brkout_mode") == plan.current_mode


# ---------------------------------------------------------------------------
# Relay-port preparation shared by parent and children
# ---------------------------------------------------------------------------

def test_gua_for(port):
    _, port_num, lane = slot_port_lane(port)
    return DEFAULT_TEST_GUA_TEMPLATE.format(port=port_num, lane=lane)


def prepare_relay_port(duthost, port, mode):
    """Make ``port`` a valid relay interface for ``mode``: ``l3_lla`` -> RIF row + no GUA; ``l3_gua`` -> test GUA."""
    mode = gu.normalize_relay_mode(mode)
    if mode == gu.RELAY_MODE_L3_LLA:
        for cidr in gu.get_port_kernel_guas(duthost, port):
            gu.del_interface_ipv6(duthost, port, cidr)
        gu.ensure_l3_interface(duthost, port)
        return []
    if mode == gu.RELAY_MODE_L3_GUA:
        addr, plen = gu.get_interface_ipv6(duthost, port)
        cidr = "{}/{}".format(addr, plen) if addr else test_gua_for(port)
        return gu.set_interface_ipv6(duthost, port, cidr)
    raise ValueError("DPB test supports l3_lla / l3_gua only, got {!r}".format(mode))

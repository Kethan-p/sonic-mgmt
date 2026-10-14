"""GPINS helper layer for the DHCP relay tests.

Upstream SONiC tests talk to the switch through SONiC CLI tools
(``sonic-db-cli``, ``config ...``, ``show ...``), ``systemctl`` and Docker.
GPINS does not ship those tools, so every switch interaction the DHCPv6
relay tests need goes through the helpers in this module instead:

* Redis access uses plain ``redis-cli`` (a wrapper that runs redis-cli inside
  the ``db-con`` container) with numeric DB indexes and ``--raw`` output.
* Configuration is written directly into CONFIG_DB.
* Containers are managed by containerd (``ctr``), not Docker. Each container
  ``<name>-con`` has an init script ``/etc/init.d/<name>-container``.

All GPINS-specific names (DB numbers, commands, container names) live in
the constants below so they can be adjusted in one place.
"""

import csv
import ipaddress
import logging
import os
import re
import shlex
import time

from tests.common.utilities import wait_until

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SONiC Redis database indexes (see database_config.json on the switch).
DB_IDS = {
    "APPL_DB": 0,
    "ASIC_DB": 1,
    "COUNTERS_DB": 2,
    "LOGLEVEL_DB": 3,
    "CONFIG_DB": 4,
    "FLEX_COUNTER_DB": 5,
    "STATE_DB": 6,
    "APPL_STATE_DB": 14,
}

# Key separator per DB. CONFIG_DB and STATE_DB use '|', APPL_DB and
# COUNTERS_DB use ':'.
DB_SEPARATORS = {
    "APPL_DB": ":",
    "ASIC_DB": ":",
    "COUNTERS_DB": ":",
    "LOGLEVEL_DB": ":",
    "CONFIG_DB": "|",
    "FLEX_COUNTER_DB": "|",
    "STATE_DB": "|",
    "APPL_STATE_DB": ":",
}

# redis-cli on GPINS is /usr/tools/bin/redis-cli, a wrapper around
# `ctr tasks exec ... db-con redis-cli`. /usr/tools/bin is only on PATH in an
# interactive login shell, so the absolute path is required for commands run
# over SSH (verified on ju5u1m1.sqs04, 2026-10-01). It is redis-cli 6.2, so
# --json is NOT supported; we always use --raw.
# TODO(GPINS, Virgo testbed): check whether the DHCP-enabled build exposes a
# host-side socket (`test -S /var/run/redis/redis.sock && command -v redis-cli`).
# If it does, add auto-detection here and prefer `redis-cli -s <sock>` (much
# faster than a ctr exec per call). Do not change this until verified there.
REDIS_CLI = "/usr/tools/bin/redis-cli"

# containerd CLI used to inspect and exec into GPINS containers. Tests log in
# as root; there is no sudo binary on the switch.
CTR = "ctr"

# Verified on ju3u1m1.sqs901 running zhouboyang_dhcp_test (2026-10-08):
# /etc/init.d/dhcp-relay-container sets DHCP_RELAY_CONTAINER_NAME="dhcp-relay-con"
# (via common-container.sh) and runs ctr image docker-dhcp-relay:latest.
DHCP_RELAY_CONTAINER = "dhcp-relay-con"
DHCP_RELAY_INIT_SCRIPT = "/etc/init.d/dhcp-relay-container"
# common-container.sh operations (verified 2026-10-08): "start" imports the image
# on demand and launches the container (~1.5 s), "stop" removes task+container,
# "restart" = stop + start. Nothing on the switch watches FEATURE to do this.
DHCP_RELAY_START_CMD = "{script} start"
DHCP_RELAY_STOP_CMD = "{script} stop"
DHCP_RELAY_RESTART_CMD = "{script} restart"

# FEATURE|dhcp_relay semantics on GPINS (Boyang, 2026-10-08): the flag does NOT
# start/stop the container (hostcfgd has no feature handler), but the relay
# manager inside the container only brings up dhcp6relay while state=enabled.
# Tests therefore set the flag AND make sure the container is running.
DHCP_RELAY_FEATURE_KEY = "FEATURE|dhcp_relay"

# Inside dhcp-relay-con, supervisord runs the managers dhcprelayd/dhcprelaydv6;
# dhcprelaydv6 spawns /usr/sbin/dhcp6relay (<=5 s) once a valid DHCP_RELAY|<intf>
# entry exists and stops it (<=10 s) when the last one is removed. Verified
# 2026-10-08 (`pgrep -x dhcp6relay` from the host PID namespace).
DHCP6RELAY_PROCESS = "dhcp6relay"
DHCPV6_SERVER_PORT = 547

# CONFIG_DB port names are "Ethernet1/1/1" but the Linux netdevs are
# "Ethernet1_1_1" (DEVICE_METADATA|intf_translate=enabled). Use these when
# matching redis names against `ip addr` / /proc output.


def to_kernel_ifname(port):
    """CONFIG_DB port name -> Linux netdev name (Ethernet1/1/1 -> Ethernet1_1_1)."""
    return port.replace("/", "_")


def to_config_ifname(ifname):
    """Linux netdev name -> CONFIG_DB port name (Ethernet1_1_1 -> Ethernet1/1/1)."""
    return ifname.replace("_", "/")


# CONFIG_DB tables that describe DHCPv6 relay and interface configuration.
DEFAULT_RUNNING_CONFIG_TABLES = (
    "DEVICE_METADATA",
    "FEATURE",
    "DHCP_RELAY",
    "VLAN",
    "VLAN_MEMBER",
    "VLAN_INTERFACE",
    "INTERFACE",
    "PORTCHANNEL",
    "PORTCHANNEL_MEMBER",
    "PORTCHANNEL_INTERFACE",
    "LOOPBACK_INTERFACE",
    "PORT",
)

# DHCP_RELAY options that are booleans in CONFIG_DB ("true"/"false").
_DHCP_RELAY_BOOL_FIELDS = ("rfc6939_support", "enable_interface_id", "enable_remote_id")


# ---------------------------------------------------------------------------
# Low-level Redis access
# ---------------------------------------------------------------------------

def _db_id(db):
    """Return the numeric Redis index for a DB given by name or number."""
    if isinstance(db, int):
        return db
    try:
        return DB_IDS[db.upper()]
    except KeyError:
        raise ValueError("Unknown Redis DB '{}'. Known DBs: {}".format(db, sorted(DB_IDS)))


def _db_separator(db):
    """Return the key separator used by the given DB."""
    if isinstance(db, int):
        for name, idx in DB_IDS.items():
            if idx == db:
                return DB_SEPARATORS[name]
        return "|"
    return DB_SEPARATORS.get(db.upper(), "|")


def build_redis_cmd(db, *args):
    """Build (but do not run) a redis-cli command line.

    ``--raw`` is always passed so the output is one plain value per line
    even if the command runs with a TTY. The GPINS redis-cli (6.2) does not
    support ``--json``.

    Every argument is shell-quoted, so keys like ``DHCP_RELAY|Ethernet1_21_5``
    and values containing spaces or brackets are passed through verbatim.
    """
    parts = [REDIS_CLI, "-n", str(_db_id(db)), "--raw"]
    parts.extend(shlex.quote(str(arg)) for arg in args)
    return " ".join(parts)


# stderr fragments from the ctr-based redis-cli wrapper that indicate a
# transient containerd hiccup rather than a Redis error. Seen on ju5u1m1.sqs04
# (2026-10-02): ~1 in 200 back-to-back calls failed with "cannot exec in a
# stopped state" and succeeded immediately on retry.
_TRANSIENT_CTR_ERRORS = (
    "cannot exec in a stopped state",
    "exec-id already exists",
    "resize pty error",
    "transport is closing",
    "connection reset",
)
REDIS_RETRIES = 3
REDIS_RETRY_DELAY = 0.5


def _is_transient_ctr_error(res):
    text = "{} {}".format(res.get("stderr", ""), res.get("stdout", "")).lower()
    return any(pattern in text for pattern in _TRANSIENT_CTR_ERRORS)


def redis_cmd_lines(duthost, db, *args, module_ignore_errors=False):
    """Run a redis-cli command on the DUT and return its stdout as a list of lines.

    Only stdout is parsed; the ``ctr`` wrapper may print noise such as
    ``ERRO[0000] resize pty ...`` which is not part of the result.
    Empty lines are kept, because they are meaningful (e.g. an empty hash value).

    Transient containerd exec failures are retried ``REDIS_RETRIES`` times.
    """
    cmd = build_redis_cmd(db, *args)
    res = None
    for attempt in range(1, REDIS_RETRIES + 1):
        res = duthost.shell(cmd, module_ignore_errors=True, verbose=False)
        if res.get("rc", 0) == 0:
            break
        if attempt < REDIS_RETRIES and _is_transient_ctr_error(res):
            logger.warning("redis-cli transient failure (attempt %d/%d): %s -- %s",
                           attempt, REDIS_RETRIES, cmd, res.get("stderr", "").strip())
            time.sleep(REDIS_RETRY_DELAY * attempt)
            continue
        if not module_ignore_errors:
            raise RuntimeError("redis-cli failed (rc={}): {}\nstderr: {}".format(
                res.get("rc"), cmd, res.get("stderr", "").strip()))
        break
    return [line for line in res.get("stdout", "").splitlines() if not line.startswith("ERRO[")]


def redis_cmd(duthost, db, *args, module_ignore_errors=False):
    """Run a redis-cli command on the DUT and return its stripped stdout.

    Replacement for ``sonic-db-cli <DB_NAME> <args...>``.

    Example:
        redis_cmd(duthost, "STATE_DB", "HGET", "DHCPv6_COUNTER_TABLE|Ethernet0", "Solicit")
    """
    lines = redis_cmd_lines(duthost, db, *args, module_ignore_errors=module_ignore_errors)
    return "\n".join(lines).strip()


def redis_hgetall(duthost, db, key):
    """Return all fields of a Redis hash as a dict ({} if the key is missing).

    In ``--raw`` mode HGETALL prints field and value on alternating lines.
    Values that contain newlines are not supported; SONiC tables don't use them.
    """
    lines = redis_cmd_lines(duthost, db, "HGETALL", key)
    # A missing key prints a single empty line.
    if lines == [""]:
        return {}
    if len(lines) % 2:
        # Ansible strips trailing whitespace, which drops an empty final value.
        lines.append("")
    return dict(zip(lines[0::2], lines[1::2]))


def redis_hget(duthost, db, key, field):
    """Return one field of a Redis hash, or None if it does not exist."""
    value = redis_cmd(duthost, db, "HGET", key, field)
    return value if value else None


def redis_keys(duthost, db, pattern):
    """Return the list of keys matching a glob pattern."""
    return sorted(line for line in redis_cmd_lines(duthost, db, "KEYS", pattern) if line)


def redis_exists(duthost, db, key):
    """Return True if the key exists."""
    return redis_cmd(duthost, db, "EXISTS", key) == "1"


def redis_hset(duthost, db, key, fields):
    """Set one or more fields of a Redis hash.

    An empty ``fields`` dict writes the SONiC placeholder ``NULL NULL``, which
    is how field-less entries (e.g. ``INTERFACE|Ethernet0|fe80::1/64``) are
    stored.
    """
    if not fields:
        fields = {"NULL": "NULL"}
    args = []
    for field, value in fields.items():
        args.extend([field, value])
    redis_cmd(duthost, db, "HSET", key, *args)


def redis_hdel(duthost, db, key, *fields):
    """Delete one or more fields from a Redis hash."""
    if fields:
        redis_cmd(duthost, db, "HDEL", key, *fields)


def redis_del(duthost, db, *keys):
    """Delete one or more keys."""
    if keys:
        redis_cmd(duthost, db, "DEL", *keys)


# One-shot table fetch: KEYS + HGETALL for every match inside Redis (Lua), so a
# 66-entry table costs one ssh/ctr round trip instead of 67. Output format
# (one element per line with --raw):  "==KEY <key>", "<n>", then n lines of
# alternating field/value. The explicit count keeps empty values unambiguous
# even though ansible strips trailing whitespace from stdout.
_GET_TABLE_LUA = ("local out = {} "
                  "for _, k in ipairs(redis.call('KEYS', ARGV[1])) do "
                  "out[#out+1] = '==KEY ' .. k "
                  "local h = redis.call('HGETALL', k) "
                  "out[#out+1] = tostring(#h) "
                  "for _, v in ipairs(h) do out[#out+1] = v end "
                  "end "
                  "return out")


def _parse_table_dump(lines, prefix):
    entries = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("==KEY "):
            i += 1
            continue
        key = line[len("==KEY "):]
        count = int(lines[i + 1]) if i + 1 < len(lines) and lines[i + 1].isdigit() else 0
        values = lines[i + 2:i + 2 + count]
        values += [""] * (count - len(values))      # trailing empty value stripped by ansible
        entries[key[len(prefix):] if key.startswith(prefix) else key] = dict(zip(values[0::2], values[1::2]))
        i += 2 + count
    return entries


def get_table(duthost, db, table):
    """Return every entry of a table as {key_without_table_prefix: fields}.

    Example: get_table(duthost, "CONFIG_DB", "DHCP_RELAY") ->
        {"Vlan1000": {"dhcpv6_servers": "fc02:2000::1,fc02:2000::2"}}

    Uses a single EVAL round trip; falls back to KEYS + one HGETALL per key
    if the server rejects EVAL (older/locked-down redis).
    """
    sep = _db_separator(db)
    prefix = table + sep
    cmd = build_redis_cmd(db, "EVAL", _GET_TABLE_LUA, "0", prefix + "*")
    res = duthost.shell(cmd, module_ignore_errors=True, verbose=False)
    lines = [line for line in res.get("stdout", "").splitlines() if not line.startswith("ERRO[")]
    if res.get("rc", 1) == 0 and not (lines and lines[0].startswith("ERR")):
        return _parse_table_dump(lines, prefix)
    logger.debug("EVAL table fetch unavailable (rc=%s %r); falling back to per-key HGETALL",
                 res.get("rc"), (res.get("stderr") or "")[:120])
    entries = {}
    for key in redis_keys(duthost, db, prefix + "*"):
        entries[key[len(prefix):]] = redis_hgetall(duthost, db, key)
    return entries


# ---------------------------------------------------------------------------
# CONFIG_DB helpers (replace `config ...` CLI and `config_facts`)
# ---------------------------------------------------------------------------

def config_db_set(duthost, key, fields=None):
    """Create or update a CONFIG_DB entry. Replacement for `config ...` commands."""
    redis_hset(duthost, "CONFIG_DB", key, fields or {})


def config_db_del(duthost, key):
    """Delete a CONFIG_DB entry."""
    redis_del(duthost, "CONFIG_DB", key)


def get_running_config(duthost, tables=DEFAULT_RUNNING_CONFIG_TABLES):
    """Return {table: {key: fields}} read directly from CONFIG_DB.

    Replacement for ``duthost.config_facts(source="running")``, which depends
    on sonic-cfggen and is not available on GPINS.
    """
    return {table: get_table(duthost, "CONFIG_DB", table) for table in tables}


def snapshot_config_db(duthost, patterns):
    """Save every CONFIG_DB entry matching the given key patterns.

    Use together with ``restore_config_db`` in test setup/teardown.

    Returns:
        dict: {"patterns": [...], "entries": {key: fields}}
    """
    entries = {}
    for pattern in patterns:
        for key in redis_keys(duthost, "CONFIG_DB", pattern):
            entries[key] = redis_hgetall(duthost, "CONFIG_DB", key)
    return {"patterns": list(patterns), "entries": entries}


def restore_config_db(duthost, snapshot):
    """Restore CONFIG_DB entries saved by ``snapshot_config_db``.

    Keys that match the snapshot patterns but did not exist at snapshot time
    are deleted, and saved keys are rewritten with exactly their saved fields.
    """
    saved = snapshot["entries"]
    current = set()
    for pattern in snapshot["patterns"]:
        current.update(redis_keys(duthost, "CONFIG_DB", pattern))

    extra = sorted(current - set(saved))
    if extra:
        logger.info("Removing CONFIG_DB keys added during test: %s", extra)
        redis_del(duthost, "CONFIG_DB", *extra)

    for key, fields in saved.items():
        current_fields = redis_hgetall(duthost, "CONFIG_DB", key) if key in current else {}
        if current_fields == fields:
            continue
        # Rewrite the entry so fields added during the test are dropped too.
        redis_del(duthost, "CONFIG_DB", key)
        redis_hset(duthost, "CONFIG_DB", key, fields)


def feature_state(duthost, feature):
    """Return FEATURE|<feature> 'state' from CONFIG_DB (replacement for `show feature status`)."""
    return redis_hget(duthost, "CONFIG_DB", "FEATURE|{}".format(feature), "state")


def set_feature_state(duthost, feature, state):
    """Set FEATURE|<feature> 'state' ("enabled" / "disabled") in CONFIG_DB."""
    config_db_set(duthost, "FEATURE|{}".format(feature), {"state": state})


def dhcp_relay_available(duthost):
    """Return True if this GPINS build ships the DHCP-relay feature.

    The init script is the authoritative indicator: builds without the relay
    (e.g. gpins_release_20260824_05_prod_RC03) have no
    /etc/init.d/dhcp-relay-container. The FEATURE|dhcp_relay row is only
    informational on GPINS and is not required here.
    """
    res = duthost.shell("test -x {}".format(DHCP_RELAY_INIT_SCRIPT), module_ignore_errors=True)
    return res.get("rc", 1) == 0


# ---------------------------------------------------------------------------
# DHCP_RELAY table helpers
# ---------------------------------------------------------------------------

def _bool_str(value):
    return "true" if value else "false"


def build_dhcp_relay_fields(dhcpv6_servers, rfc6939_support=None, enable_interface_id=None,
                            interface_id=None, enable_remote_id=None, remote_id=None):
    """Build the field dict for a DHCP_RELAY|<intf> entry.

    Options left as None are not written, so the relay uses its defaults.
    Field names follow the GPINS schema in the Virgo-Bloom DHCP-relay design:
      * rfc6939_support      - option 79 on/off
      * enable_interface_id  - option 18 on/off
      * interface_id         - option 18 value, e.g. "[2001:db8::1]:Ethernet1/1/1"
      * enable_remote_id     - option 37 on/off
      * remote_id            - option 37 value
    """
    if isinstance(dhcpv6_servers, str):
        dhcpv6_servers = [dhcpv6_servers]
    if not dhcpv6_servers:
        raise ValueError("At least one DHCPv6 server address is required")

    fields = {"dhcpv6_servers": ",".join(str(s) for s in dhcpv6_servers)}
    options = {
        "rfc6939_support": rfc6939_support,
        "enable_interface_id": enable_interface_id,
        "interface_id": interface_id,
        "enable_remote_id": enable_remote_id,
        "remote_id": remote_id,
    }
    for name, value in options.items():
        if value is None:
            continue
        if name in _DHCP_RELAY_BOOL_FIELDS:
            fields[name] = _bool_str(value)
        else:
            fields[name] = str(value)

    if fields.get("enable_remote_id") == "true" and not fields.get("remote_id"):
        raise ValueError("remote_id must be set and non-empty when enable_remote_id is true")
    return fields


def set_dhcp_relay_config(duthost, interface, dhcpv6_servers, replace=True, relay_mode=None, **options):
    """Enable DHCPv6 relay on an interface by writing CONFIG_DB DHCP_RELAY|<interface>.

    Args:
        interface: VLAN or L3 interface name in CONFIG_DB form, e.g. "Vlan1000"
            or "Ethernet1/21/5". The key uses the *slash* form, the same as the
            PORT table (confirmed with Boyang 2026-10-09; the design doc's
            ``Ethernet1_21_5`` example is the kernel netdev name, not the key).
        dhcpv6_servers: server address or list of addresses.
        replace: if True, delete the existing entry first so stale options
            from a previous test do not leak into this one.
        relay_mode: optional relay mode ('vlan'/'svi', 'l3_lla', 'l3_gua'); when
            given, asserts ``interface`` matches that mode (e.g. has no GUA in
            ``l3_lla`` mode) before writing ``DHCP_RELAY|<interface>``.
        **options: see ``build_dhcp_relay_fields``. On an interface without a
            GUA (l3_lla) the relay manager rejects the entry unless
            ``enable_interface_id=True`` and ``interface_id`` is locally unique;
            a rejected entry never appears in STATE_DB.
    """
    if relay_mode is not None:
        verify_interface_mode(duthost, interface, relay_mode)
    key = "DHCP_RELAY|{}".format(interface)
    fields = build_dhcp_relay_fields(dhcpv6_servers, **options)
    if replace:
        redis_del(duthost, "CONFIG_DB", key)
    logger.info("Setting %s: %s", key, fields)
    redis_hset(duthost, "CONFIG_DB", key, fields)
    return fields


def del_dhcp_relay_config(duthost, interface):
    """Remove the DHCP_RELAY|<interface> entry from CONFIG_DB."""
    config_db_del(duthost, "DHCP_RELAY|{}".format(interface))


def get_dhcp_relay_config(duthost, interface):
    """Return CONFIG_DB DHCP_RELAY|<interface> fields (desired config)."""
    return redis_hgetall(duthost, "CONFIG_DB", "DHCP_RELAY|{}".format(interface))


def get_dhcp_relay_state(duthost, interface):
    """Return STATE_DB DHCP_RELAY|<interface> fields (config applied by dhcp6relay)."""
    return redis_hgetall(duthost, "STATE_DB", "DHCP_RELAY|{}".format(interface))


def dhcp_relay_config_converged(duthost, interface):
    """Return True if STATE_DB DHCP_RELAY|<intf> matches CONFIG_DB DHCP_RELAY|<intf>.

    Only fields present in CONFIG_DB are compared, because STATE_DB may also
    report defaults for fields that were not configured.
    """
    desired = get_dhcp_relay_config(duthost, interface)
    applied = get_dhcp_relay_state(duthost, interface)
    if not desired:
        return not applied
    return all(applied.get(field) == value for field, value in desired.items())


def get_dhcpv6_counters(duthost, interface):
    """Return the raw STATE_DB DHCPv6_COUNTER_TABLE|<interface> fields."""
    return redis_hgetall(duthost, "STATE_DB", "DHCPv6_COUNTER_TABLE|{}".format(interface))


# ---------------------------------------------------------------------------
# ACL helpers (replace `config acl add table` / `acl-loader`)
#
# Observed on ju5u1m1: orchagent consumes CONFIG_DB ACL_TABLE|<name> and
# reports STATE_DB ACL_TABLE_TABLE|<name> status=Active (likewise
# ACL_RULE_TABLE|<table>|<rule>). That status is used as the convergence signal.
# ---------------------------------------------------------------------------

ACL_STATUS_ACTIVE = "Active"
ETHERTYPE_IPV6 = "0x86DD"


def set_acl_table(duthost, name, table_type, stage="ingress", ports=None, description=None):
    """Create/replace CONFIG_DB ACL_TABLE|<name>.

    Equivalent of ``config acl add table <name> <type> -s <stage> -p <ports> -d <desc>``.
    ``ports`` are CONFIG_DB port names (e.g. ``Ethernet1/33``). Lists are stored
    in redis under ``<field>@`` as a comma-separated string.
    """
    fields = {"type": table_type, "stage": stage,
              "policy_desc": description or name}
    if ports:
        if isinstance(ports, str):
            ports = [ports]
        fields["ports@"] = ",".join(ports)
    key = "ACL_TABLE|{}".format(name)
    redis_del(duthost, "CONFIG_DB", key)
    logger.info("Setting %s: %s", key, fields)
    redis_hset(duthost, "CONFIG_DB", key, fields)
    return fields


def set_acl_rules(duthost, table, rules):
    """Write CONFIG_DB ACL_RULE|<table>|<rule> entries. ``rules`` = {rule_name: {field: value}}."""
    for rule, fields in rules.items():
        key = "ACL_RULE|{}|{}".format(table, rule)
        redis_del(duthost, "CONFIG_DB", key)
        logger.info("Setting %s: %s", key, fields)
        redis_hset(duthost, "CONFIG_DB", key, fields)


def del_acl_table(duthost, name):
    """Remove ACL_TABLE|<name> and all of its ACL_RULE entries (rules first, like acl-loader delete)."""
    rules = redis_keys(duthost, "CONFIG_DB", "ACL_RULE|{}|*".format(name))
    redis_del(duthost, "CONFIG_DB", *rules)
    redis_del(duthost, "CONFIG_DB", "ACL_TABLE|{}".format(name))


def get_acl_table_state(duthost, name):
    """STATE_DB ACL_TABLE_TABLE|<name> fields (e.g. {'status': 'Active'})."""
    return redis_hgetall(duthost, "STATE_DB", "ACL_TABLE_TABLE|{}".format(name))


def get_acl_rule_state(duthost, table, rule):
    """STATE_DB ACL_RULE_TABLE|<table>|<rule> fields."""
    return redis_hgetall(duthost, "STATE_DB", "ACL_RULE_TABLE|{}|{}".format(table, rule))


def acl_table_active(duthost, name, rules=()):
    """True if the table (and every listed rule) is reported Active in STATE_DB."""
    if get_acl_table_state(duthost, name).get("status") != ACL_STATUS_ACTIVE:
        return False
    return all(get_acl_rule_state(duthost, name, r).get("status") == ACL_STATUS_ACTIVE for r in rules)


def wait_acl_active(duthost, name, rules=(), timeout=60, interval=3):
    """Wait for orchagent to report ACL_TABLE|<name> (and rules) as Active."""
    return wait_until(timeout, interval, 0, acl_table_active, duthost, name, rules)


def acl_rules_from_openconfig(oc_json, table_name, table_type="L3V6"):
    """Translate an acl-loader style OpenConfig JSON document into CONFIG_DB ACL_RULE fields.

    Supports the subset used by the dhcp_relay tests (forwarding-action,
    sequence-id, destination/source-ip-address, protocol, l4 ports) and adds
    the implicit lowest-priority DEFAULT_RULE (DROP, ETHER_TYPE for L3V6/L3)
    that ``acl-loader update full`` appends.
    """
    entries = (oc_json.get("acl", {}).get("acl-sets", {}).get("acl-set", {})
               .get(table_name, {}).get("acl-entries", {}).get("acl-entry", {}))
    rules = {}
    for name, entry in entries.items():
        fields = {}
        seq = entry.get("config", {}).get("sequence-id")
        if seq is not None:
            fields["PRIORITY"] = str(seq)
        action = entry.get("actions", {}).get("config", {}).get("forwarding-action", "")
        if action:
            fields["PACKET_ACTION"] = "FORWARD" if action.upper() == "ACCEPT" else "DROP"
        ip = entry.get("ip", {}).get("config", {})
        if "destination-ip-address" in ip:
            fam = "IPV6" if ":" in ip["destination-ip-address"] else "IP"
            fields["DST_{}".format(fam)] = ip["destination-ip-address"]
        if "source-ip-address" in ip:
            fam = "IPV6" if ":" in ip["source-ip-address"] else "IP"
            fields["SRC_{}".format(fam)] = ip["source-ip-address"]
        if "protocol" in ip:
            fields["IP_PROTOCOL"] = str(ip["protocol"])
        transport = entry.get("transport", {}).get("config", {})
        if "destination-port" in transport:
            fields["L4_DST_PORT"] = str(transport["destination-port"])
        if "source-port" in transport:
            fields["L4_SRC_PORT"] = str(transport["source-port"])
        rules[name] = fields

    default = {"PRIORITY": "1", "PACKET_ACTION": "DROP"}
    if table_type.upper() == "L3V6":
        default["IP_TYPE"] = "IPV6ANY"
    elif table_type.upper() == "L3":
        default["ETHER_TYPE"] = "0x0800"
    rules["DEFAULT_RULE"] = default
    return rules


# ---------------------------------------------------------------------------
# Container / process management (containerd, not Docker)
# ---------------------------------------------------------------------------

def get_container_tasks(duthost):
    """Return {task_name: status} from ``ctr tasks ls``.

    Example output parsed:
        TASK        PID     STATUS
        db-con      5867    RUNNING
        lldp-con    12206   RUNNING
    """
    res = duthost.shell("{} tasks ls".format(CTR), module_ignore_errors=True, verbose=False)
    tasks = {}
    for line in res["stdout"].splitlines()[1:]:
        cols = line.split()
        if len(cols) >= 3:
            tasks[cols[0]] = cols[2]
    return tasks


def is_container_running(duthost, container=DHCP_RELAY_CONTAINER):
    """Return True if the containerd task is RUNNING."""
    return get_container_tasks(duthost).get(container) == "RUNNING"


def ctr_exec(duthost, command, container=DHCP_RELAY_CONTAINER, module_ignore_errors=True):
    """Run a command inside a GPINS container (the ``docker exec`` equivalent).

    Mirrors the redis-cli wrapper: ``ctr tasks exec --exec-id <unique> <task> <cmd>``.
    """
    exec_id = "sonicmgmt{}".format(time.time_ns())
    cmd = "{} tasks exec --exec-id {} {} {}".format(CTR, exec_id, container, command)
    return duthost.shell(cmd, module_ignore_errors=module_ignore_errors, verbose=False)


def get_process_pid(duthost, process=DHCP6RELAY_PROCESS):
    """Return the PID of a process, or None if it is not running.

    Container processes are visible from the host PID namespace, so this
    avoids depending on the container name or on pgrep inside the container.
    """
    res = duthost.shell("pgrep -x {}".format(shlex.quote(process)),
                        module_ignore_errors=True, verbose=False)
    pids = res["stdout"].split()
    return int(pids[0]) if res["rc"] == 0 and pids else None


def is_dhcp6relay_running(duthost):
    """Return True if the dhcp6relay process is running."""
    return get_process_pid(duthost) is not None


def get_dhcp6relay_sockets(duthost):
    """Return the `ss -nlpu` lines for UDP sockets owned by dhcp6relay.

    Returns [] if ``ss`` is not installed (minimal GPINS host) or the relay has
    no UDP sockets; callers should use ``is_dhcp6relay_bound`` which reads /proc.
    Assumes the DHCP-relay container uses the host network namespace, like
    other SONiC containers.
    """
    res = duthost.shell("ss -nlpu", module_ignore_errors=True, verbose=False)
    if res["rc"] != 0:
        return []
    return [line for line in res["stdout"].splitlines() if DHCP6RELAY_PROCESS in line]


def _proc_hex_to_ipv6(hex_addr):
    """Decode a /proc/net/udp6 local address (4 little-endian 32-bit words in hex)."""
    raw = bytes.fromhex(hex_addr)
    words = [raw[i:i + 4][::-1] for i in range(0, 16, 4)]
    return ipaddress.IPv6Address(b"".join(words)).compressed


# Listener kinds reported by ``dhcp6relay_listeners``.
LISTENER_UDP = "udp547"      # upstream style: UDP socket(s) on port 547
LISTENER_PACKET = "packet"   # GPINS style: AF_PACKET socket (seen 2026-10-08)


def dhcp6relay_listeners(duthost, port=DHCPV6_SERVER_PORT):
    """Describe the listening sockets owned by the dhcp6relay process.

    Returns ``{"pid": int|None, LISTENER_UDP: set(addresses), LISTENER_PACKET: int}``
    where ``addresses`` are the local IPv6 addresses of UDP ``port`` sockets
    ('::' = wildcard) and the packet count is the number of AF_PACKET sockets.

    Works without ``ss``: joins the process fd table with /proc/net/udp6 and
    /proc/net/packet by socket inode. The GPINS relay on zhouboyang_dhcp_test
    opens no UDP 547 socket at all and receives via one AF_PACKET socket, so
    callers must not assume the UDP set is non-empty when the relay is healthy.
    """
    result = {"pid": None, LISTENER_UDP: set(), LISTENER_PACKET: 0}
    pid = get_process_pid(duthost)
    if pid is None:
        return result
    result["pid"] = pid
    cmd = ("awk 'NR>1 {{split($2, a, \":\"); if (a[2] == \"{port:04X}\") print \"U\", a[1], $10}}' /proc/net/udp6; "
           "awk 'NR>1 {{print \"P\", $NF}}' /proc/net/packet 2>/dev/null; "
           "echo ---; ls -l /proc/{pid}/fd 2>/dev/null | grep -o 'socket:\\[[0-9]*\\]' | tr -dc '0-9\\n'"
           ).format(port=port, pid=pid)
    res = duthost.shell(cmd, module_ignore_errors=True, verbose=False)
    if res["rc"] != 0 or "---" not in res["stdout"]:
        return result
    listening, owned = res["stdout"].split("---", 1)
    owned_inodes = set(owned.split())
    for line in listening.splitlines():
        parts = line.split()
        if not parts or parts[-1] not in owned_inodes:
            continue
        if parts[0] == "U" and len(parts) == 3:
            try:
                result[LISTENER_UDP].add(_proc_hex_to_ipv6(parts[1]))
            except ValueError:
                logger.debug("Unparsable /proc/net/udp6 address %r", parts[1])
        elif parts[0] == "P":
            result[LISTENER_PACKET] += 1
    return result


def dhcp6relay_bound_addresses(duthost, port=DHCPV6_SERVER_PORT):
    """Return the set of local IPv6 addresses on which dhcp6relay has a UDP ``port`` socket.

    Empty when the process is not running *or* when the relay uses an AF_PACKET
    listener instead of per-address UDP sockets (the GPINS build). Use
    ``dhcp6relay_uses_address_sockets`` before asserting on specific addresses.
    """
    return dhcp6relay_listeners(duthost, port)[LISTENER_UDP]


def dhcp6relay_uses_address_sockets(duthost):
    """True if the running relay exposes per-address UDP 547 sockets (upstream style)."""
    return bool(dhcp6relay_listeners(duthost)[LISTENER_UDP])


def is_dhcp6relay_bound(duthost, interface=None):
    """Return True if dhcp6relay is running and listening (the "relay is ready" gate).

    "Listening" means it owns a UDP 547 socket (upstream) or an AF_PACKET
    socket (GPINS). ``interface`` is honoured only when per-interface UDP
    sockets exist and ``ss`` is available (it prints ``[fe80::1]%Ethernet0:547``);
    with an AF_PACKET listener the relay is interface-agnostic, so the filter
    is ignored (logged at debug level).
    """
    info = dhcp6relay_listeners(duthost)
    if info["pid"] is None:
        return False
    if info[LISTENER_UDP] and interface is not None:
        scoped = "%{}:{}".format(interface, DHCPV6_SERVER_PORT)
        lines = get_dhcp6relay_sockets(duthost)
        if lines:
            return any(scoped in line for line in lines)
        logger.debug("ss unavailable; cannot filter dhcp6relay sockets by interface %s", interface)
        return True
    if interface is not None and info[LISTENER_PACKET]:
        logger.debug("dhcp6relay uses an AF_PACKET listener; interface filter %s ignored", interface)
    return bool(info[LISTENER_UDP]) or info[LISTENER_PACKET] > 0


def wait_dhcp6relay_bound(duthost, interface=None, timeout=120, interval=5):
    """Wait until dhcp6relay is running and listening. Returns True on success."""
    return wait_until(timeout, interval, 0, is_dhcp6relay_bound, duthost, interface)


def _run_dhcp_relay_init(duthost, cmd_template, wait, timeout, expect_running):
    cmd = cmd_template.format(script=DHCP_RELAY_INIT_SCRIPT)
    logger.info("%s: %s", DHCP_RELAY_CONTAINER, cmd)
    duthost.shell(cmd)
    if not wait:
        return True
    if expect_running:
        return wait_until(timeout, 3, 0, is_container_running, duthost)
    return wait_until(timeout, 3, 0, lambda d: not is_container_running(d), duthost)


def start_dhcp_relay(duthost, wait=True, timeout=60):
    """Start the DHCP-relay container via its init script (no-op if already running).

    GPINS has no daemon that starts the container from FEATURE|dhcp_relay, so
    this is the only enable path. Returns True if the container is running
    afterwards (always True if wait=False).
    """
    if is_container_running(duthost):
        return True
    return _run_dhcp_relay_init(duthost, DHCP_RELAY_START_CMD, wait, timeout, expect_running=True)


def stop_dhcp_relay(duthost, wait=True, timeout=60):
    """Stop and remove the DHCP-relay container via its init script.

    Returns True once the containerd task is gone (always True if wait=False).
    """
    if not is_container_running(duthost):
        return True
    return _run_dhcp_relay_init(duthost, DHCP_RELAY_STOP_CMD, wait, timeout, expect_running=False)


def restart_dhcp_relay(duthost, wait=True, timeout=120):
    """Restart the DHCP-relay container via its GPINS init script.

    Replacement for ``systemctl reset-failed/restart dhcp_relay``.

    Args:
        wait: wait for the container task to be RUNNING again.
        timeout: seconds to wait.

    Returns:
        bool: True if the container is running afterwards (always True if wait=False).
    """
    return _run_dhcp_relay_init(duthost, DHCP_RELAY_RESTART_CMD, wait, timeout, expect_running=True)


# ---------------------------------------------------------------------------
# Topology discovery (replacement for minigraph facts)
# ---------------------------------------------------------------------------
#
# GPINS switches have no /etc/sonic/minigraph.xml, so everything the upstream
# fixtures read from minigraph (VLANs, neighbors, port channels, loopbacks,
# PTF indices) has to come from CONFIG_DB + the kernel + tbinfo instead.

# PORT|<name> field that distinguishes fabric-facing ports. On ju5u1m1.sqs04
# 64 ports carry "fabric_uplink" and 2 carry an empty value (2026-10-01); the
# empty ones are treated as host-facing / downlink ports.
PORT_DIRECTION_FIELD = "port_direction"
UPLINK_PORT_DIRECTION = "fabric_uplink"

DEFAULT_LOOPBACK = "Loopback0"


_LAB_LINKS_CACHE = {}


def _lab_links_for_dut(duthost):
    """Return ``{port_name: link_index}`` from ``ansible/files/sonic_lab_links.csv`` for ``duthost``.

    On a freshly imaged GPINS switch (before ``config_dut.py``), ``PORT|<name>.index``
    holds the physical cage number (e.g. ``Ethernet1/2/1`` and ``Ethernet1/2/5`` both
    have ``index=2``) rather than the 0-based ``sonic_lab_links.csv`` row index used by
    ``tbinfo['topo']`` and the PTF container (``eth0..eth31``). Reading the CSV directly
    maps each cabled port to its true 0-based link index in both states.
    """
    hostname = getattr(duthost, "hostname", None) if duthost is not None else None
    if not isinstance(hostname, str) or not hostname:
        return {}
    if hostname in _LAB_LINKS_CACHE:
        return _LAB_LINKS_CACHE[hostname]
    csv_path = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ansible", "files", "sonic_lab_links.csv"))
    mapping = {}
    if os.path.isfile(csv_path):
        try:
            with open(csv_path, "r") as fh:
                idx = 0
                for row in csv.reader(fh):
                    if len(row) >= 2 and row[0].strip() == hostname:
                        mapping[row[1].strip()] = idx
                        idx += 1
        except OSError:
            mapping = {}
    _LAB_LINKS_CACHE[hostname] = mapping
    return mapping


def _natural_port_key(name):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(name))]


def _port_sort_key(item):
    name, entry = item
    try:
        return (0, int(entry.get("index", 0)), _natural_port_key(name))
    except ValueError:
        return (1, 0, _natural_port_key(name))


def get_ports(duthost):
    """Return CONFIG_DB PORT table as {name: fields}."""
    ports = get_table(duthost, "CONFIG_DB", "PORT")
    for name, entry in ports.items():
        if isinstance(entry, dict):
            entry.setdefault("_port_name", name)
    return ports


def get_port_oper_status(duthost):
    """Return {port: oper_status} from APPL_DB PORT_TABLE (``up`` / ``down``).

    Only 12 of the 62 host-facing ports on ju3u1m1.sqs901 are cabled to the PTF
    (oper up); the rest must never be chosen as relay/client ports.
    """
    table = get_table(duthost, "APPL_DB", "PORT_TABLE")
    return {name: entry.get("oper_status", "") for name, entry in table.items()}


def usable_downlinks(duthost, downlinks, tbinfo=None):
    """Filter ``downlinks`` to ports that can carry test traffic.

    A port is usable when APPL_DB reports it oper ``up`` and, if the testbed
    defines a ``ptf_map`` (or ``sonic_lab_links.csv`` entries) for this DUT,
    the port has a PTF index in that map.
    Order is preserved. Logs what was dropped so a wrong selection is visible.
    """
    oper = get_port_oper_status(duthost)
    ports = get_ports(duthost) if tbinfo is not None else {}
    lab_links = _lab_links_for_dut(duthost) if tbinfo is not None else {}
    ptf_map = None
    if tbinfo is not None:
        try:
            ptf_map = tbinfo["topo"]["ptf_map"][str(tbinfo["duts"].index(duthost.hostname))] or None
        except (KeyError, ValueError, TypeError):
            ptf_map = None
    usable, dropped = [], {}
    for port in downlinks:
        if oper.get(port) != "up":
            dropped[port] = "oper {}".format(oper.get(port) or "unknown")
            continue
        if lab_links and port not in lab_links:
            dropped[port] = "not in sonic_lab_links.csv"
            continue
        port_idx = str(lab_links.get(port, ports.get(port, {}).get("index", "")))
        if ptf_map is not None and port_idx not in ptf_map:
            dropped[port] = "not in ptf_map"
            continue
        usable.append(port)
    if dropped:
        logger.info("Ignoring %d downlink(s) without a usable link: %s", len(dropped),
                    ", ".join("{} ({})".format(p, r) for p, r in sorted(dropped.items())))
    return usable


def classify_ports(ports, duthost=None, tbinfo=None):
    """Split PORT entries into (downlinks, uplinks), each sorted by PORT.index.

    Uplinks are ports whose ``port_direction`` is ``fabric_uplink``; every
    other port (including ones with no ``port_direction``) is a downlink.

    If *no* port carries ``port_direction`` at all (the upstream sonic-mgmt
    t0 testbeds, where ``config_dut.py`` writes PORT without it), fall back to:
      1. BGP (needs ``duthost``): a port is an uplink when one of its
         ``INTERFACE|<port>|<prefix>`` IPv6 prefixes contains a configured
         ``BGP_NEIGHBOR`` address (verified on ju3u1m1.sqs901, 2026-10-08);
      2. the topology file (needs ``tbinfo``): ports whose PTF index is listed
         under a VM in ``topo.properties.topology.VMs`` - the BGP-free path
         Boyang asked for, since bgp-con may be dropped from the DHCP image.
    """
    ordered = [name for name, _ in sorted(ports.items(), key=_port_sort_key)]
    if any(entry.get(PORT_DIRECTION_FIELD) for entry in ports.values()):
        uplink_set = {name for name, entry in ports.items()
                      if entry.get(PORT_DIRECTION_FIELD) == UPLINK_PORT_DIRECTION}
    else:
        uplink_set = _bgp_uplinks(duthost, ordered) if duthost is not None else set()
        if not uplink_set and tbinfo is not None:
            uplink_set = _topology_uplinks(tbinfo, duthost, ports)
    downlinks = [p for p in ordered if p not in uplink_set]
    uplinks = [p for p in ordered if p in uplink_set]
    return downlinks, uplinks


def _topology_uplinks(tbinfo, duthost, ports):
    """Ports that face a topology VM according to tbinfo (no BGP needed).

    Upstream topology files list, per VM, the DUT port indices (``vlans``) it is
    wired to; those are the T1-facing uplinks of a t0.
    """
    try:
        vms = tbinfo["topo"]["properties"]["topology"]["VMs"]
    except (KeyError, TypeError):
        return set()
    vm_indices = set()
    for vm in vms.values():
        for idx in vm.get("vlans", []):
            vm_indices.add(int(idx))
    if not vm_indices:
        return set()
    lab_links = _lab_links_for_dut(duthost)
    uplinks = set()
    for name, entry in ports.items():
        if lab_links and name not in lab_links:
            continue
        if duthost is not None and ptf_index_for_port(tbinfo, duthost, entry, name) in vm_indices:
            uplinks.add(name)
    return uplinks


def get_bgp_neighbor_addresses(duthost):
    """Return the set of BGP_NEIGHBOR peer IPs from CONFIG_DB.

    Keys are ``BGP_NEIGHBOR|<ip>`` or ``BGP_NEIGHBOR|<vrf>|<ip>``.
    """
    peers = set()
    for key in get_table(duthost, "CONFIG_DB", "BGP_NEIGHBOR"):
        peers.add(key.rsplit("|", 1)[-1])
    return peers


def _bgp_uplinks(duthost, ports):
    """Ports whose INTERFACE IPv6 prefix contains a BGP_NEIGHBOR address."""
    peers = set()
    for ip in get_bgp_neighbor_addresses(duthost):
        try:
            peers.add(ipaddress.ip_address(ip))
        except ValueError:
            continue
    if not peers:
        return set()
    sep = _db_separator("CONFIG_DB")
    prefix = "INTERFACE" + sep
    port_set = set(ports)
    uplinks = set()
    # One round-trip for the whole table: keys look like INTERFACE|<port>|<ip>/<len>.
    for key in redis_keys(duthost, "CONFIG_DB", prefix + "*"):
        parts = key[len(prefix):].split(sep)
        if len(parts) != 2 or parts[0] not in port_set or "/" not in parts[1]:
            continue
        try:
            net = ipaddress.ip_network(parts[1], strict=False)
        except ValueError:
            continue
        if any(p.version == net.version and p in net for p in peers):
            uplinks.add(parts[0])
    return uplinks


def _ip_entries(duthost, table, interface):
    """Return [(ip, prefixlen)] for ``<table>|<interface>|<ip>/<len>`` keys."""
    sep = _db_separator("CONFIG_DB")
    prefix = "{}{}{}{}".format(table, sep, interface, sep)
    result = []
    for key in redis_keys(duthost, "CONFIG_DB", prefix + "*"):
        cidr = key[len(prefix):]
        if "/" not in cidr:
            continue
        ip, plen = cidr.rsplit("/", 1)
        result.append((ip, int(plen)))
    return result


def get_interface_ipv6(duthost, interface, table="INTERFACE"):
    """Return (ipv6, prefixlen) configured on ``interface`` in CONFIG_DB, or (None, None)."""
    for ip, plen in _ip_entries(duthost, table, interface):
        if ":" in ip:
            return ip, plen
    return None, None


def get_loopback_ipv6(duthost, loopback=DEFAULT_LOOPBACK):
    """Return (ipv6, prefixlen) of ``LOOPBACK_INTERFACE|<loopback>|<v6>/<len>``, or (None, None)."""
    return get_interface_ipv6(duthost, loopback, table="LOOPBACK_INTERFACE")


def get_port_mac(duthost, port):
    """MAC address of a CONFIG_DB port name, read from the kernel netdev."""
    res = duthost.shell("cat /sys/class/net/{}/address".format(to_kernel_ifname(port)))
    return res["stdout"].strip()


def get_port_link_local(duthost, port):
    """Return (lla, lla_with_prefix) of the kernel netdev for ``port``, or ("", "")."""
    cmd = "ip -6 addr show dev {} scope link | awk '/inet6/ {{print $2; exit}}'".format(to_kernel_ifname(port))
    res = duthost.shell(cmd, module_ignore_errors=True)
    with_prefix = res.get("stdout", "").strip()
    if not with_prefix:
        return "", ""
    return with_prefix.split("/")[0], with_prefix


def get_port_kernel_guas(duthost, port):
    """Return list of global-scope IPv6 CIDRs (``['2001:db8:1::1/64', ...]``) on the kernel netdev for ``port``."""
    cmd = "ip -6 addr show dev {} scope global | awk '/inet6/ {{print $2}}'".format(
        shlex.quote(to_kernel_ifname(port)))
    res = duthost.shell(cmd, module_ignore_errors=True, verbose=False)
    if res.get("rc", 1) != 0:
        return []
    return [line.strip() for line in res.get("stdout", "").splitlines() if line.strip()]


def check_interface_mode(duthost, interface, mode, ports=None):
    """Verify ``interface`` matches the expected relay ``mode`` ('vlan'/'svi', 'l3_lla', 'l3_gua').

    Returns ``(ok, reason)``:
      * ``vlan`` (``svi``): ``interface`` is a VLAN interface (``Vlan*``) present in
        CONFIG_DB ``VLAN`` and the kernel, with at least one physical member and
        an IPv6 GUA in ``VLAN_INTERFACE`` and on the kernel netdev.
      * ``l3_gua``: ``interface`` is a physical L3 port in CONFIG_DB ``PORT`` (not a
        VLAN or VLAN member) with both a link-local address and a global IPv6
        address (GUA) in CONFIG_DB ``INTERFACE`` and on the kernel netdev.
      * ``l3_lla``: ``interface`` is a physical L3 port in CONFIG_DB ``PORT`` (not a
        VLAN or VLAN member) with a link-local address and *no* GUA in either
        CONFIG_DB ``INTERFACE`` or on the kernel netdev.
    """
    mode = normalize_relay_mode(mode)
    if mode == RELAY_MODE_VLAN:
        if not str(interface).startswith("Vlan"):
            return False, "{} is not a VLAN interface (mode={})".format(interface, mode)
        vlans = get_vlans(duthost)
        if interface not in vlans:
            return False, "{} is not present in CONFIG_DB VLAN table".format(interface)
        members = [m for m in vlans[interface]["members"] if not m.startswith("PortChannel")]
        if not members:
            return False, "{} has no physical VLAN_MEMBER ports".format(interface)
        cfg_gua, _ = vlans[interface]["ipv6"]
        if not cfg_gua:
            return False, "{} has no IPv6 GUA in CONFIG_DB VLAN_INTERFACE".format(interface)
        return True, ""

    if str(interface).startswith("Vlan"):
        return False, "{} is a VLAN interface, expected L3 physical port for mode={}".format(interface, mode)
    ports = ports if ports is not None else get_ports(duthost)
    if ports and interface not in ports:
        return False, "{} is not in CONFIG_DB PORT table".format(interface)
    sep = _db_separator("CONFIG_DB")
    vlan_memberships = redis_keys(duthost, "CONFIG_DB", "VLAN_MEMBER{}*{}{}".format(sep, sep, interface))
    if vlan_memberships:
        return False, "{} is a L2 VLAN member ({}), expected L3 port for mode={}".format(
            interface, vlan_memberships, mode)

    cfg_gua, _ = get_interface_ipv6(duthost, interface)
    kernel_guas = get_port_kernel_guas(duthost, interface)

    if mode == RELAY_MODE_L3_LLA:
        if cfg_gua or kernel_guas:
            return False, "{} must not have a GUA in l3_lla mode (CONFIG_DB GUA={}, kernel GUAs={})".format(
                interface, cfg_gua, kernel_guas)
        return True, ""

    # mode == RELAY_MODE_L3_GUA
    if not cfg_gua:
        return False, "{} must have a GUA in l3_gua mode (CONFIG_DB GUA={})".format(
            interface, cfg_gua)
    return True, ""


def verify_interface_mode(duthost, interface, mode, ports=None):
    """Assert ``interface`` satisfies ``check_interface_mode(duthost, interface, mode)`` before enabling DHCP relay."""
    ok, reason = check_interface_mode(duthost, interface, mode, ports=ports)
    if not ok:
        raise AssertionError("Interface {} does not match relay mode {}: {}".format(interface, mode, reason))


def ptf_index_for_port(tbinfo, duthost, port_entry, port_name=None):
    """Map a CONFIG_DB PORT entry to its PTF port index.

    Mirrors ``SonicHost.get_extended_minigraph_facts``: the DUT port index
    (from ``sonic_lab_links.csv`` if present for this DUT, otherwise
    ``PORT.index``) is looked up in ``tbinfo['topo']['ptf_map'][<dut idx>]``.
    If the testbed defines no ptf_map, the port index is used as-is.
    """
    port_entry = port_entry or {}
    port_name = port_name or port_entry.get("_port_name")
    lab_links = _lab_links_for_dut(duthost)
    if port_name and port_name in lab_links:
        port_index = int(lab_links[port_name])
    else:
        port_index = int(port_entry.get("index", 0))
    try:
        dut_index = tbinfo["duts"].index(duthost.hostname)
        ptf_map = tbinfo["topo"]["ptf_map"][str(dut_index)]
        if ptf_map and str(port_index) in ptf_map:
            return int(ptf_map[str(port_index)])
    except (ValueError, KeyError, TypeError):
        pass
    return port_index


def parse_dhcpv6_servers(dhcp_relay_entry):
    """Return the server list from a DHCP_RELAY|<intf> entry ({'dhcpv6_servers': 'a,b'} or 'dhcpv6_servers@')."""
    for field in ("dhcpv6_servers", "dhcpv6_servers@"):
        value = dhcp_relay_entry.get(field)
        if value:
            return [s.strip() for s in value.split(",") if s.strip()]
    return []


# ---------------------------------------------------------------------------
# Relay modes
# ---------------------------------------------------------------------------
#
# The test plan requires every test to run in all three configuration
# scenarios without duplicating test bodies:
#   l3_lla  - relay on an L3 port with link-local only (no GUA in INTERFACE);
#             Interface-ID carries "<port>" (e.g. "Ethernet1/4/1")
#   l3_gua  - relay on an L3 port with a global IPv6 address configured in
#             CONFIG_DB ("INTERFACE|<port>|2001:db8:<n>::1/64")
#   vlan    - relay on a VLAN interface (Vlan<id>) with a GUA, hosts are L2
#             VLAN members (supported automatically once GPINS adds VLANs)

RELAY_MODE_L3_LLA = "l3_lla"
RELAY_MODE_L3_GUA = "l3_gua"
RELAY_MODE_VLAN = "vlan"
RELAY_MODE_SVI = RELAY_MODE_VLAN  # backwards-compatible alias
RELAY_MODES = (RELAY_MODE_L3_LLA, RELAY_MODE_L3_GUA, RELAY_MODE_VLAN)
RELAY_MODE_ALIASES = {
    "svi": RELAY_MODE_VLAN,
    "vlan": RELAY_MODE_VLAN,
    "l3_gua": RELAY_MODE_L3_GUA,
    "l3_lla": RELAY_MODE_L3_LLA,
}

DEFAULT_L3_GUA_TEMPLATE = "2001:db8:{idx}::1/64"
DEFAULT_TEST_VLAN_ID = 1000
DEFAULT_TEST_VLAN = "Vlan{}".format(DEFAULT_TEST_VLAN_ID)
DEFAULT_TEST_VLAN_GUA = "2001:db8:1000::1/64"


def normalize_relay_mode(mode):
    """Return the canonical relay mode name ('l3_lla', 'l3_gua', or 'vlan')."""
    if mode in RELAY_MODE_ALIASES:
        return RELAY_MODE_ALIASES[mode]
    raise ValueError("unknown relay mode {!r}; expected one of {}".format(mode, RELAY_MODES))


def set_interface_ipv6(duthost, interface, address_with_plen, table="INTERFACE"):
    """Configure an IPv6 GUA on ``interface`` in CONFIG_DB + kernel (Appendix step 1).

    Equivalent of:
        redis-cli -n 4 HSET "INTERFACE|Ethernet1/4/1|2001:db8:1::1/64" "NULL" "NULL"
        ip -6 addr replace 2001:db8:1::1/64 dev Ethernet1_4_1
    """
    sep = _db_separator("CONFIG_DB")
    base_key = "{}{}{}".format(table, sep, interface)
    addr_key = "{}{}{}{}".format(table, sep, interface, sep) + address_with_plen
    config_db_set(duthost, base_key)
    config_db_set(duthost, addr_key)
    dev = to_kernel_ifname(interface)
    duthost.shell("ip -6 addr replace {} dev {}".format(shlex.quote(address_with_plen), shlex.quote(dev)),
                  module_ignore_errors=True)
    return [base_key, addr_key]


def del_interface_ipv6(duthost, interface, address_with_plen, table="INTERFACE"):
    """Remove an IPv6 GUA from ``interface`` in CONFIG_DB + kernel."""
    sep = _db_separator("CONFIG_DB")
    addr_key = "{}{}{}{}".format(table, sep, interface, sep) + address_with_plen
    config_db_del(duthost, addr_key)
    dev = to_kernel_ifname(interface)
    duthost.shell("ip -6 addr del {} dev {}".format(shlex.quote(address_with_plen), shlex.quote(dev)),
                  module_ignore_errors=True, verbose=False)


def get_vlans(duthost):
    """Return {vlan: {"members": [ports], "ipv6": (addr, plen) or (None, None)}} from CONFIG_DB."""
    sep = _db_separator("CONFIG_DB")
    vlans = {name: {"members": [], "ipv6": (None, None)} for name in get_table(duthost, "CONFIG_DB", "VLAN")}
    if not vlans:
        return vlans
    for key in redis_keys(duthost, "CONFIG_DB", "VLAN_MEMBER" + sep + "*"):
        parts = key.split(sep)
        if len(parts) == 3 and parts[1] in vlans:
            vlans[parts[1]]["members"].append(parts[2])
    for name in vlans:
        vlans[name]["members"].sort()
        vlans[name]["ipv6"] = get_interface_ipv6(duthost, name, table="VLAN_INTERFACE")
    return vlans


def relay_targets(duthost, mode, ports=None, tbinfo=None, require_link=True):
    """Discover relay interfaces for ``mode``.

    Returns a list of dicts ``{"relay_interface", "client_port", "gua", "gua_plen"}``:
      * vlan:   one per VLAN that has a physical member and an IPv6 VLAN_INTERFACE
      * l3_gua: one per downlink port with an IPv6 INTERFACE entry
      * l3_lla: one per downlink port that has *no* GUA in CONFIG_DB or the kernel
    With ``require_link`` (default) only downlinks that are oper up - and in the
    testbed's ``ptf_map`` when one is defined - are considered, so the first
    target is always a port the PTF can actually reach.
    An empty list means the DUT is not configured for that mode.
    """
    mode = normalize_relay_mode(mode)
    ports = ports if ports is not None else get_ports(duthost)
    downlinks, _ = classify_ports(ports, duthost, tbinfo)
    if require_link:
        downlinks = usable_downlinks(duthost, downlinks, tbinfo)
    targets = []
    if mode == RELAY_MODE_VLAN:
        for vlan, info in sorted(get_vlans(duthost).items()):
            members = [m for m in info["members"] if not m.startswith("PortChannel")]
            if require_link:
                members = [m for m in members if m in downlinks]
                res = duthost.shell("test -d /sys/class/net/{}".format(shlex.quote(vlan)),
                                    module_ignore_errors=True, verbose=False)
                if res.get("rc", 1) != 0:
                    continue
            addr, plen = info["ipv6"]
            if members and addr:
                targets.append({"relay_interface": vlan, "client_port": members[0], "gua": addr, "gua_plen": plen})
    elif mode == RELAY_MODE_L3_GUA:
        for port in downlinks:
            addr, plen = get_interface_ipv6(duthost, port)
            if addr:
                targets.append({"relay_interface": port, "client_port": port, "gua": addr, "gua_plen": plen})
    else:  # RELAY_MODE_L3_LLA
        for port in downlinks:
            addr, _ = get_interface_ipv6(duthost, port)
            if addr is None and not get_port_kernel_guas(duthost, port):
                targets.append({"relay_interface": port, "client_port": port, "gua": None, "gua_plen": None})
    logger.info("relay targets for mode %s: %s", mode, [t["relay_interface"] for t in targets])
    return targets


def configure_relay_mode(duthost, mode, ports=None, tbinfo=None):
    """Provision the DUT's downlink interfaces for ``mode`` so the same tests run in every mode.

    * ``l3_lla``: ensure usable downlink ports have *no* IPv6 GUA in CONFIG_DB / kernel
                  (snapshots and removes any existing GUA, restored on teardown).
    * ``l3_gua``: if no usable downlink port has an IPv6 GUA in CONFIG_DB, configure
                  ``INTERFACE|<port>|2001:db8:<idx>::1/64`` + kernel address on usable
                  downlinks (Appendix step 1) and remove on teardown.
    * ``vlan``:   if no VLAN with an oper-up member and IPv6 VLAN_INTERFACE exists, try
                  provisioning ``Vlan1000`` on the first usable downlink. If the GPINS
                  image does not create the ``Vlan1000`` netdev yet, revert CONFIG_DB
                  and report ``supported=False`` so the test skips cleanly until VLAN
                  support lands on GPINS.

    Returns ``(supported, reason, state)`` where ``state`` is passed to ``teardown_relay_mode``.
    """
    try:
        mode = normalize_relay_mode(mode)
    except ValueError as exc:
        return False, str(exc), None

    ports = ports if ports is not None else get_ports(duthost)
    downlinks, _ = classify_ports(ports, duthost, tbinfo)
    downlinks = usable_downlinks(duthost, downlinks, tbinfo)
    if not downlinks:
        return False, "{} mode: no oper-up downlink port available".format(mode), None

    state = {
        "mode": mode,
        "snapshot": None,
        "added_kernel_addrs": [],    # [(port, cidr)]
        "removed_kernel_addrs": [],  # [(port, cidr)]
    }

    if mode == RELAY_MODE_L3_LLA:
        patterns = ["INTERFACE|{}|*".format(p) for p in downlinks]
        snap = snapshot_config_db(duthost, patterns)
        ipv6_keys = [k for k in snap["entries"] if ":" in k.rsplit("|", 1)[-1]]
        removed_set = set()
        if ipv6_keys:
            state["snapshot"] = snap
            for key in ipv6_keys:
                _, port, cidr = key.split("|", 2)
                del_interface_ipv6(duthost, port, cidr)
                state["removed_kernel_addrs"].append((port, cidr))
                removed_set.add((port, cidr))
        for port in downlinks[:2]:
            for cidr in get_port_kernel_guas(duthost, port):
                dev = to_kernel_ifname(port)
                duthost.shell("ip -6 addr del {} dev {}".format(shlex.quote(cidr), shlex.quote(dev)),
                              module_ignore_errors=True, verbose=False)
                if (port, cidr) not in removed_set:
                    state["removed_kernel_addrs"].append((port, cidr))
                    removed_set.add((port, cidr))
        return True, "", state

    if mode == RELAY_MODE_L3_GUA:
        existing = relay_targets(duthost, RELAY_MODE_L3_GUA, ports, tbinfo)
        if len(existing) >= min(2, len(downlinks)):
            return True, "", state
        target_ports = downlinks[:2]
        patterns = (["INTERFACE|{}".format(p) for p in target_ports] +
                    ["INTERFACE|{}|*".format(p) for p in target_ports])
        state["snapshot"] = snapshot_config_db(duthost, patterns)
        for idx, port in enumerate(target_ports, start=1):
            addr, _ = get_interface_ipv6(duthost, port)
            if addr and get_port_kernel_guas(duthost, port):
                continue
            cidr = "{}/{}".format(addr, _) if addr else DEFAULT_L3_GUA_TEMPLATE.format(idx=idx)
            set_interface_ipv6(duthost, port, cidr)
            state["added_kernel_addrs"].append((port, cidr))
        return True, "", state

    # mode == RELAY_MODE_VLAN
    existing = relay_targets(duthost, RELAY_MODE_VLAN, ports, tbinfo)
    if existing:
        return True, "", state

    vlan = DEFAULT_TEST_VLAN
    member = downlinks[0]
    patterns = [
        "VLAN|{}".format(vlan),
        "VLAN_MEMBER|{}|*".format(vlan),
        "VLAN_INTERFACE|{}".format(vlan),
        "VLAN_INTERFACE|{}|*".format(vlan),
    ]
    state["snapshot"] = snapshot_config_db(duthost, patterns)
    config_db_set(duthost, "VLAN|{}".format(vlan), {"vlanid": str(DEFAULT_TEST_VLAN_ID)})
    config_db_set(duthost, "VLAN_MEMBER|{}|{}".format(vlan, member), {"tagging_mode": "untagged"})
    config_db_set(duthost, "VLAN_INTERFACE|{}".format(vlan))
    config_db_set(duthost, "VLAN_INTERFACE|{}|{}".format(vlan, DEFAULT_TEST_VLAN_GUA))

    def _vlan_netdev_ready():
        res = duthost.shell("test -d /sys/class/net/{}".format(shlex.quote(vlan)),
                            module_ignore_errors=True, verbose=False)
        return res.get("rc", 1) == 0

    if not wait_until(10, 2, 0, _vlan_netdev_ready):
        restore_config_db(duthost, state["snapshot"])
        return (False,
                "vlan mode: GPINS image did not create {} netdev from CONFIG_DB VLAN tables "
                "(VLAN support not yet enabled on this build)".format(vlan),
                None)

    duthost.shell("ip -6 addr replace {} dev {}".format(
        shlex.quote(DEFAULT_TEST_VLAN_GUA), shlex.quote(vlan)), module_ignore_errors=True)
    state["added_kernel_addrs"].append((vlan, DEFAULT_TEST_VLAN_GUA))
    return True, "", state


def teardown_relay_mode(duthost, state):
    """Revert any CONFIG_DB / kernel interface changes made by ``configure_relay_mode``."""
    if not state:
        return
    for port, cidr in state.get("added_kernel_addrs", []):
        dev = to_kernel_ifname(port)
        duthost.shell("ip -6 addr del {} dev {}".format(shlex.quote(cidr), shlex.quote(dev)),
                      module_ignore_errors=True, verbose=False)
    if state.get("snapshot") is not None:
        restore_config_db(duthost, state["snapshot"])
    for port, cidr in state.get("removed_kernel_addrs", []):
        dev = to_kernel_ifname(port)
        duthost.shell("ip -6 addr replace {} dev {}".format(shlex.quote(cidr), shlex.quote(dev)),
                      module_ignore_errors=True, verbose=False)


def relay_mode_support(duthost, mode, ports=None, tbinfo=None):
    """Return (supported, reason). ``reason`` explains an unsupported mode for pytest.skip."""
    try:
        targets = relay_targets(duthost, mode, ports, tbinfo)
    except ValueError as exc:
        return False, str(exc)
    if targets:
        return True, ""
    mode = normalize_relay_mode(mode)
    reasons = {
        RELAY_MODE_VLAN: "no VLAN with an oper-up physical member and an IPv6 VLAN_INTERFACE in CONFIG_DB",
        RELAY_MODE_L3_GUA: "no oper-up downlink port with an IPv6 INTERFACE entry in CONFIG_DB",
        RELAY_MODE_L3_LLA: "no oper-up downlink port (every PORT is an uplink or has no link)",
    }
    return False, "{} mode: {}".format(mode, reasons[mode])


# ---------------------------------------------------------------------------
# Static path to the DHCPv6 server and client NDv6 resolution (BGP/ND bypass)
# ---------------------------------------------------------------------------
#
# Per the Sample DHCP Relay Port Configuration Appendix:
#   1. Resolve client MAC on the relay port if NDv6 does not work:
#        ip -6 neigh replace <client_lla> lladdr <client_mac> dev <relay_kernel_iface>
#   2. Steer server packets out of the egress port with static neighbour + route:
#        ip -6 neigh replace <nexthop_or_server> lladdr <mac> dev <egress_iface>
#        ip -6 route replace <server>/128 [via <nexthop>] dev <egress_iface>

def add_static_client_neighbor(duthost, relay_interface, client_lla, client_mac):
    """Install a static IPv6 neighbor entry for the DHCPv6 client on ``relay_interface`` (Appendix step 1)."""
    if not client_lla or not client_mac:
        return None
    dev = to_kernel_ifname(relay_interface)
    duthost.shell("ip -6 neigh replace {} lladdr {} dev {} nud permanent".format(
        shlex.quote(str(client_lla)), shlex.quote(str(client_mac)), shlex.quote(dev)),
        module_ignore_errors=True)
    return (str(client_lla), dev)


def del_static_client_neighbor(duthost, entry):
    """Remove the static client neighbor entry added by ``add_static_client_neighbor``."""
    if not entry:
        return
    client_lla, dev = entry
    duthost.shell("ip -6 neigh del {} dev {}".format(shlex.quote(client_lla), shlex.quote(dev)),
                  module_ignore_errors=True, verbose=False)


def ipv6_route_exists(duthost, address):
    """True if the kernel has any route for ``address`` (``ip -6 route get`` succeeds)."""
    res = duthost.shell("ip -6 route get {}".format(shlex.quote(str(address))),
                        module_ignore_errors=True, verbose=False)
    return res["rc"] == 0 and "unreachable" not in res["stdout"]


def ipv6_explicit_route_exists(duthost, address):
    """True if the kernel has a non-default route specifically covering ``address``.

    On t0 testbeds with bgp-con running, T1 peers advertise ``default`` (``::/0``)
    whose nexthop is the Arista cEOS container rather than the PTF interface.
    Checking for a non-default route ensures ``--dhcp_relay_static_route=auto``
    still installs the PTF-facing ``/128`` static route + neighbour unless an
    explicit server route was already configured.
    """
    cidr = "{}/128".format(address)
    res = duthost.shell("ip -6 route show to match {}".format(shlex.quote(cidr)),
                        module_ignore_errors=True, verbose=False)
    if res["rc"] != 0:
        return False
    lines = [line.strip() for line in res["stdout"].splitlines()
             if line.strip() and not line.strip().startswith("default")]
    return bool(lines)


def add_static_server_path(duthost, server, uplink_port, nexthop=None, nexthop_mac=None):
    """Steer traffic for ``server`` out of ``uplink_port`` without BGP.

    * ``nexthop``/``nexthop_mac`` given: ``ip -6 neigh replace <nexthop> lladdr
      <mac> dev <uplink> nud permanent`` + ``ip -6 route replace <server>/128
      via <nexthop> dev <uplink>`` (Appendix step 2).
    * only ``nexthop_mac`` given: treat the server as on-link - static neighbour
      for the server address itself, on-link /128 route. This is what the PTF
      based tests need: the "server" is the PTF port behind the uplink.
    * neither: on-link route only (kernel will ND for the server).
    Returns the list of (kind, spec) entries created so they can be removed.
    """
    dev = to_kernel_ifname(uplink_port)
    created = []
    neigh = nexthop or (str(server) if nexthop_mac else None)
    if neigh and nexthop_mac:
        duthost.shell("ip -6 neigh replace {} lladdr {} dev {} nud permanent".format(
            shlex.quote(neigh), shlex.quote(nexthop_mac), shlex.quote(dev)))
        created.append(("neigh", "{} dev {}".format(neigh, dev)))
    if nexthop:
        duthost.shell("ip -6 route replace {}/128 via {} dev {}".format(
            shlex.quote(str(server)), shlex.quote(nexthop), shlex.quote(dev)))
    else:
        duthost.shell("ip -6 route replace {}/128 dev {}".format(shlex.quote(str(server)), shlex.quote(dev)))
    created.append(("route", "{}/128 dev {}".format(server, dev)))
    logger.info("Static server path for %s via %s: %s", server, dev, created)
    return created


def del_static_server_path(duthost, created):
    """Remove the entries returned by ``add_static_server_path`` (errors ignored)."""
    for kind, spec in reversed(created):
        duthost.shell("ip -6 {} del {}".format(kind, spec), module_ignore_errors=True, verbose=False)


# ---------------------------------------------------------------------------
# Loopback IPv6 (needed by the relay once Interface-ID is enabled)
# ---------------------------------------------------------------------------
#
# Verified on ju3u1m1 (2026-10-09): with enable_interface_id=true dhcp6relay
# opens a UDP 547 listen socket bound to Loopback0's IPv6 address (where the
# server's Relay-Reply arrives). If Loopback0 has no IPv6 the bind fails
# ("Failed to create loopback listen socket") and the daemon exits 1, so no
# l3_lla relay can come up. Lab t0 configs written by config_dut.py leave
# Loopback0 without an IPv6, hence this helper. Upstream t0 uses fc00:1::32/128.

DEFAULT_LOOPBACK_IPV6 = "fc00:1::32/128"


def ensure_loopback_ipv6(duthost, address_with_plen=DEFAULT_LOOPBACK_IPV6, loopback=DEFAULT_LOOPBACK):
    """Make sure ``loopback`` has an IPv6 address; return the CONFIG_DB key created (or None).

    Writes ``LOOPBACK_INTERFACE|<lb>|<addr>/<plen>`` and the kernel address.
    No-op (returns None) when the loopback already has any IPv6 address, so a
    properly configured DUT is never modified.
    """
    addr, _ = get_loopback_ipv6(duthost, loopback)
    if addr:
        return None
    key = "LOOPBACK_INTERFACE|{}|{}".format(loopback, address_with_plen)
    logger.warning("%s has no IPv6 address; adding %s for this session (dhcp6relay needs it for its loopback "
                   "listen socket when Interface-ID is enabled)", loopback, address_with_plen)
    config_db_set(duthost, "LOOPBACK_INTERFACE|{}".format(loopback))
    config_db_set(duthost, key)
    duthost.shell("ip -6 addr add {} dev {}".format(shlex.quote(address_with_plen), shlex.quote(loopback)),
                  module_ignore_errors=True)
    return key


def remove_loopback_ipv6(duthost, key, loopback=DEFAULT_LOOPBACK):
    """Undo ``ensure_loopback_ipv6`` (``key`` is its return value; None = nothing to do)."""
    if not key:
        return
    address_with_plen = key.rsplit("|", 1)[-1]
    config_db_del(duthost, key)
    # The GPINS stack removes the kernel address when the row disappears; be explicit anyway.
    duthost.shell("ip -6 addr del {} dev {}".format(shlex.quote(address_with_plen), shlex.quote(loopback)),
                  module_ignore_errors=True, verbose=False)


# ---------------------------------------------------------------------------
# DHCPv6 CoPP trap + policer (needed when copp_cfg.json omits dhcpv6)
# ---------------------------------------------------------------------------
#
# Verified on ju3u1m1.sqs901 running zhouboyang_dhcp_test (2026-10-12):
# /etc/sonic/copp_cfg.json inside or-con defines no dhcpv6 / dhcpv6_l2 trap,
# so ASIC_DB has no SAI_HOSTIF_TRAP_TYPE_DHCPV6 / SAI_HOSTIF_TRAP_TYPE_DHCPV6_L2
# and the ASIC drops all incoming DHCPv6 packets (SAI_PORT_STAT_IF_IN_DISCARDS)
# before they reach the Linux netdevs. Writing COPP_GROUP|dhcpv6_group (queue 20,
# 1600 pps / burst 512 per the design doc) and COPP_TRAP|dhcpv6 in CONFIG_DB
# causes coppmgrd + orchagent to install both hostif traps and the policer.

DHCPV6_COPP_GROUP_KEY = "COPP_GROUP|dhcpv6_group"
DHCPV6_COPP_TRAP_KEY = "COPP_TRAP|dhcpv6"
DEFAULT_DHCPV6_COPP_GROUP_FIELDS = {
    "queue": "20",
    "trap_action": "trap",
    "trap_priority": "20",
    "meter_type": "packets",
    "mode": "sr_tcm",
    "cir": "1600",
    "cbs": "512",
    "red_action": "drop",
}
DEFAULT_DHCPV6_COPP_TRAP_FIELDS = {
    "trap_ids": "dhcpv6,dhcpv6_l2",
    "trap_group": "dhcpv6_group",
    "always_enabled": "true",
}


def ensure_dhcpv6_copp_trap(duthost):
    """Ensure ``dhcpv6`` CoPP trap + 1600 pps policer exist; return CONFIG_DB keys created."""
    for group_name, entry in get_table(duthost, "APPL_DB", "COPP_TABLE").items():
        trap_ids = [t.strip() for t in entry.get("trap_ids", "").split(",") if t.strip()]
        if "dhcpv6" in trap_ids or "dhcpv6_l2" in trap_ids:
            if entry.get("cir"):
                return []
            group_key = "COPP_GROUP|{}".format(group_name)
            fields = dict(DEFAULT_DHCPV6_COPP_GROUP_FIELDS)
            if entry.get("queue"):
                fields["queue"] = entry["queue"]
            if entry.get("trap_priority"):
                fields["trap_priority"] = entry["trap_priority"]
            config_db_set(duthost, group_key, fields)
            time.sleep(1)
            updated = redis_hgetall(duthost, "APPL_DB", "COPP_TABLE:{}".format(group_name))
            if not updated.get("cir"):
                duthost.shell(
                    "ctr tasks exec --exec-id copp_rst_{} or-con supervisorctl restart coppmgrd".format(int(time.time())),
                    module_ignore_errors=True, verbose=False)
                time.sleep(2)
            logger.warning("COPP_TABLE:%s had no cir policer; provisioned %s in CONFIG_DB", group_name, group_key)
            return []
    created = []
    if not redis_exists(duthost, "CONFIG_DB", DHCPV6_COPP_GROUP_KEY):
        config_db_set(duthost, DHCPV6_COPP_GROUP_KEY, DEFAULT_DHCPV6_COPP_GROUP_FIELDS)
        created.append(DHCPV6_COPP_GROUP_KEY)
    if not redis_exists(duthost, "CONFIG_DB", DHCPV6_COPP_TRAP_KEY):
        config_db_set(duthost, DHCPV6_COPP_TRAP_KEY, DEFAULT_DHCPV6_COPP_TRAP_FIELDS)
        created.append(DHCPV6_COPP_TRAP_KEY)
    logger.warning("No dhcpv6 CoPP trap in APPL_DB; provisioned %s for this session", created)
    return created


def remove_dhcpv6_copp_trap(duthost, created_keys):
    """Undo ``ensure_dhcpv6_copp_trap``."""
    if not created_keys:
        return
    for key in created_keys:
        config_db_del(duthost, key)


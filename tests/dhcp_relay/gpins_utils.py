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

import base64
import csv
import ipaddress
import logging
import os
import random
import re
import shlex
import time
from contextlib import contextmanager

from tests.common.helpers.assertions import pytest_assert
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

# DHCP-relay container of the daily images (/etc/init.d/common-container.sh DHCPRELAY_CONTAINER_NAME;
# verified on gpins_daily_20261022_02).
DHCP_RELAY_INIT_SCRIPT = "/etc/init.d/dhcprelay-container"
DHCP_RELAY_CONTAINER = "dhcprelay-con"


# Inside dhcprelay-con, supervisord runs the managers dhcprelayd/dhcprelaydv6;
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


# DHCP_RELAY options that are booleans in CONFIG_DB ("true"/"false").
_DHCP_RELAY_BOOL_FIELDS = ("rfc6939_support", "enable_interface_id", "enable_remote_id")


# ---------------------------------------------------------------------------
# Low-level Redis access
# ---------------------------------------------------------------------------

def db_id(db):
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
    parts = [REDIS_CLI, "-n", str(db_id(db)), "--raw"]
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


def redis_pubsub_channels(duthost, pattern="*", db="STATE_DB"):
    """Return active Redis PUBSUB channels matching ``pattern``."""
    return [
        line for line in redis_cmd_lines(duthost, db, "PUBSUB", "CHANNELS", pattern, module_ignore_errors=True)
        if line
    ]


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


def redis_del(duthost, db, *keys):
    """Delete one or more keys."""
    if keys:
        redis_cmd(duthost, db, "DEL", *keys)


def redis_pipeline(duthost, db, commands):
    """Send a batch of Redis commands in a single SSH call via ``redis-cli --pipe`` (with stdin fallback).

    ``commands`` is a list of argv lists, e.g.:
        ``[["HSET", "DHCP_RELAY|Ethernet1/4/1", "dhcpv6_servers", "2001:db8::1"], ["DEL", "DHCP_RELAY|Ethernet1/2/5"]]``

    Encodes the batch in Redis RESP protocol format, base64-encodes it to avoid
    shell/PTY quoting issues, and pipes stdout through ``| cat`` so the GPINS
    ``/usr/tools/bin/redis-cli`` wrapper does not allocate a container PTY (``-t``).
    """
    if not commands:
        return
    db_num = db_id(db)
    payload_lines = []
    for cmd in commands:
        payload_lines.append("*{}".format(len(cmd)))
        for arg in cmd:
            arg_str = str(arg)
            arg_bytes = arg_str.encode("utf-8")
            payload_lines.append("${}".format(len(arg_bytes)))
            payload_lines.append(arg_str)
    resp_payload = "\r\n".join(payload_lines) + "\r\n"
    b64_resp = base64.b64encode(resp_payload.encode("utf-8")).decode("ascii")
    pipe_cmd = "printf '%s' {} | base64 -d | {} -n {} --pipe | cat".format(
        shlex.quote(b64_resp), REDIS_CLI, db_num)

    res = None
    for attempt in range(1, REDIS_RETRIES + 1):
        res = duthost.shell(pipe_cmd, module_ignore_errors=True, verbose=False)
        if res.get("rc", 1) == 0 and "errors: 0" in res.get("stdout", ""):
            return
        if attempt < REDIS_RETRIES and _is_transient_ctr_error(res):
            logger.warning("redis_pipeline transient failure (attempt %d/%d): %s",
                           attempt, REDIS_RETRIES, res.get("stderr", "").strip())
            time.sleep(REDIS_RETRY_DELAY * attempt)
            continue
        break

    # Fallback: standard stdin batching if --pipe is unsupported or returned non-zero.
    fallback_lines = []
    for cmd in commands:
        fallback_lines.append(" ".join(
            '"{}"'.format(str(a).replace('"', '\\"')) if (" " in str(a) or "|" in str(a)) else str(a)
            for a in cmd
        ))
    fallback_input = "\n".join(fallback_lines) + "\n"
    b64_fallback = base64.b64encode(fallback_input.encode("utf-8")).decode("ascii")
    fallback_cmd = "printf '%s' {} | base64 -d | {} -n {} | cat".format(
        shlex.quote(b64_fallback), REDIS_CLI, db_num)
    res = duthost.shell(fallback_cmd, module_ignore_errors=True, verbose=False)
    if res.get("rc", 0) != 0:
        raise RuntimeError("redis_pipeline failed (rc={}): {}".format(
            res.get("rc"), res.get("stderr", "").strip()))


# One-shot table fetch: KEYS + HGETALL for every match inside Redis (Lua), so a
# 66-entry table costs one ssh/ctr round trip instead of 67. Output format
# (one element per line with --raw):  "==KEY <key>", "<n>", then n lines of
# alternating field/value. The explicit count keeps empty values unambiguous
# even though ansible strips trailing whitespace from stdout. Non-hash keys
# (CONFIG_DB has the string CONFIG_DB_INITIALIZED) are skipped: an HGETALL on
# one of them would abort the whole script with WRONGTYPE, which is what made a
# full "*" dump fall back to per-key reads (9 minutes on ju3u1m1, 2026-10-21).
_GET_TABLE_LUA = ("local out = {} "
                  "for _, k in ipairs(redis.call('KEYS', ARGV[1])) do "
                  "if redis.call('TYPE', k)['ok'] == 'hash' then "
                  "out[#out+1] = '==KEY ' .. k "
                  "local h = redis.call('HGETALL', k) "
                  "out[#out+1] = tostring(#h) "
                  "for _, v in ipairs(h) do out[#out+1] = v end "
                  "end "
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


def dump_db(duthost, db, pattern):
    """``{key: fields}`` for every hash in ``db`` whose key matches the glob ``pattern``, in one round trip.

    Falls back to KEYS + one HGETALL per key if the server rejects EVAL (older/locked-down redis).
    """
    cmd = build_redis_cmd(db, "EVAL", _GET_TABLE_LUA, "0", pattern)
    res = duthost.shell(cmd, module_ignore_errors=True, verbose=False)
    lines = [line for line in (res.get("stdout") or "").splitlines() if not line.startswith("ERRO[")]
    if res.get("rc", 1) == 0 and not (lines and lines[0].startswith("ERR")):
        return _parse_table_dump(lines, "")
    logger.warning("EVAL dump of %s %r failed on %s (rc=%s, reply=%r); falling back to per-key reads (slow)",
                   db, pattern, duthost.hostname, res.get("rc"),
                   (lines[0] if lines else (res.get("stderr") or ""))[:300])
    return {key: redis_hgetall(duthost, db, key) for key in redis_keys(duthost, db, pattern)}


def get_table(duthost, db, table):
    """Return every entry of a table as {key_without_table_prefix: fields}.

    Example: get_table(duthost, "CONFIG_DB", "DHCP_RELAY") ->
        {"Vlan1000": {"dhcpv6_servers": "fc02:2000::1,fc02:2000::2"}}
    """
    prefix = table + _db_separator(db)
    return {key[len(prefix):]: fields for key, fields in dump_db(duthost, db, prefix + "*").items()}


# ---------------------------------------------------------------------------
# CONFIG_DB helpers (replace `config ...` CLI and `config_facts`)
# ---------------------------------------------------------------------------

def config_db_set(duthost, key, fields=None):
    """Create or update a CONFIG_DB entry. Replacement for `config ...` commands."""
    redis_hset(duthost, "CONFIG_DB", key, fields or {})


def config_db_del(duthost, key):
    """Delete a CONFIG_DB entry."""
    redis_del(duthost, "CONFIG_DB", key)


def dump_config_db(duthost, pattern="*"):
    """``{key: fields}`` for every CONFIG_DB hash matching ``pattern`` (default: the whole DB)."""
    return dump_db(duthost, "CONFIG_DB", pattern)


def snapshot_config_db(duthost, patterns):
    """Save every CONFIG_DB entry matching the given key patterns (one round trip per pattern).

    Use together with ``restore_config_db`` in test setup/teardown.

    Returns:
        dict: {"patterns": [...], "entries": {key: fields}}
    """
    entries = {}
    for pattern in patterns:
        entries.update(dump_config_db(duthost, pattern))
    return {"patterns": list(patterns), "entries": entries}


def restore_config_db(duthost, snapshot):
    """Restore CONFIG_DB entries saved by ``snapshot_config_db``.

    Keys that match the snapshot patterns but did not exist at snapshot time
    are deleted, and saved keys are rewritten with exactly their saved fields.
    """
    saved = snapshot["entries"]
    current = {}
    for pattern in snapshot["patterns"]:
        current.update(dump_config_db(duthost, pattern))

    extra = sorted(set(current) - set(saved))
    if extra:
        logger.info("Removing CONFIG_DB keys added during test: %s", extra)
        redis_del(duthost, "CONFIG_DB", *extra)

    for key, fields in saved.items():
        if current.get(key, {}) == fields:
            continue
        # Rewrite the entry so fields added during the test are dropped too.
        redis_del(duthost, "CONFIG_DB", key)
        redis_hset(duthost, "CONFIG_DB", key, fields)


DHCP_RELAY_FEATURE = "dhcp_relay"


def feature_state(duthost, feature):
    """Return FEATURE|<feature> 'state' from CONFIG_DB (replacement for `show feature status`)."""
    return redis_hget(duthost, "CONFIG_DB", "FEATURE|{}".format(feature), "state")


def set_feature_state(duthost, feature, state):
    """Set FEATURE|<feature> 'state' ("enabled" / "disabled") in CONFIG_DB."""
    config_db_set(duthost, "FEATURE|{}".format(feature), {"state": state})


def dhcp_relay_available(duthost):
    """Return True if this GPINS build ships the DHCP-relay feature.

    The init script is the authoritative indicator: builds without the relay
    (e.g. gpins_release_20260824_05_prod_RC03) have no /etc/init.d/dhcprelay-container.
    The FEATURE|dhcp_relay row is only informational on GPINS and is not required here.
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


# How long the relay manager gets to mirror a CONFIG_DB DHCP_RELAY change into STATE_DB.
DHCP_RELAY_CONVERGE_TIMEOUT = 60


def assert_dhcp_relay_converged(duthost, interface, timeout=DHCP_RELAY_CONVERGE_TIMEOUT):
    """Assert STATE_DB DHCP_RELAY|<interface> matches CONFIG_DB within ``timeout`` seconds."""
    ok = wait_until(timeout, 3, 0, dhcp_relay_config_converged, duthost, interface)
    pytest_assert(ok, "DHCP_RELAY|{} not applied within {}s: CONFIG_DB={} STATE_DB={}".format(
        interface, timeout, get_dhcp_relay_config(duthost, interface), get_dhcp_relay_state(duthost, interface)))


def assert_dhcp_relay_state_cleared(duthost, interface, timeout=DHCP_RELAY_CONVERGE_TIMEOUT):
    """Assert STATE_DB DHCP_RELAY|<interface> disappears after the CONFIG_DB entry was removed."""
    ok = wait_until(timeout, 3, 0, lambda: not get_dhcp_relay_state(duthost, interface))
    pytest_assert(ok, "STATE_DB DHCP_RELAY|{} still present {}s after removal: {}".format(
        interface, timeout, get_dhcp_relay_state(duthost, interface)))


def get_dhcpv6_counters(duthost, interface):
    """Return the raw STATE_DB DHCPv6_COUNTER_TABLE|<interface> fields."""
    return redis_hgetall(duthost, "STATE_DB", "DHCPv6_COUNTER_TABLE|{}".format(interface))


def discover_candidate_interfaces(duthost, primary_intf=None, relay_mode=None, exclude_intfs=None):
    """Discover Ethernet* ports with a non-tentative IPv6 link-local (fe80::) address in CONFIG_DB '/' notation.

    Scans both kernel netdevs (``ip -6 -o addr show scope link``) and
    ``CONFIG_DB`` ``PORT|Ethernet*``. When ``relay_mode`` is ``l3_lla``, ports
    that have a global IPv6 address (GUA) in ``CONFIG_DB`` or the kernel are
    excluded so every candidate satisfies ``check_interface_mode(..., 'l3_lla')``.
    """
    exclude = set(exclude_intfs or ())
    res = duthost.shell("ip -6 -o addr show scope link", module_ignore_errors=True, verbose=False)
    valid_kernel_ifaces = set()
    tentative_kernel_ifaces = set()
    for line in res.get("stdout", "").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        k_ifname = parts[1].rstrip(":")
        if not k_ifname.startswith("Ethernet") or "inet6 fe80:" not in line:
            continue
        if "tentative" in line or "dadfailed" in line:
            tentative_kernel_ifaces.add(k_ifname)
        else:
            valid_kernel_ifaces.add(k_ifname)
    valid_kernel_ifaces -= tentative_kernel_ifaces

    candidates = {to_config_ifname(k) for k in valid_kernel_ifaces}
    for key in redis_keys(duthost, "CONFIG_DB", "PORT|Ethernet*"):
        db_intf = key.split("|", 1)[1]
        if to_kernel_ifname(db_intf) in valid_kernel_ifaces:
            candidates.add(db_intf)

    if relay_mode and normalize_relay_mode(relay_mode) == RELAY_MODE_L3_LLA:
        gua_intfs = set()
        for key in redis_keys(duthost, "CONFIG_DB", "INTERFACE|Ethernet*|*"):
            parts = key.split("|")
            if len(parts) >= 3 and ":" in parts[2]:
                gua_intfs.add(parts[1])
        gua_res = duthost.shell("ip -6 -o addr show scope global", module_ignore_errors=True, verbose=False)
        for line in gua_res.get("stdout", "").splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[1].rstrip(":").startswith("Ethernet"):
                gua_intfs.add(to_config_ifname(parts[1].rstrip(":")))
        candidates -= gua_intfs

    candidates -= exclude
    if primary_intf:
        candidates.add(primary_intf)
    sorted_list = sorted(candidates)
    logger.info("Discovered %d candidate interfaces with IPv6 LLA (primary=%s): %s",
                len(sorted_list), primary_intf, sorted_list)
    return sorted_list


def run_single_wave_db_burst(duthost, wave_idx, total_waves, candidate_intfs, primary_db_intf,
                             dhcpv6_servers, ops_per_wave=500, final_active_count=8,
                             primary_interface_id=None):
    """Execute one wave of random CONFIG_DB DHCP_RELAY burst churn and converge to a deterministic set.

    * Generates ``ops_per_wave`` random ``HSET`` (70%) and ``DEL`` (30%) commands
      across ``candidate_intfs`` with unique ``interface_id`` and ``remote_id``.
    * Converges at the end of the wave to ``final_active_intfs`` (always including
      ``primary_db_intf`` plus up to ``final_active_count - 1`` random candidates)
      and ``DEL`` on all remaining ``deleted_intfs``.
    * Sends the entire batch in a single ``redis_pipeline`` call.

    Returns ``(expected_configs, deleted_intfs)``.
    """
    if isinstance(dhcpv6_servers, (list, tuple, set)):
        servers_str = ",".join(str(s) for s in dhcpv6_servers)
    else:
        servers_str = str(dhcpv6_servers)

    logger.info("[Wave %d/%d] Starting CONFIG_DB DHCP_RELAY burst churn (%d random HSET/DEL ops across %d ports)...",
                wave_idx, total_waves, ops_per_wave, len(candidate_intfs))

    cmds = []
    for i in range(ops_per_wave):
        intf = random.choice(candidate_intfs)
        key = "DHCP_RELAY|{}".format(intf)
        if random.random() < 0.7:
            intf_tag = to_kernel_ifname(intf)
            remote_id = "remote_id_{}w{}{}".format(intf_tag, wave_idx, i)
            cmds.append([
                "HSET", key,
                "dhcpv6_servers", servers_str,
                "enable_interface_id", "true",
                "interface_id", intf,
                "enable_remote_id", "true",
                "remote_id", remote_id,
            ])
        else:
            cmds.append(["DEL", key])

    other_intfs = [x for x in candidate_intfs if x != primary_db_intf]
    random.shuffle(other_intfs)
    selected_others = other_intfs[:max(0, min(final_active_count - 1, len(other_intfs)))]
    final_active_intfs = [primary_db_intf] + selected_others
    deleted_intfs = [x for x in candidate_intfs if x not in final_active_intfs]

    for intf in candidate_intfs:
        cmds.append(["DEL", "DHCP_RELAY|{}".format(intf)])

    expected_configs = {}
    for idx, intf in enumerate(final_active_intfs, start=1):
        key = "DHCP_RELAY|{}".format(intf)
        iid = (primary_interface_id or intf) if intf == primary_db_intf else intf
        cfg = {
            "dhcpv6_servers": servers_str,
            "enable_interface_id": "true",
            "interface_id": iid,
            "enable_remote_id": "true",
            "remote_id": "remote_id_w{}_{:03d}".format(wave_idx, idx),
        }
        expected_configs[intf] = cfg
        cmds.append([
            "HSET", key,
            "dhcpv6_servers", cfg["dhcpv6_servers"],
            "enable_interface_id", cfg["enable_interface_id"],
            "interface_id", cfg["interface_id"],
            "enable_remote_id", cfg["enable_remote_id"],
            "remote_id", cfg["remote_id"],
        ])

    t0 = time.time()
    redis_pipeline(duthost, "CONFIG_DB", cmds)
    elapsed_ms = (time.time() - t0) * 1000.0
    logger.info("[Wave %d/%d] CONFIG_DB burst complete: %d pipeline ops in %.1f ms (active=%s, deleted=%d ports)",
                wave_idx, total_waves, len(cmds), elapsed_ms, final_active_intfs, len(deleted_intfs))
    return expected_configs, deleted_intfs


def wait_and_verify_state_db(duthost, expected_configs, deleted_intfs, timeout_sec=30.0):
    """Wait for dhcprelaydv6/dhcp6relay to settle; verify STATE_DB matches ``expected_configs``, no stale keys."""
    deadline = time.time() + timeout_sec
    last_mismatch_reason = ""

    while time.time() < deadline:
        pid = get_process_pid(duthost)
        if not pid:
            last_mismatch_reason = "dhcp6relay process is not running (pgrep -x dhcp6relay is empty)"
            time.sleep(0.5)
            continue

        state_table = get_table(duthost, "STATE_DB", "DHCP_RELAY")
        all_matched = True
        for intf, exp in expected_configs.items():
            state_key = "DHCP_RELAY|{}".format(intf)
            actual = state_table.get(intf)
            if not actual:
                all_matched = False
                last_mismatch_reason = "STATE_DB missing key {}".format(state_key)
                break
            for field, exp_val in exp.items():
                act_val = actual.get(field)
                if act_val != exp_val:
                    all_matched = False
                    last_mismatch_reason = "{} field {} mismatch: expected={!r}, actual={!r}".format(
                        state_key, field, exp_val, act_val)
                    break
            if not all_matched:
                break

        if all_matched:
            state_keys = {"DHCP_RELAY|{}".format(k) for k in state_table}
            expected_keys = {"DHCP_RELAY|{}".format(intf) for intf in expected_configs}
            stale_keys = state_keys - expected_keys
            if stale_keys:
                all_matched = False
                last_mismatch_reason = "STATE_DB has uncleaned stale keys: {}".format(sorted(stale_keys))

        if all_matched:
            time.sleep(1.0)
            pid_after = get_process_pid(duthost)
            if pid_after == pid:
                logger.info("STATE_DB verified (dhcp6relay stable PID=%s, %d active interfaces, %d deleted cleaned)",
                            pid, len(expected_configs), len(deleted_intfs))
                return True
            last_mismatch_reason = "dhcp6relay PID still changing ({} -> {})".format(pid, pid_after)

        time.sleep(0.5)

    raise AssertionError("STATE_DB Applied Configuration verification failed after {}s: {}".format(
        timeout_sec, last_mismatch_reason))


# ---------------------------------------------------------------------------
# ACL helpers (replace `config acl add table` / `acl-loader`)
#
# Observed on ju5u1m1: orchagent consumes CONFIG_DB ACL_TABLE|<name> and
# reports STATE_DB ACL_TABLE_TABLE|<name> status=Active (likewise
# ACL_RULE_TABLE|<table>|<rule>). That status is used as the convergence signal.
# ---------------------------------------------------------------------------

ACL_STATUS_ACTIVE = "Active"


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

def parse_container_tasks(text):
    """``{task_name: status}`` from ``ctr tasks ls`` output (the header line is skipped).

    Example output parsed:
        TASK        PID     STATUS
        db-con      5867    RUNNING
        lldp-con    12206   RUNNING
    """
    tasks = {}
    for line in (text or "").splitlines()[1:]:
        cols = line.split()
        if len(cols) >= 3:
            tasks[cols[0]] = cols[2]
    return tasks


def get_container_tasks(duthost):
    """Return {task_name: status} from ``ctr tasks ls``."""
    res = duthost.shell("{} tasks ls".format(CTR), module_ignore_errors=True, verbose=False)
    return parse_container_tasks(res["stdout"])


def is_container_running(duthost, container=DHCP_RELAY_CONTAINER):
    """Return True if the containerd task is RUNNING (defaults to the DHCP-relay container)."""
    return get_container_tasks(duthost).get(container) == "RUNNING"


def ctr_exec(duthost, command, container, module_ignore_errors=True):
    """Run a command inside a GPINS container (the ``docker exec`` equivalent).

    Mirrors the redis-cli wrapper: ``ctr tasks exec --exec-id <unique> <task> <cmd>``.
    """
    exec_id = "sonicmgmt{}".format(time.time_ns())
    cmd = "{} tasks exec --exec-id {} {} {}".format(CTR, exec_id, container, command)
    return duthost.shell(cmd, module_ignore_errors=module_ignore_errors, verbose=False)


def shell_stdout(duthost, cmd):
    """Run ``cmd`` on the DUT and return its stdout, or None if the DUT is unreachable or ``cmd`` failed.

    Safe to poll across a reboot: connection errors are swallowed. The Ansible result is
    duck-typed on purpose -- pytest-ansible's ``ModuleResult`` is a ``collections.UserDict``,
    not a ``dict`` subclass, so ``isinstance(res, dict)`` is False even on success.
    """
    try:
        res = duthost.shell(cmd, module_ignore_errors=True, verbose=False)
        if res is None or res.get("unreachable") or res.get("rc") != 0:
            return None
        return res.get("stdout", "") or ""
    except Exception as exc:
        logger.debug("shell on %s failed (%s): %s", duthost.hostname, cmd, exc)
        return None


# Init script and containerd task of every GPINS container the suite (re)starts, keyed by the component name
# container_monitor uses in STATE_DB COMPONENT_STATE_TABLE. common-container.sh operations (verified
# 2026-10-08): "start" imports the image on demand and launches the container (~1.5 s), "stop" removes
# task + container, "restart" = stop + start; nothing on the switch watches FEATURE to do this.
COMPONENT_INIT_SCRIPTS = {
    "dhcp_relay": DHCP_RELAY_INIT_SCRIPT,
    "framework": "/etc/init.d/framework-container",
    "telemetry": "/etc/init.d/telemetry-container",
    "p4rt": "/etc/init.d/p4rt-container",
    "hst": "/etc/init.d/hst-container",
    "linkqual": "/etc/init.d/linkqual-container",
    "lldp": "/etc/init.d/lldp-container",
    "sflow": "/etc/init.d/sflow-container",
    "teamd": "/etc/init.d/teamd-container",
    "bgp": "/etc/init.d/bgp-container",
    "swss": "/etc/init.d/orchagent-container",
    "syncd": "/etc/init.d/syncd-container",
    "database": "/etc/init.d/redis-container",
}
# container_monitor component name -> containerd task name (common-container.sh *_CONTAINER_NAME).
COMPONENT_CONTAINERS = {
    "dhcp_relay": DHCP_RELAY_CONTAINER,
    "framework": "framework-con",
    "telemetry": "telemetry-con",
    "p4rt": "p4rt-con",
    "hst": "hst-con",
    "linkqual": "linkqual-con",
    "lldp": "lldp-con",
    "sflow": "sflow-con",
    "teamd": "teamd-con",
    "bgp": "bgp-con",
    "swss": "or-con",
    "syncd": "sy-con",
    "database": "db-con",
    "ptp": "ptp-con",
}


def run_component_init(duthost, component, action, module_ignore_errors=True):
    """Run ``<init script> <action>`` (``start``/``stop``/``restart``) for ``component``; returns ``(action, rc)``.

    A ``restart`` of a container that is not running is sent as ``start``.
    """
    script = COMPONENT_INIT_SCRIPTS.get(component)
    if script is None:
        raise ValueError("no init script known for component {!r}".format(component))
    container = COMPONENT_CONTAINERS.get(component)
    if action == "restart" and container and not is_container_running(duthost, container):
        action = "start"
    logger.info("%s %s on %s via %s", action, container or component, duthost.hostname, script)
    res = duthost.shell("{} {}".format(script, action), module_ignore_errors=module_ignore_errors)
    return action, res.get("rc")


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


def is_dhcp6relay_bound(duthost):
    """Return True if dhcp6relay is running and listening (the "relay is ready" gate).

    "Listening" means it owns a UDP 547 socket (upstream) or an AF_PACKET socket (GPINS).
    """
    info = dhcp6relay_listeners(duthost)
    if info["pid"] is None:
        return False
    return bool(info[LISTENER_UDP]) or info[LISTENER_PACKET] > 0


def wait_dhcp6relay_bound(duthost, timeout=120, interval=5):
    """Wait until dhcp6relay is running and listening. Returns True on success."""
    return wait_until(timeout, interval, 0, is_dhcp6relay_bound, duthost)


def start_dhcp_relay(duthost, timeout=60):
    """Start the DHCP-relay container via its init script (no-op if already running).

    GPINS has no daemon that starts the container from FEATURE|dhcp_relay, so
    this is the only enable path. Returns True if the container is running afterwards.
    """
    container = DHCP_RELAY_CONTAINER
    if is_container_running(duthost, container):
        return True
    run_component_init(duthost, "dhcp_relay", "start", module_ignore_errors=False)
    return wait_until(timeout, 3, 0, is_container_running, duthost, container)


def stop_dhcp_relay(duthost, timeout=60):
    """Stop and remove the DHCP-relay container via its init script; True once the containerd task is gone."""
    container = DHCP_RELAY_CONTAINER
    if not is_container_running(duthost, container):
        return True
    run_component_init(duthost, "dhcp_relay", "stop", module_ignore_errors=False)
    return wait_until(timeout, 3, 0, lambda: not is_container_running(duthost, container))


def restart_dhcp_relay(duthost, wait=True, timeout=120):
    """Restart the DHCP-relay container via its GPINS init script.

    Replacement for ``systemctl reset-failed/restart dhcp_relay``.

    Args:
        wait: wait for the container task to be RUNNING again.
        timeout: seconds to wait.

    Returns:
        bool: True if the container is running afterwards (always True if wait=False).
    """
    container = DHCP_RELAY_CONTAINER
    run_component_init(duthost, "dhcp_relay", "restart", module_ignore_errors=False)
    return not wait or wait_until(timeout, 3, 0, is_container_running, duthost, container)


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


def kernel_ipv6_addrs(duthost, dev, scope):
    """IPv6 CIDRs of kernel netdev ``dev`` in ``scope`` (``link``/``global``), in kernel order ([] if none)."""
    out = shell_stdout(duthost, "ip -6 addr show dev {} scope {} 2>/dev/null | awk '/inet6/ {{print $2}}'".format(
        shlex.quote(dev), shlex.quote(scope)))
    return [line.strip() for line in (out or "").splitlines() if line.strip()]


def get_port_link_local(duthost, port):
    """Return (lla, lla_with_prefix) of the kernel netdev for ``port``, or ("", "")."""
    addrs = kernel_ipv6_addrs(duthost, to_kernel_ifname(port), "link")
    if not addrs:
        return "", ""
    return addrs[0].split("/")[0], addrs[0]


def get_port_kernel_guas(duthost, port):
    """Return list of global-scope IPv6 CIDRs (``['2001:db8:1::1/64', ...]``) on the kernel netdev for ``port``."""
    return kernel_ipv6_addrs(duthost, to_kernel_ifname(port), "global")


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
        if not has_l3_interface(duthost, interface):
            return False, ("{} has no CONFIG_DB INTERFACE|{} row, so it is not an L3 (RIF) port; the ASIC would "
                           "flood DHCPv6 instead of punting it".format(interface, interface))
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


INTERFACE_BASE_KEY = "INTERFACE|{}"
# Upstream SONiC's way of declaring an L3 port that only uses its link-local address
# (config interface ipv6 enable use-link-local-only). intfmgrd creates the RIF for the
# base row regardless of the field, so this is safe on builds that ignore it.
L3_LLA_ONLY_FIELDS = {"ipv6_use_link_local_only": "enable"}


def has_l3_interface(duthost, interface):
    """True if CONFIG_DB has the base ``INTERFACE|<interface>`` row (i.e. the port is an L3 RIF)."""
    return bool(redis_keys(duthost, "CONFIG_DB", INTERFACE_BASE_KEY.format(interface)))


def ensure_l3_interface(duthost, interface):
    """Create ``INTERFACE|<interface>`` if missing so the port becomes an L3 interface.

    Returns True if the row was created by this call (caller owns its removal), False if
    it already existed (left untouched, including its fields).
    """
    if has_l3_interface(duthost, interface):
        return False
    logger.info("Creating %s %s so %s is an L3 (RIF) port", INTERFACE_BASE_KEY.format(interface),
                L3_LLA_ONLY_FIELDS, interface)
    config_db_set(duthost, INTERFACE_BASE_KEY.format(interface), L3_LLA_ONLY_FIELDS)
    return True


def set_interface_ipv6(duthost, interface, address_with_plen):
    """Configure an IPv6 GUA on ``interface`` in CONFIG_DB + kernel (Appendix step 1).

    Equivalent of:
        redis-cli -n 4 HSET "INTERFACE|Ethernet1/4/1|2001:db8:1::1/64" "NULL" "NULL"
        ip -6 addr replace 2001:db8:1::1/64 dev Ethernet1_4_1
    """
    sep = _db_separator("CONFIG_DB")
    base_key = "INTERFACE{}{}".format(sep, interface)
    addr_key = "INTERFACE{}{}{}".format(sep, interface, sep) + address_with_plen
    config_db_set(duthost, base_key)
    config_db_set(duthost, addr_key)
    dev = to_kernel_ifname(interface)
    duthost.shell("ip -6 addr replace {} dev {}".format(shlex.quote(address_with_plen), shlex.quote(dev)),
                  module_ignore_errors=True)
    return [base_key, addr_key]


def del_interface_ipv6(duthost, interface, address_with_plen):
    """Remove an IPv6 GUA from ``interface`` in CONFIG_DB + kernel."""
    sep = _db_separator("CONFIG_DB")
    addr_key = "INTERFACE{}{}{}".format(sep, interface, sep) + address_with_plen
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


def relay_targets(duthost, mode, ports, tbinfo):
    """Discover relay interfaces for ``mode``.

    Returns a list of dicts ``{"relay_interface", "client_port", "gua", "gua_plen"}``:
      * vlan:   one per VLAN that has a physical member and an IPv6 VLAN_INTERFACE
      * l3_gua: one per downlink port with an IPv6 INTERFACE entry
      * l3_lla: one per downlink port that has *no* GUA in CONFIG_DB or the kernel
    Only downlinks that are oper up - and in the testbed's ``ptf_map`` when one is defined - are
    considered, so the first target is always a port the PTF can actually reach.
    An empty list means the DUT is not configured for that mode.
    """
    mode = normalize_relay_mode(mode)
    downlinks, _ = classify_ports(ports, duthost, tbinfo)
    downlinks = usable_downlinks(duthost, downlinks, tbinfo)
    targets = []
    if mode == RELAY_MODE_VLAN:
        for vlan, info in sorted(get_vlans(duthost).items()):
            members = [m for m in info["members"] if not m.startswith("PortChannel") and m in downlinks]
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


def configure_relay_mode(duthost, mode, tbinfo):
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

    ports = get_ports(duthost)
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
        # Two things make a port an LLA-only relay interface: no GUA anywhere, and an
        # INTERFACE|<port> row so intfmgrd/orchagent create the L3 RIF. Without the RIF
        # the ASIC treats the port as L2 and floods the client's Solicit out the uplinks
        # instead of punting it (seen on a freshly provisioned ju3u1m1, 2026-10-20).
        target_ports = downlinks[:2]
        patterns = ([INTERFACE_BASE_KEY.format(p) for p in target_ports] +
                    ["INTERFACE|{}|*".format(p) for p in downlinks])
        snap = snapshot_config_db(duthost, patterns)
        state["snapshot"] = snap
        ipv6_keys = [k for k in snap["entries"] if ":" in k.rsplit("|", 1)[-1]]
        removed_set = set()
        for key in ipv6_keys:
            _, port, cidr = key.split("|", 2)
            del_interface_ipv6(duthost, port, cidr)
            state["removed_kernel_addrs"].append((port, cidr))
            removed_set.add((port, cidr))
        for port in target_ports:
            for cidr in get_port_kernel_guas(duthost, port):
                dev = to_kernel_ifname(port)
                duthost.shell("ip -6 addr del {} dev {}".format(shlex.quote(cidr), shlex.quote(dev)),
                              module_ignore_errors=True, verbose=False)
                if (port, cidr) not in removed_set:
                    state["removed_kernel_addrs"].append((port, cidr))
                    removed_set.add((port, cidr))
            if ensure_l3_interface(duthost, port):
                state.setdefault("added_interface_keys", []).append(INTERFACE_BASE_KEY.format(port))
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


# Accessors for one ``dut_dhcp_relay_data`` entry (test_dhcpv6_relay.build_relay_data).

def relay_servers(dhcp_relay):
    """DHCPv6 server addresses the relay forwards to."""
    return list(dhcp_relay['downlink_vlan_iface']['dhcpv6_server_addrs'])


def design_interface_id(dhcp_relay):
    """Interface-ID in the form the sample config prescribes: ``<port>`` (e.g. ``Ethernet1/4/1``)."""
    return dhcp_relay['interface_id'] or dhcp_relay['relay_interface']


def lla_relay_options(dhcp_relay):
    """Options an LLA-only relay must always carry (the server routes replies via the loopback in Option 18)."""
    if dhcp_relay['relay_mode'] == RELAY_MODE_L3_LLA:
        return {"enable_interface_id": True, "interface_id": design_interface_id(dhcp_relay)}
    return {}


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


def add_static_server_path(duthost, server, uplink_port, nexthop_mac):
    """Steer traffic for ``server`` out of ``uplink_port`` without BGP.

    The server is treated as on-link behind the uplink (it is the PTF port there): a permanent neighbour
    entry for the server address with ``nexthop_mac`` plus an on-link /128 route.
    Returns the list of (kind, spec) entries created so they can be removed.
    """
    dev = to_kernel_ifname(uplink_port)
    duthost.shell("ip -6 neigh replace {} lladdr {} dev {} nud permanent".format(
        shlex.quote(str(server)), shlex.quote(nexthop_mac), shlex.quote(dev)))
    duthost.shell("ip -6 route replace {}/128 dev {}".format(shlex.quote(str(server)), shlex.quote(dev)))
    created = [("neigh", "{} dev {}".format(server, dev)), ("route", "{}/128 dev {}".format(server, dev))]
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
# DHCPv6 CoPP trap + rate limit
# ---------------------------------------------------------------------------
#
# The 2026-10 images (zhouboyang_dhcp_test 10-14, the dailies) trap "dhcp,dhcpv6" into queue20_group1 via
# copp_cfg.json WITHOUT a policer; CPU queue 20 is shaped by SCHEDULER|default_cpu_queue_scheduler (pir 1000 pps,
# packets). Nothing is provisioned: what the image polices is what the CoPP test measures (dhcpv6_copp_rate_limit()).
# Adding a 1600 pps policer on top of the 1000 pps queue shaper only measured the shaper (1120-1145 pps) and leaked
# a COPP_GROUP row into the module config check.


def dhcpv6_copp_trap_group(duthost):
    """APPL_DB ``COPP_TABLE`` group trapping dhcpv6/dhcpv6_l2 as ``(name, fields)``, or ``(None, {})``."""
    for group_name, entry in get_table(duthost, "APPL_DB", "COPP_TABLE").items():
        trap_ids = [t.strip() for t in entry.get("trap_ids", "").split(",") if t.strip()]
        if "dhcpv6" in trap_ids or "dhcpv6_l2" in trap_ids:
            return group_name, entry
    return None, {}


def _cpu_queue_shaper(duthost, queue):
    """``(scheduler_name, pir_pps)`` of the packet-mode shaper on CPU queue ``queue`` (CONFIG_DB
    ``QUEUE|CPU|<q>`` -> ``SCHEDULER|<name>``); ``pir_pps`` is None when there is none or it is not in packets."""
    if queue in (None, ""):
        return None, None
    row = redis_hgetall(duthost, "CONFIG_DB", "QUEUE|CPU|{}".format(queue))
    name = (row.get("scheduler") or "").strip("[]").split("|")[-1]   # plain name, or legacy "[SCHEDULER|name]"
    if not name:
        return None, None
    sched = redis_hgetall(duthost, "CONFIG_DB", "SCHEDULER|{}".format(name))
    if sched.get("meter_type", "packets") != "packets":
        logger.warning("CPU queue %s scheduler %s meters in %s, not packets; not usable as a pps limit",
                       queue, name, sched.get("meter_type"))
        return name, None
    try:
        return name, int(sched["pir"])
    except (KeyError, ValueError):
        return name, None


def dhcpv6_copp_rate_limit(duthost):
    """What caps the DHCPv6 punt rate on *this* image, read from the switch (nothing is written).

    Two mechanisms can cap it on GPINS: the trap group's SAI policer (``cir`` in APPL_DB
    ``COPP_TABLE:<group>``, packets) and the CPU queue's packet-mode shaper (CONFIG_DB ``QUEUE|CPU|<queue>``
    -> ``SCHEDULER|<name>`` ``pir``). The effective limit is the smaller of the ones that exist.

    Returns ``{"group", "queue", "policer_cir", "scheduler", "scheduler_pir", "pps", "source"}``;
    ``group`` is None when the image has no DHCPv6 trap, ``pps``/``source`` are None when nothing limits it
    (``source`` is ``"policer"`` or ``"scheduler"``).
    """
    group, entry = dhcpv6_copp_trap_group(duthost)
    info = {"group": group, "queue": entry.get("queue"), "policer_cir": None,
            "scheduler": None, "scheduler_pir": None, "pps": None, "source": None}
    if not group:
        return info
    if entry.get("cir") and entry.get("meter_type", "packets") == "packets":
        try:
            info["policer_cir"] = int(entry["cir"])
        except ValueError:
            pass
    info["scheduler"], info["scheduler_pir"] = _cpu_queue_shaper(duthost, info["queue"])
    limits = [(pps, src) for pps, src in ((info["policer_cir"], "policer"), (info["scheduler_pir"], "scheduler"))
              if pps]
    if limits:
        info["pps"], info["source"] = min(limits)
    return info


# ---------------------------------------------------------------------------
# Lab runtime state (re-applied after an NSF reboot or a DPB stack restart)
# ---------------------------------------------------------------------------

def reapply_lab_runtime_state(duthost, relay_data, ensure_routes_fn=None):
    """Put back the lab state the suite relies on after a kernel NSF reboot or a ``gpins-stack restart``.

    Neither keeps kernel-only IPv6 addresses (Loopback0, the l3_gua port GUA) or the static PTF routes, and
    ``FEATURE|dhcp_relay`` and the relay container are re-asserted as well.
    ``ensure_routes_fn`` is the callable the ``validate_dut_routes_exist`` fixture yields (static server routes).
    """
    dhcp_relay = relay_data[0]
    # 1. Loopback0 IPv6 in CONFIG_DB + kernel (dhcp6relay needs it once Interface-ID is enabled).
    for lb in dhcp_relay.get("loopback_iface", []):
        lb_name = lb.get("name") or DEFAULT_LOOPBACK
        lb_addr = lb.get("addr")
        lb_plen = lb.get("prefixlen") or 128
        if lb_addr:
            cidr = "{}/{}".format(lb_addr, lb_plen)
            config_db_set(duthost, "LOOPBACK_INTERFACE|{}".format(lb_name))
            config_db_set(duthost, "LOOPBACK_INTERFACE|{}|{}".format(lb_name, cidr))
            duthost.shell("ip -6 addr replace {} dev {}".format(shlex.quote(cidr), shlex.quote(lb_name)),
                          module_ignore_errors=True)

    # 2. l3_gua: the relay interface's kernel GUA back; l3_lla: make sure no GUA came back.
    for entry in relay_data:
        intf = entry["relay_interface"]
        mode = entry["relay_mode"]
        gua = entry["downlink_vlan_iface"].get("addr")
        mask = entry["downlink_vlan_iface"].get("mask")
        if mode == RELAY_MODE_L3_GUA and gua and mask:
            set_interface_ipv6(duthost, intf, "{}/{}".format(gua, mask))
        elif mode == RELAY_MODE_L3_LLA:
            for cidr in get_port_kernel_guas(duthost, intf):
                dev = to_kernel_ifname(intf)
                duthost.shell("ip -6 addr del {} dev {}".format(shlex.quote(cidr), shlex.quote(dev)),
                              module_ignore_errors=True, verbose=False)

    # 3. FEATURE|dhcp_relay=enabled, static PTF server routes/neighbours.
    set_feature_state(duthost, "dhcp_relay", "enabled")
    if callable(ensure_routes_fn):
        ensure_routes_fn()

    # 4. The DHCP-relay container (on lab switches without /keys/ca_crt.pem gpins-stack stops before Phase 2b,
    #    so nothing restarts it after a kernel reboot).
    pytest_assert(start_dhcp_relay(duthost, timeout=120),
                  "{} did not start after the reboot/stack restart".format(DHCP_RELAY_CONTAINER))


# ---------------------------------------------------------------------------
# OpenConfig gNMI Telemetry (telemetry-con / openconfig-relay-agent)
# ---------------------------------------------------------------------------

TELEMETRY_CONTAINER = COMPONENT_CONTAINERS["telemetry"]
TELEMETRY_PORT = 9339

OPENCONFIG_DHCPV6_COUNTER_TO_STATE_DB = {
    "dhcpv6-solicit-received": "Solicit",
    "dhcpv6-adverstise-sent": "Advertise",
    "dhcpv6-request-received": "Request",
    "dhcpv6-confirm-received": "Confirm",
    "dhcpv6-renew-received": "Renew",
    "dhcpv6-rebind-received": "Rebind",
    "dhcpv6-reply-sent": "Reply",
    "dhcpv6-release-received": "Release",
    "dhcpv6-decline-received": "Decline",
    "dhcpv6-reconfigure-sent": "Reconfigure",
    "dhcpv6-info-request-received": "Information-Request",
    "dhcpv6-relay-forw-sent": "Relay-Forward",
    "dhcpv6-relay-reply-received": "Relay-Reply",
    "invalid-opcode": "Unknown",
    "total-dropped": "Malformed",
}

OPENCONFIG_DHCPV6_INTERFACE_KEYS = frozenset({"id", "config", "state", "options"})
# GPINS augment (google-pins-relay-agent) names of counters, mapped onto the names above (ju3u1m1, 2026-10-26).
OPENCONFIG_DHCPV6_COUNTER_ALIASES = {"pins-dhcpv6-renew-received": "dhcpv6-renew-received"}
OPENCONFIG_DHCPV6_CONFIG_KEYS = frozenset({"id", "helper-address"})
OPENCONFIG_DHCPV6_STATE_KEYS = frozenset({"id", "helper-address", "counters"})
OPENCONFIG_DHCPV6_OPTION_KEYS = frozenset({
    "enable-client-mac",
    # The GPINS augment (google-pins-relay-agent) names the RFC 6939 leaf pins-enable-client-mac (ju3u1m1, 2026-10-26).
    "pins-enable-client-mac",
    "enable-interface-id",
    "enable-remote-id",
    "interface-id",
    "remote-id",
})


def _strip_oc_prefix(key):
    """Strip an OpenConfig module prefix (e.g. ``openconfig-relay-agent-ext:foo`` -> ``foo``)."""
    return key.split(":", 1)[1] if ":" in key else key


# Names of the RFC 6939 (client link-layer address) option leaf: upstream augment, GPINS augment.
OPENCONFIG_CLIENT_MAC_LEAVES = ("enable-client-mac", "pins-enable-client-mac")


def openconfig_client_mac(options):
    """Value of the client-MAC (RFC 6939) leaf in a normalized options/config or options/state dict (None if absent)."""
    for leaf in OPENCONFIG_CLIENT_MAC_LEAVES:
        if leaf in options:
            return options[leaf]
    return None


def normalize_openconfig_dict(obj):
    """Recursively strip module prefixes from dict keys in an OpenConfig JSON tree."""
    if isinstance(obj, dict):
        return {_strip_oc_prefix(k): normalize_openconfig_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_openconfig_dict(item) for item in obj]
    return obj


# CONFIG_DB rows telemetry.sh renders the gNMI server flags from (telemetry_vars.j2; the GNMI|* rows are ignored --
# verified on ju3u1m1 2026-10-21/22): with TELEMETRY|certs present the server gets --server_crt/--server_key plus
# --ca_crt (client certificates required, "mTLS") only when the ca_crt field exists; with no row at all it starts
# with --insecure (plain gRPC). The lab CA == server cert is CN-only, which Go's client rejects, so every local
# CLI use needs one of the two reduced modes of gnmi_cli_session(). supervisord runs telemetry.sh, so a
# `telemetry-container restart` re-renders the flags from the current rows.
TELEMETRY_CERTS_ROW = "TELEMETRY|certs"
TELEMETRY_GNMI_ROW = "TELEMETRY|gnmi"
TELEMETRY_MTLS_FIELDS = ("ca_crt", "cert_crl_dir")
# Policy fields of TELEMETRY|gnmi that the "+no-policies" modes drop for the session (the configuration the
# telemetry test has always run with); fed_policy renders no server flag on the 2026-10 builds but is dropped too.
TELEMETRY_POLICY_FIELDS = ("authz_policy", "pathz_policy", "fed_policy")
TELEMETRY_POLICY_FLAGS = ("--authz_policy_enabled", "--gnmi_pathz_enabled")
# Client flags per server mode (measured on ju3u1m1 2026-10-22 against a TLS server without client auth:
# gnmi_get -insecure OK / -notls EOF; gnmi_cli -tls_skip_verify OK / -insecure dial timeout). The openconfig
# gnmi_cli's -insecure is PLAIN gRPC and -tls_skip_verify is TLS without verification; sonic-gnmi gnmi_get /
# gnmi_set: -notls = plain gRPC, -insecure = TLS without verification. The flag must match the server mode.
GNMI_ACCESS_MODES = ("server-tls+no-policies", "plaintext")
GNMI_SERVING_PROBES = {
    "server-tls": "gnmi_cli -address 127.0.0.1:{port} -tls_skip_verify -timeout 10s -capabilities",
    "plaintext": "gnmi_cli -address 127.0.0.1:{port} -insecure -timeout 10s -capabilities",
}


def telemetry_cmdline(duthost):
    """Command line of the gNMI server process (``""`` while it is not running). Container processes are
    visible in the host PID namespace, so this is a plain host-side read."""
    out = shell_stdout(duthost, "for p in $(pgrep -x telemetry); do tr '\\0' ' ' < /proc/$p/cmdline; echo; done")
    for line in (out or "").splitlines():
        if line.strip().startswith("/usr/sbin/telemetry"):
            return line.strip()
    return ""


def telemetry_mtls_enabled(cmdline):
    """True if the server requires client certificates (``--ca_crt`` without ``--insecure``/``--noTLS``)."""
    flags = (cmdline or "").split()
    return "--ca_crt" in flags and not ({"--insecure", "--noTLS"} & set(flags))


def telemetry_plaintext(cmdline):
    """True if the server runs plain gRPC (``--insecure``)."""
    return "--insecure" in (cmdline or "").split()


def restart_telemetry(duthost, timeout=120, expect_mtls=None):
    """``/etc/init.d/telemetry-container restart`` (``start`` if it is down) and wait for the server process.

    With ``expect_mtls`` True/False also wait until the new command line has/lacks ``--ca_crt``, i.e. until
    supervisord really re-ran ``telemetry.sh`` against the current CONFIG_DB. Returns the command line.
    """
    action, _ = run_component_init(duthost, "telemetry", "restart")
    state = {"cmdline": ""}

    def _server_up():
        state["cmdline"] = telemetry_cmdline(duthost)
        if not state["cmdline"]:
            return False
        return expect_mtls is None or telemetry_mtls_enabled(state["cmdline"]) == expect_mtls

    if not wait_until(timeout, 3, 5, _server_up):
        raise RuntimeError("gNMI server in {} on {} not up{} {}s after {} (cmdline: {!r})".format(
            TELEMETRY_CONTAINER, duthost.hostname,
            "" if expect_mtls is None else " with mTLS {}".format("on" if expect_mtls else "off"), timeout, action,
            state["cmdline"]))
    logger.info("gNMI server on %s: %s", duthost.hostname, state["cmdline"])
    return state["cmdline"]


def gnmi_server_access_state(duthost):
    """``{row: fields}`` of the CONFIG_DB rows telemetry.sh reads plus the live server command line."""
    return {TELEMETRY_CERTS_ROW: redis_hgetall(duthost, "CONFIG_DB", TELEMETRY_CERTS_ROW),
            TELEMETRY_GNMI_ROW: redis_hgetall(duthost, "CONFIG_DB", TELEMETRY_GNMI_ROW),
            "cmdline": telemetry_cmdline(duthost)}


def write_config_db_rows(duthost, rows):
    """Make each CONFIG_DB row exactly ``fields`` (``{}`` = delete the row) in one pipeline."""
    commands = []
    for key, fields in rows.items():
        commands.append(["DEL", key])
        if fields:
            flat = []
            for field, value in sorted(fields.items()):
                flat.extend([field, value])
            commands.append(["HSET", key] + flat)
    if commands:
        redis_pipeline(duthost, "CONFIG_DB", commands)


def gnmi_serving_probe(access="server-tls"):
    """Read-only ``gnmi_cli -capabilities`` command whose transport flag matches the server mode ``access``."""
    key = "plaintext" if access.startswith("plaintext") else "server-tls"
    return GNMI_SERVING_PROBES[key].format(port=TELEMETRY_PORT)


def wait_gnmi_serving(duthost, access="server-tls", timeout=120, interval=5):
    """Repeat the serving probe until the gNMI server answers. The server process is up long before it serves
    (it loads the YANG models first; 60 s was not enough on the 2026-10 builds). Returns ``(ok, detail)``."""
    cmd = gnmi_serving_probe(access)
    deadline = time.time() + timeout
    started = time.time()
    while True:
        res = ctr_exec(duthost, cmd, container=TELEMETRY_CONTAINER, module_ignore_errors=True)
        if res.get("rc") == 0:
            logger.info("gNMI server on %s answers %r after %.0f s", duthost.hostname, cmd, time.time() - started)
            return True, cmd
        if time.time() >= deadline:
            break
        time.sleep(interval)
    detail = "{!r} failed for {}s (rc={} {!r}); server cmdline: {!r}".format(
        cmd, timeout, res.get("rc"), ((res.get("stderr") or res.get("stdout") or "").strip())[:300],
        telemetry_cmdline(duthost))
    logger.error("gNMI server on %s not serving: %s", duthost.hostname, detail)
    return False, detail


# Not compiled into every image's gNMI server: gpins_daily_20261021_02 (sonic-gnmi d3d63ef, sonic-mgmt-common
# 478aedd) and gpins_daily_20261022_02 have none of it, zhouboyang_dhcp_test (sonic-gnmi 5594b57 "Add gNMI tests for
# DHCPv6 relay ...") has it. The telemetry test skips when the server does not advertise the model.
OC_RELAY_AGENT_MODEL = "openconfig-relay-agent"


def gnmi_supported_models(duthost, access="server-tls"):
    """Model names in the gNMI server's Capabilities (needs a reduced mode of :func:`gnmi_cli_session`).

    Returns an empty set if the call failed, so callers can tell "no answer" from "model missing".
    """
    res = ctr_exec(duthost, gnmi_serving_probe(access), container=TELEMETRY_CONTAINER, module_ignore_errors=True)
    if res.get("rc") != 0:
        return set()
    return set(re.findall(r'name:\s+"([^"]+)"', res.get("stdout") or ""))


def ensure_telemetry_running(duthost, timeout=120):
    """Make sure ``telemetry-con`` is RUNNING (``telemetry-container start`` if not) with the gNMI server process up.

    Touches nothing in CONFIG_DB; wrap local CLI use in :func:`gnmi_cli_session` to get past the server's mTLS.
    """
    if not is_container_running(duthost, TELEMETRY_CONTAINER) or not telemetry_cmdline(duthost):
        restart_telemetry(duthost, timeout=timeout)
    return True


def gnmi_server_in_mode(cmdline, access):
    """True if the running server (``cmdline``) already matches the reduced mode ``access``."""
    if not cmdline or telemetry_mtls_enabled(cmdline):
        return False
    if telemetry_plaintext(cmdline) != access.startswith("plaintext"):
        return False
    return not (access.endswith("+no-policies") and set(TELEMETRY_POLICY_FLAGS) & set(cmdline.split()))


@contextmanager
def gnmi_cli_session(duthost, access, serving_timeout=120):
    """UMF runbook recipe (go/gpins-umf-kb "Running gNMI CLI", option 1) around any local gNMI CLI use.

    Rewrites the CONFIG_DB rows telemetry.sh reads (Redis only, never ``config_db.json``), restarts
    telemetry-con so the server comes up without client-certificate authentication, waits until it serves,
    yields, then puts the rows back and restarts again. ``access``:

    * ``server-tls+no-policies`` (telemetry test): drop ``ca_crt``/``cert_crl_dir`` from ``TELEMETRY|certs``
      -> TLS server without client auth (client ``gnmi_cli -tls_skip_verify``), and drop the
      ``authz_policy``/``pathz_policy``/``fed_policy`` fields of ``TELEMETRY|gnmi``;
    * ``plaintext`` (DPB test): delete ``TELEMETRY|certs`` -> ``--insecure`` server (client ``gnmi_cli -insecure``).

    Fail-safe: the rows are read and logged before anything is deleted and every step after the delete
    runs inside ``try/finally``, so a failure anywhere (including the restart or the serving wait) still
    writes the rows back and restarts telemetry-con; nothing raised by the restore masks the caller's
    exception. The server's own save-on-set may rewrite ``config_db.json`` on an accepted Set, which is what
    the DPB test's persisted-config backup is for. Yields the pre-session state (:func:`gnmi_server_access_state`).
    """
    before = gnmi_server_access_state(duthost)
    logger.info("gNMI server access before the session on %s: %s", duthost.hostname, before)
    if access not in GNMI_ACCESS_MODES:
        raise ValueError("unknown gNMI access mode {!r} (expected one of {})".format(access, GNMI_ACCESS_MODES))
    certs = before[TELEMETRY_CERTS_ROW]
    if access.startswith("plaintext"):
        temporary = {TELEMETRY_CERTS_ROW: {}}
    else:
        if not ({"server_crt", "server_key"} <= set(certs)):
            raise RuntimeError("Cannot run the gNMI server on {} in TLS mode: CONFIG_DB {} has no "
                               "server_crt/server_key ({})".format(duthost.hostname, TELEMETRY_CERTS_ROW, certs))
        temporary = {TELEMETRY_CERTS_ROW: {k: v for k, v in certs.items() if k not in TELEMETRY_MTLS_FIELDS}}
    if access.endswith("+no-policies"):
        temporary[TELEMETRY_GNMI_ROW] = {k: v for k, v in before[TELEMETRY_GNMI_ROW].items()
                                         if k not in TELEMETRY_POLICY_FIELDS}
    to_restore = {key: before[key] for key in temporary if before[key] != temporary[key]}

    def _wait_serving():
        ok, detail = wait_gnmi_serving(duthost, access, serving_timeout)
        if not ok:
            raise RuntimeError("gNMI server on {} does not serve the local client in {} mode: {}".format(
                duthost.hostname, access, detail))

    if not to_restore and gnmi_server_in_mode(before["cmdline"], access):
        logger.info("gNMI server on %s already runs in %s mode; nothing to change", duthost.hostname, access)
        _wait_serving()
        yield before
        return
    logger.warning("Switching the gNMI server on %s to %s for the local CLI (runbook option 1): rows %s -> %s, "
                   "then restarting %s", duthost.hostname, access, to_restore,
                   {k: temporary[k] for k in to_restore}, TELEMETRY_CONTAINER)
    write_config_db_rows(duthost, {k: temporary[k] for k in to_restore})
    try:
        restart_telemetry(duthost, expect_mtls=False)
        _wait_serving()
        yield before
    finally:
        try:
            write_config_db_rows(duthost, to_restore)
            after = gnmi_server_access_state(duthost)
            bad = {k: (to_restore[k], after.get(k)) for k in to_restore if after.get(k) != to_restore[k]}
            if bad:
                logger.error("CONFIG_DB gNMI rows NOT restored on %s: %s", duthost.hostname, bad)
        except Exception as exc:  # never mask the caller's exception
            logger.error("Restoring the CONFIG_DB gNMI rows %s on %s failed: %s", to_restore, duthost.hostname, exc)
        try:
            # telemetry.sh renders --ca_crt iff TELEMETRY|certs has a ca_crt field.
            restart_telemetry(duthost, expect_mtls="ca_crt" in before[TELEMETRY_CERTS_ROW])
        except Exception as exc:  # never mask the caller's exception
            logger.error("%s restart after restoring the gNMI rows failed: %s", TELEMETRY_CONTAINER, exc)


def gnmi_get_json(duthost, xpath):
    """Run ``gnmi_get`` inside ``telemetry-con`` for ``xpath`` and return the parsed ``json_ietf_val`` dict.

    ``-insecure`` = TLS without server verification: needs the ``server-tls`` mode of :func:`gnmi_cli_session`.
    """
    import json
    cmd = "gnmi_get -target_addr 127.0.0.1:{} -alsologtostderr -insecure -xpath {}".format(
        TELEMETRY_PORT, shlex.quote(xpath))
    res = ctr_exec(duthost, cmd, container=TELEMETRY_CONTAINER, module_ignore_errors=False)
    stdout = res.get("stdout", "")
    matches = re.findall(r'json_ietf_val:\s*"((?:[^"\\]|\\.)*)"', stdout)
    if not matches:
        raise RuntimeError("No json_ietf_val found in gnmi_get output for {}:\n{}".format(xpath, stdout))
    raw_json = bytes(matches[-1], "utf-8").decode("unicode_escape")
    return json.loads(raw_json)


# Substrings that mark a JSON object in gnmi_cli output as a relay-agent update (vs. log noise).
GNMI_UPDATE_KEY_MARKERS = ("relay-agent", "interface", "counters", "state", "config", "options", "remote-id")


def gnmi_subscribe_json(duthost, path, query_type="s", streaming_type="SAMPLE",
                        sample_interval=1, streaming_duration="2s"):
    """Subscribe to ``path`` via ``gnmi_cli`` (``STREAM``: ``SAMPLE`` / ``ON_CHANGE``, or ``once``).

    Returns the parsed JSON updates.

    Returns a list of parsed JSON dicts streamed by the gNMI server.
    """
    import json
    args = [
        "gnmi_cli",
        "-client_types=gnmi",
        "-a", "127.0.0.1:{}".format(TELEMETRY_PORT),
        "-tls_skip_verify",
        "-logtostderr",
        "-t", "OC-YANG",
        "-qt={}".format(query_type),
    ]
    if query_type in ("s", "streaming"):
        args.append("-streaming_type={}".format(streaming_type))
        if streaming_type.upper() == "SAMPLE":
            args.append("-streaming_sample_interval={}".format(int(sample_interval)))
        if streaming_duration:
            args.append("-sd={}".format(streaming_duration))
    args.extend(["-q", path])
    cmd = " ".join(shlex.quote(a) for a in args)
    # Streaming queries with -sd exit when the streaming duration deadline expires.
    res = ctr_exec(duthost, cmd, container=TELEMETRY_CONTAINER, module_ignore_errors=True)
    stdout = res.get("stdout", "")

    decoder = json.JSONDecoder()
    updates = []
    idx = 0
    while idx < len(stdout):
        brace = stdout.find("{", idx)
        if brace == -1:
            break
        try:
            obj, end_idx = decoder.raw_decode(stdout, brace)
            if isinstance(obj, dict) and any(marker in k for k in obj for marker in GNMI_UPDATE_KEY_MARKERS):
                updates.append(obj)
            idx = end_idx
        except ValueError:
            idx = brace + 1

    if not updates:
        raise RuntimeError("No JSON telemetry updates received from gnmi_cli ({} / {}) on {}:\n"
                           "stdout: {}\nstderr: {}".format(
                               query_type, streaming_type, path, stdout, res.get("stderr", "")))
    return updates


def extract_openconfig_interface_node(payload, interface):
    """Extract the normalized ``interface[id=<interface>]`` dict from a ``gnmi_cli`` or ``gnmi_get`` payload."""
    norm = normalize_openconfig_dict(payload)
    if "relay-agent" in norm:
        ifaces = norm.get("relay-agent", {}).get("dhcpv6", {}).get("interfaces", {}).get("interface", {})
        if isinstance(ifaces, dict) and interface in ifaces:
            node = dict(ifaces[interface])
            node.setdefault("id", interface)
            return node
        if isinstance(ifaces, list):
            for item in ifaces:
                if isinstance(item, dict) and item.get("id") == interface:
                    return item
    if "interface" in norm:
        ifaces = norm["interface"]
        if isinstance(ifaces, list):
            for item in ifaces:
                if isinstance(item, dict) and item.get("id") == interface:
                    return item
        if isinstance(ifaces, dict) and interface in ifaces:
            node = dict(ifaces[interface])
            node.setdefault("id", interface)
            return node
    if "state" in norm or "counters" in norm:
        return norm
    raise AssertionError("Interface {!r} not found in OpenConfig telemetry payload: {}".format(interface, payload))


def validate_openconfig_dhcpv6_relay_schema(iface_node, expected_iface):
    """Strictly validate ``iface_node`` against the ``openconfig-relay-agent`` (+ ``-ext``) YANG."""
    norm = normalize_openconfig_dict(iface_node)
    unknown_top = set(norm) - OPENCONFIG_DHCPV6_INTERFACE_KEYS
    if unknown_top:
        raise AssertionError("Unknown top-level OpenConfig interface keys: {} in {}".format(sorted(unknown_top), norm))
    if "id" in norm and norm["id"] != expected_iface:
        raise AssertionError("Interface id mismatch: expected {!r}, got {!r}".format(expected_iface, norm["id"]))

    for section_name, allowed_keys in (("config", OPENCONFIG_DHCPV6_CONFIG_KEYS),
                                       ("state", OPENCONFIG_DHCPV6_STATE_KEYS)):
        section = norm.get(section_name)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise AssertionError("OpenConfig {} must be a dict, got {!r}".format(section_name, section))
        unknown_sec = set(section) - allowed_keys
        if unknown_sec:
            raise AssertionError("Unknown keys in OpenConfig {}: {}".format(section_name, sorted(unknown_sec)))
        if "id" in section and section["id"] != expected_iface:
            raise AssertionError("OpenConfig {}.id mismatch: expected {!r}, got {!r}".format(
                section_name, expected_iface, section["id"]))
        if "helper-address" in section:
            helpers = section["helper-address"]
            if not isinstance(helpers, list) or not all(isinstance(h, str) for h in helpers):
                raise AssertionError("OpenConfig {}.helper-address must be list[str], got {!r}".format(
                    section_name, helpers))
            for h in helpers:
                ipaddress.IPv6Address(h)

    if "options" in norm:
        opts = norm["options"]
        if not isinstance(opts, dict) or (set(opts) - {"config", "state"}):
            raise AssertionError("Invalid OpenConfig options container: {!r}".format(opts))
        for opt_sec_name in ("config", "state"):
            opt_sec = opts.get(opt_sec_name)
            if opt_sec is None:
                continue
            unknown_opt = set(opt_sec) - OPENCONFIG_DHCPV6_OPTION_KEYS
            if unknown_opt:
                raise AssertionError("Unknown keys in OpenConfig options.{}: {}".format(
                    opt_sec_name, sorted(unknown_opt)))
            if set(OPENCONFIG_CLIENT_MAC_LEAVES) <= set(opt_sec):
                raise AssertionError("OpenConfig options.{} carries both client-MAC leaves {}".format(
                    opt_sec_name, OPENCONFIG_CLIENT_MAC_LEAVES))
            for bool_leaf in OPENCONFIG_CLIENT_MAC_LEAVES + ("enable-interface-id", "enable-remote-id"):
                if bool_leaf in opt_sec and not isinstance(opt_sec[bool_leaf], bool):
                    raise AssertionError("OpenConfig options.{}.{} must be bool, got {!r}".format(
                        opt_sec_name, bool_leaf, opt_sec[bool_leaf]))
            for str_leaf in ("interface-id", "remote-id"):
                if str_leaf in opt_sec and not isinstance(opt_sec[str_leaf], str):
                    raise AssertionError("OpenConfig options.{}.{} must be str, got {!r}".format(
                        opt_sec_name, str_leaf, opt_sec[str_leaf]))

    counters = norm.get("state", {}).get("counters")
    if counters is not None:
        validate_openconfig_dhcpv6_counters_schema(counters)


def validate_openconfig_dhcpv6_counters_schema(counters):
    """Strictly validate that ``counters`` contains the exact 15 YANG counter leaves with uint64 values.

    Returns ``{leaf: int}`` keyed by the names in :data:`OPENCONFIG_DHCPV6_COUNTER_TO_STATE_DB` (GPINS augment
    names such as ``pins-dhcpv6-renew-received`` are mapped onto them).
    """
    raw = normalize_openconfig_dict(counters)
    both = sorted(alias for alias, name in OPENCONFIG_DHCPV6_COUNTER_ALIASES.items() if alias in raw and name in raw)
    if both:
        raise AssertionError("OpenConfig state/counters carries both the upstream and the GPINS name of {}".format(
            both))
    norm = {OPENCONFIG_DHCPV6_COUNTER_ALIASES.get(k, k): v for k, v in raw.items()}
    expected_keys = set(OPENCONFIG_DHCPV6_COUNTER_TO_STATE_DB)
    actual_keys = set(norm)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise AssertionError("OpenConfig state/counters schema mismatch: missing={}, extra={}".format(missing, extra))
    for leaf, val in norm.items():
        if isinstance(val, bool) or not (isinstance(val, int) or (isinstance(val, str) and val.isdigit())):
            raise AssertionError("OpenConfig counter {!r} value {!r} is not a valid uint64".format(leaf, val))
        if int(val) < 0:
            raise AssertionError("OpenConfig counter {!r} value {!r} is negative".format(leaf, val))
    return {k: int(v) for k, v in norm.items()}


# ---------------------------------------------------------------------------
# GPINS NSF (Non-Stop Forwarding / Warm Reboot)
# ---------------------------------------------------------------------------

GPINS_NSF_SCRIPT = "/etc/init.d/gpins-nsf-boot"
FRAMEWORK_CONTAINER = COMPONENT_CONTAINERS["framework"]
BOOT_INFO_KEY = "BOOT_INFO|system"


NSF_CORE_CONTAINERS = ("db-con", "or-con", "sy-con")
# WARM_RESTART_TABLE|<app> ``state`` values that mean the app finished warm-boot reconciliation.
NSF_RECONCILED_STATES = ("reconciled", "completed")


def get_kernel_uptime(duthost):
    """Return the DUT's kernel uptime in seconds from ``/proc/uptime``, or None if unreachable."""
    out = shell_stdout(duthost, "cut -d' ' -f1 /proc/uptime")
    if out is None:
        return None
    try:
        return float(out.strip())
    except ValueError:
        return None


def get_warm_restart_states(duthost):
    """Return ``{app: state}`` from STATE_DB ``WARM_RESTART_TABLE|*`` in one round-trip.

    Apps without a ``state`` field map to ''. Returns {} if the DUT or Redis is unreachable
    (never raises, so it is safe to use for diagnostics around a reboot).
    """
    db = db_id("STATE_DB")
    out = shell_stdout(
        duthost,
        "for k in $({0} -n {1} --raw KEYS 'WARM_RESTART_TABLE|*'); do "
        "echo \"${{k#WARM_RESTART_TABLE|}}=$({0} -n {1} --raw HGET \"$k\" state)\"; done".format(REDIS_CLI, db),
    )
    states = {}
    for line in (out or "").splitlines():
        if "=" in line:
            app, _, state = line.partition("=")
            states[app.strip()] = state.strip()
    return states


NSF_RECONCILE_APPS = ("orchagent", "syncd")


def get_warm_restart_restore_counts(duthost, apps=NSF_RECONCILE_APPS):
    """Return ``{app: restore_count}`` from STATE_DB ``WARM_RESTART_TABLE|<app>`` (-1 if absent).

    ``restore_count`` is incremented by each app every time it warm-starts, so it is the only
    field that distinguishes a freshly reconciled app from the stale ``state`` carried over
    from the previous warm boot through ``dump.rdb``.
    """
    counts = {}
    for app in apps:
        value = redis_hget(duthost, "STATE_DB", "WARM_RESTART_TABLE|{}".format(app), "restore_count")
        try:
            counts[app] = int(value)
        except (TypeError, ValueError):
            counts[app] = -1
    return counts


def _is_gpins_nsf_stack_ready(duthost, restore_counts_before=None):
    """Return True once the DUT is back after an NSF reboot: SSH + Redis answer, ``orchagent`` and
    ``syncd`` finished warm-boot reconciliation *in this boot* (their ``restore_count`` grew past
    ``restore_counts_before`` -- STATE_DB is restored from ``dump.rdb``, so ``state`` alone can be stale),
    the kernel port netdevs exist, and the core containers (``db-con``, ``or-con``, ``sy-con``) are RUNNING.
    """
    restore_counts_before = restore_counts_before or {}
    try:
        db = db_id("STATE_DB")
        out = shell_stdout(
            duthost,
            "{0} -n {1} --raw PING && {0} -n {1} --raw HMGET 'WARM_RESTART_TABLE|orchagent' state restore_count && "
            "{0} -n {1} --raw HMGET 'WARM_RESTART_TABLE|syncd' state restore_count && "
            "ip -o link show | grep -q Ethernet".format(REDIS_CLI, db),
        )
        if out is None:
            return False
        # Expected lines: PONG, <oa state>, <oa restore_count>, <syncd state>, <syncd restore_count>.
        # Ansible strips trailing whitespace, so missing trailing fields simply yield fewer lines.
        lines = [ln.strip() for ln in out.splitlines()]
        if len(lines) < 5 or lines[0] != "PONG":
            logger.debug("NSF readiness on %s: redis/warm-restart not ready yet (%s)", duthost.hostname, lines)
            return False
        observed = {"orchagent": (lines[1], lines[2]), "syncd": (lines[3], lines[4])}
        for app, (state, count) in observed.items():
            if state not in NSF_RECONCILED_STATES:
                logger.debug("NSF readiness on %s: %s state=%s", duthost.hostname, app, state)
                return False
            try:
                count = int(count)
            except ValueError:
                count = -1
            if count <= restore_counts_before.get(app, -1):
                logger.debug("NSF readiness on %s: %s restore_count=%s not yet past pre-reboot %s (stale state)",
                             duthost.hostname, app, count, restore_counts_before.get(app))
                return False
        tasks = get_container_tasks(duthost)
        not_running = [c for c in NSF_CORE_CONTAINERS if tasks.get(c) != "RUNNING"]
        if not_running:
            logger.debug("NSF readiness on %s: containers not RUNNING yet: %s", duthost.hostname, not_running)
            return False
        logger.info("NSF readiness on %s: %s, %s RUNNING", duthost.hostname,
                    ", ".join("{}={} (restore_count {})".format(a, s, c) for a, (s, c) in observed.items()),
                    list(NSF_CORE_CONTAINERS))
        return True
    except Exception as exc:
        logger.debug("NSF readiness probe on %s raised: %s", duthost.hostname, exc)
        return False


CONTAINER_MONITOR_POLL_SECONDS = 30   # /opt/gpins/container_monitor DEFAULT_CHECK_IN_PERIOD_SECONDS
LAB_KEYS_DIR = "/mnt/shared/container_files/keys"


def system_critical_state(duthost, container=None):
    """Return ``(is_critical, reason)`` from ``swss::SystemStateHelper``.

    This is the exact check ``rebootbackend`` performs before allowing NSF
    (``RebootThread: in critical state, NSF not allowed``): it aggregates
    ``STATE_DB`` ``COMPONENT_STATE_TABLE`` as written by ``container_monitor``.
    Run on the host by default, or inside ``container`` (e.g. ``framework-con``).
    """
    py = ("from swsscommon import swsscommon; h = swsscommon.StateHelperManager_SystemSingleton(); "
          "print('CRITICAL' if h.IsSystemCritical() else 'OK'); print(h.GetSystemCriticalReason())")
    cmd = "python3 -c {}".format(shlex.quote(py))
    if container:
        res = ctr_exec(duthost, cmd, container=container)
    else:
        res = duthost.shell(cmd, module_ignore_errors=True, verbose=False)
    lines = [ln.strip() for ln in (res.get("stdout") or "").splitlines() if ln.strip()]
    if not lines or lines[0] not in ("CRITICAL", "OK"):
        raise RuntimeError("Could not read SystemStateHelper on {}: rc={} out={!r} err={!r}".format(
            duthost.hostname, res.get("rc"), res.get("stdout"), res.get("stderr")))
    return lines[0] == "CRITICAL", " ".join(lines[1:])


def critical_components(reason):
    """Top-level component names from a critical reason.

    ``"Container monitor reports INACTIVE for components: p4rt, swss:orchagent, framework"``
    -> ``["framework", "p4rt", "swss"]``.
    """
    m = re.search(r"components?:\s*(.+)$", reason or "")
    if not m:
        return []
    return sorted({c.strip().split(":")[0] for c in m.group(1).rstrip(".").split(",") if c.strip()})


def _clear_latched_component_state(duthost, component):
    """Rewrite latched ``INACTIVE``/``ERROR`` entries of one component to ``UP``.

    Only call this after verifying the component's container is RUNNING: the entry is then
    provably stale (``container_monitor`` would not re-record INACTIVE for a running container,
    and never records UP for ``RECORD_INACTIVE``-only components). Covers the top-level key and
    its ``component:process`` critical-process keys. Returns the keys rewritten.
    """
    keys = redis_keys(duthost, "STATE_DB", "COMPONENT_STATE_TABLE|{}".format(component))
    keys += redis_keys(duthost, "STATE_DB", "COMPONENT_STATE_TABLE|{}:*".format(component))
    cleared = []
    for key in keys:
        fields = redis_hgetall(duthost, "STATE_DB", key)
        if fields.get("state") in ("INACTIVE", "ERROR"):
            redis_hset(duthost, "STATE_DB", key,
                       {"state": "UP", "reason": "stale {} cleared by sonic-mgmt after container start".format(
                           fields.get("state"))})
            cleared.append(key)
    return cleared


def ensure_lab_ca_cert(duthost):
    """Lab switches installed via non_modeled_switch_install have server_crt.pem but no ca_crt.pem.

    gpins-stack refuses to start framework/telemetry/p4rt without it. Reusing the server cert as
    the CA is the agreed lab workaround (logged loudly so it is never mistaken for a test step).
    """
    res = duthost.shell(
        "if [ ! -f {d}/ca_crt.pem ] && [ -f {d}/server_crt.pem ]; then "
        "cp -n {d}/server_crt.pem {d}/ca_crt.pem && echo SEEDED; fi".format(d=LAB_KEYS_DIR),
        module_ignore_errors=True, verbose=False)
    if "SEEDED" in (res.get("stdout") or ""):
        logger.warning("Seeded %s/ca_crt.pem from server_crt.pem on %s (lab-only workaround)",
                       LAB_KEYS_DIR, duthost.hostname)


def restart_framework(duthost, timeout=120):
    """(Re)start ``framework-con`` and wait until ``rebootbackend`` listens on ``Reboot_Request_Channel``."""
    action, _ = run_component_init(duthost, "framework", "restart")
    ok = wait_until(timeout, 3, 0,
                    lambda: is_container_running(duthost, FRAMEWORK_CONTAINER)
                    and bool(redis_pubsub_channels(duthost, "Reboot_Request_Channel")))
    if not ok:
        raise RuntimeError("{} (rebootbackend) is not ready on {} after {}".format(
            FRAMEWORK_CONTAINER, duthost.hostname, action))


def ensure_nsf_preconditions(duthost, timeout=300):
    """Bring the switch to a state in which ``rebootbackend`` accepts ``Reboot(NSF)``.

    ``rebootbackend`` refuses NSF while ``swss::SystemStateHelper`` reports the system critical, i.e.
    while ``STATE_DB`` ``COMPONENT_STATE_TABLE`` marks an essential component ``INACTIVE``/``ERROR``.
    On lab switches the usual blockers are feature containers ``gpins-stack`` did not start
    (``framework``, ``telemetry``, ``p4rt``). Two platform facts shape the fix:

    * ``container_monitor`` (30 s poll) records these components with ``on_poll: RECORD_INACTIVE``
      only - it writes ``INACTIVE`` when the container is down but never writes ``UP``.
    * ``component_state_helper.lua`` latches ``INACTIVE``: the component's own ``UP`` report is ignored.

    So starting the container alone leaves a stale ``INACTIVE`` behind. The procedure is therefore:
    start the blocker through its init script, **verify its containerd task is RUNNING**, and only
    then rewrite that component's latched entry to ``UP`` (a ``SET`` is also what live subscribers
    such as ``rebootbackend`` act on; a ``DEL`` is ignored by them). Nothing is cleared for a
    component whose container is not running - that stays a real blocker and is reported.
    ``framework-con`` is handled last so ``rebootbackend`` starts from a clean table.
    """
    ensure_lab_ca_cert(duthost)
    deadline = time.time() + timeout
    started = {}
    while True:
        critical, reason = system_critical_state(duthost)
        blockers = [c for c in critical_components(reason) if c != "framework"] if critical else []
        if not blockers:
            break
        for comp in blockers:
            if comp not in started:
                if comp not in COMPONENT_INIT_SCRIPTS:
                    logger.error("NSF blocked by %r on %s; no init script known for it", comp, duthost.hostname)
                    started[comp] = None
                    continue
                logger.info("NSF blocked on %s (%s); starting %s", duthost.hostname, reason, comp)
                started[comp] = run_component_init(duthost, comp, "start")[1]
            container = COMPONENT_CONTAINERS.get(comp)
            if container and is_container_running(duthost, container):
                cleared = _clear_latched_component_state(duthost, comp)
                if cleared:
                    logger.info("%s is RUNNING on %s; cleared latched entries %s", container, duthost.hostname, cleared)
        if time.time() > deadline:
            raise RuntimeError(
                "{} still critical after {}s: {!r}. Started: {}. Check `ctr -n default tasks ls` and "
                "/var/log/messages for the failing container (image missing from bundle? certs?)".format(
                    duthost.hostname, timeout, reason, started))
        time.sleep(CONTAINER_MONITOR_POLL_SECONDS // 3)

    if started:
        logger.info("Components brought UP on %s before NSF: %s", duthost.hostname, sorted(started))

    # framework-con last: start it if needed, then clear its own latched entry once it is RUNNING.
    if not is_container_running(duthost, FRAMEWORK_CONTAINER):
        restart_framework(duthost)
    else:
        if not redis_pubsub_channels(duthost, "Reboot_Request_Channel"):
            restart_framework(duthost)
    cleared = _clear_latched_component_state(duthost, "framework")
    if cleared:
        logger.info("%s is RUNNING on %s; cleared latched entries %s", FRAMEWORK_CONTAINER, duthost.hostname, cleared)

    if not wait_until(60, 5, 0, lambda: not system_critical_state(duthost)[0]):
        raise RuntimeError("{} still critical before NSF: {!r}".format(
            duthost.hostname, system_critical_state(duthost)[1]))
    # Best-effort cross-check with the Python binding inside framework-con. Some builds
    # (e.g. zhouboyang_dhcp_test) ship an older swsscommon there without
    # StateHelperManager_SystemSingleton; the host check above reads the same STATE_DB
    # and is authoritative, so a missing binding is logged, not fatal.
    try:
        critical, reason = system_critical_state(duthost, container=FRAMEWORK_CONTAINER)
    except RuntimeError as exc:
        logger.warning("Skipping in-container SystemStateHelper check on %s: %s", duthost.hostname, exc)
    else:
        if critical:
            raise RuntimeError("rebootbackend on {} would reject NSF: {!r}".format(duthost.hostname, reason))


# ---------------------------------------------------------------------------
# NSF diagnostics: orchagent retry loops + rebootbackend's verdict
# ---------------------------------------------------------------------------
#
# Found 2026-10-22 on ju3u1m1 (zhouboyang_dhcp_test 10-14; same symptom on gpins_daily_20261021_02):
# rebootbackend accepts Reboot(NSF) (SWSS_RC_SUCCESS), freezes the apps, then waits for each to report
# quiescent. orchagent stayed "frozen" but never quiescent because it kept retrying a failing request
# (removeVlan ... SAI_STATUS_OBJECT_IN_USE every ~1 s; while frozen it logs "start: Orchagent is not quiescent.
# Pending requests: VLAN_TABLE:Vlan1000|DEL"), so exactly 300 s later rebootbackend logged "timeout occurred
# during reboot stage freeze ... Outstanding apps: orchagent", did not reboot, and from then on answers every NSF
# request with "RebootThread: last NSF failed with non-retriable failure" until the next cold boot. All of these
# go to the host syslog (tags framework#rebootbackend and swss#orchagent).
# SONiC's RESTARTCHECK (orchagent_restart_check -n) is NOT a usable quiescence probe on GPINS: on a freshly
# booted, idle gpins_daily_20261021_02 with no orchagent errors it still answers NOT_READY within 1 ms. The
# pre-check therefore looks for what preceded the observed failure: the same orchagent error repeating.

SANDCASTLE_LOG = "/var/log/tmp/sandcastle.log"
REBOOTBACKEND_FREEZE_TIMEOUT = 300
ORCHAGENT_ERROR_WINDOW = 20


def _syslog_tag_pattern(tag, rest="", level=r"[^ ]+"):
    """ERE for sandcastle.log lines written BY ``tag`` (``<ts> <host> <level> <tag>: ...<rest>``).

    Anchoring on the tag field matters: Ansible logs every command it runs into the same file (tag
    ``ansible-ansible.legacy.command``), so an unanchored pattern matches the line recording our own grep.
    """
    return r"^[^ ]+ [^ ]+ {} {}: .*{}".format(level, tag, rest)   # tags are ERE-safe (letters, '#', '-')


_REBOOTBACKEND_FAILURE = _syslog_tag_pattern(
    "framework#rebootbackend", "(log_error_and_set_non_retry_failure|EXIT_EARLY|Outstanding apps|non-retriable)")
_ORCHAGENT_NOT_QUIESCENT = _syslog_tag_pattern("swss#orchagent", "Orchagent is not quiescent")
_ORCHAGENT_ERROR = _syslog_tag_pattern("swss#orchagent", level=r"user\.err")


def _syslog_lines(duthost, pattern, lines=5, rotated=False):
    """Last ``lines`` host-syslog lines matching the ERE ``pattern`` (``rotated`` also reads ``.1.gz`` first)."""
    source = "{{ gzip -dc {0}.1.gz 2>/dev/null; cat {0}; }}".format(SANDCASTLE_LOG) if rotated else \
        "cat {}".format(SANDCASTLE_LOG)
    out = shell_stdout(duthost, "{} | grep -a -E {} | tail -n {}".format(source, shlex.quote(pattern), int(lines)))
    return [line.strip() for line in (out or "").splitlines() if line.strip()]


def _syslog_message(line):
    """The message part of a sandcastle.log line (after the ``... :- `` prefix)."""
    return line.split(":- ", 1)[-1].replace("#012", " ").strip()


def orchagent_pending_requests(duthost):
    """orchagent's own most recent "not quiescent" report and its last errors (diagnostics only)."""
    pending = _syslog_lines(duthost, _ORCHAGENT_NOT_QUIESCENT, 1, rotated=True)
    errors = _syslog_lines(duthost, _ORCHAGENT_ERROR, 2, rotated=True)
    return "orchagent reports: {}; last orchagent errors: {}".format(
        [_syslog_message(p) for p in pending] or "nothing", [_syslog_message(e) for e in errors] or "none")


def orchagent_repeating_errors(duthost, window=ORCHAGENT_ERROR_WINDOW):
    """Error messages orchagent logged at least twice during the next ``window`` seconds (``[]`` = none).

    A message that repeats is a request orchagent keeps retrying; such a request is never drained, so
    rebootbackend's freeze cannot complete. A clean boot logs no orchagent errors at all.
    """
    start = shell_stdout(duthost, "wc -l < {}".format(SANDCASTLE_LOG))
    time.sleep(window)
    try:
        first = int((start or "").strip()) + 1
    except ValueError:
        return []
    out = shell_stdout(duthost, "tail -n +{} {} | grep -a -E {} | tail -n 200".format(
        first, SANDCASTLE_LOG, shlex.quote(_ORCHAGENT_ERROR)))
    counts = {}
    for line in (out or "").splitlines():
        if line.strip():
            message = _syslog_message(line)
            counts[message] = counts.get(message, 0) + 1
    return sorted(message for message, count in counts.items() if count >= 2)


def _switch_epoch(duthost):
    """The switch's own clock (``date +%s``); its syslog timestamps use the same clock."""
    out = shell_stdout(duthost, "date +%s")
    try:
        return int((out or "").strip())
    except ValueError:
        return None


def rebootbackend_verdict(duthost, since_epoch):
    """rebootbackend's failure messages logged at/after ``since_epoch`` (switch clock), or ``[]``.

    Matches the lines it writes when it gives up on an accepted NSF (e.g. ``timeout occurred during reboot
    stage freeze`` and ``perform_freeze_quiescence: returned EXIT_EARLY. Outstanding apps: orchagent``).
    """
    import datetime
    messages = []
    for line in _syslog_lines(duthost, _REBOOTBACKEND_FAILURE, 10):
        try:
            stamp = datetime.datetime.fromisoformat(line.split()[0]).timestamp()
        except (ValueError, IndexError):
            continue
        if since_epoch is None or stamp >= since_epoch - 2:
            messages.append(_syslog_message(line))
    return messages


# orchagent records every task it consumes in swss.rec ("<ts>|ROUTE_TABLE:<prefix>|SET|...|ifname:<if>,...").
# On gpins_daily_20261021_02 bgp-con (zebra + fpmsyncd) exports the management bond's routes (::/0 and the
# 2607:f8b0:8093:af03::/64 subnet "via bond0") ~70 s after boot. bond0 is not an ASIC interface, so orchagent can
# never program them and keeps them pending forever ("Orchagent is not quiescent. Pending requests:
# ROUTE_TABLE:2607:f8b0:8093:af03::/64|SET|...|ifname:bond0"); rebootbackend's NSF freeze then times out.
# zhouboyang_dhcp_test runs no bgp-con, which is why NSF passes there. Routes only reach orchagent over ZMQ
# (orchagent -q ipc:///zmq/...), so they are not visible in APPL_DB and cannot be withdrawn from the test.
SWSS_REC_DIR = "/var/log/tmp/or-con/swss"
UNPROGRAMMABLE_ROUTE_IFNAME_RE = r"^bond[0-9]+$"


def orchagent_unprogrammable_routes(duthost):
    """Routes orchagent holds that it can never program: last op SET and every next-hop on a management bond.

    Returns ``["ROUTE_TABLE:<prefix> via <ifnames>", ...]`` (``[]`` = none or no swss.rec on this image).
    """
    script = (
        "cd {d} 2>/dev/null || exit 0; ls -r swss.rec* 2>/dev/null | while read f; do "
        "case $f in *.gz) gzip -dc $f;; *) cat $f;; esac; done | grep -a '|ROUTE_TABLE:' | "
        "awk -F'|' '{{k=$2; last[k]=$3; line[k]=$0}} END {{for (k in last) if (last[k]==\"SET\") {{"
        "n=split(line[k], f, \"|\"); ifn=\"\"; for (i=4; i<=n; i++) if (f[i] ~ /^ifname:/) ifn=substr(f[i], 8); "
        "if (ifn==\"\") continue; m=split(ifn, a, \",\"); bad=1; "
        "for (j=1; j<=m; j++) if (a[j] !~ /{re}/) bad=0; if (bad) print k \" via \" ifn}}}}'"
    ).format(d=SWSS_REC_DIR, re=UNPROGRAMMABLE_ROUTE_IFNAME_RE)
    out = shell_stdout(duthost, script)
    return sorted(line.strip() for line in (out or "").splitlines() if line.strip())


def _send_nsf_reboot_request(duthost):
    """Send a gNOI Reboot(NSF) notification to ``rebootbackend`` and return its response string."""
    exec_id = "nsf_req_{}_{}".format(os.getpid(), int(time.time() * 1000) % 1000000)
    py_cmd = (
        "from swsscommon import swsscommon; "
        "db=swsscommon.DBConnector('STATE_DB', 0); "
        "nc=swsscommon.NotificationConsumer(db, 'Reboot_Response_Channel'); "
        "np=swsscommon.NotificationProducer(db, 'Reboot_Request_Channel'); "
        "np.send('Reboot', '', swsscommon.FieldValuePairs("
        "[('MESSAGE', '{\"method\":\"NSF\",\"message\":\"Performing NSF Reboot\"}')])); "
        "s=swsscommon.Select(); s.addSelectable(nc); "
        "ret, _ = s.select(10000); "
        "print(nc.pop() if ret == swsscommon.Select.OBJECT else 'TIMEOUT')"
    )
    req_res = duthost.shell(
        "ctr -n default t exec --exec-id {} {} python3 -c {}".format(
            shlex.quote(exec_id), FRAMEWORK_CONTAINER, shlex.quote(py_cmd)),
        module_ignore_errors=True,
    )
    return (req_res.get("stdout") or "").strip()


def trigger_gpins_nsf_reboot(duthost, reboot_timeout=600, reboot_start_timeout=REBOOTBACKEND_FREEZE_TIMEOUT + 120):
    """Trigger a full GPINS NSF warm reboot via ``rebootbackend`` (``framework-con``) and wait for recovery.

    1. Refuses to go on while orchagent holds routes it can never program (:func:`orchagent_unprogrammable_routes`:
       every next hop on a management bond, exported by bgp-con): rebootbackend's freeze would wait for them to
       drain, time out after 300 s, not reboot, and refuse every further NSF until a cold boot.
    2. Brings the switch out of critical state (``ensure_nsf_preconditions``) and makes sure
       ``framework-con`` (``rebootbackend``) is up and subscribed to ``Reboot_Request_Channel``.
    3. Refuses to go on while orchagent keeps retrying a failing request (:func:`orchagent_repeating_errors`),
       which ends the same way ("Outstanding apps: orchagent").
    4. Records pre-reboot kernel uptime (``/proc/uptime``) and the switch clock.
    5. Sends a gNOI ``Reboot`` notification (``{"method": "NSF"}``) on ``STATE_DB`` ``Reboot_Request_Channel``
       so ``rebootbackend`` freezes and checkpoints ``orchagent`` and ``syncd`` (writing ``sai-warmboot.bin``)
       and invokes ``/etc/init.d/gpins-nsf-boot nsf-reboot``.
    6. Waits up to ``reboot_start_timeout`` s for the kernel reboot to begin (kernel uptime drops below the
       pre-reboot value, or SSH is unreachable on two consecutive probes), failing as soon as rebootbackend logs
       that it gave up, and then up to ``reboot_timeout`` s for ``db-con``, ``or-con``, and ``sy-con`` to be
       ``RUNNING`` with ``orchagent``/``syncd`` reconciled.
    7. Asserts the kernel really rebooted (``uptime_after < uptime_before``) and that it was a warm boot
       (``BOOT_INFO|system boot-type``), then returns ``STATE_DB`` ``BOOT_INFO|system``.
    """
    stuck_routes = orchagent_unprogrammable_routes(duthost)
    if stuck_routes:
        raise RuntimeError(
            "orchagent on {} holds route(s) it can never program: {} (next hop on the management bond, not an ASIC "
            "interface; pushed by bgp-con's zebra/fpmsyncd). rebootbackend's NSF freeze waits for orchagent to drain "
            "them, times out after {} s ('Outstanding apps: orchagent'), does not reboot and refuses NSF until the "
            "next cold boot. NSF request NOT sent. Image issue: on images without bgp-con these routes never reach "
            "orchagent and NSF passes.".format(duthost.hostname, stuck_routes, REBOOTBACKEND_FREEZE_TIMEOUT))
    ensure_nsf_preconditions(duthost)
    repeating = orchagent_repeating_errors(duthost)
    if repeating:
        raise RuntimeError(
            "orchagent on {} keeps retrying a failing request (same error repeated within {} s: {}), so "
            "rebootbackend's NSF freeze would time out after {} s ('Outstanding apps: orchagent'), not reboot, and "
            "refuse NSF until the next cold boot; NSF request NOT sent. {}".format(
                duthost.hostname, ORCHAGENT_ERROR_WINDOW, repeating, REBOOTBACKEND_FREEZE_TIMEOUT,
                orchagent_pending_requests(duthost)))
    uptime_before = None
    for _ in range(3):
        uptime_before = get_kernel_uptime(duthost)
        if uptime_before is not None:
            break
        time.sleep(2)
    if uptime_before is None:
        raise RuntimeError("Cannot read /proc/uptime on {} before the NSF reboot".format(duthost.hostname))
    restore_counts_before = get_warm_restart_restore_counts(duthost)
    boot_info_before = redis_hgetall(duthost, "STATE_DB", BOOT_INFO_KEY)
    logger.info("Triggering GPINS NSF reboot on %s via rebootbackend (pre-reboot uptime=%.0f s, BOOT_INFO=%s, "
                "restore_counts=%s, warm-restart states=%s)", duthost.hostname, uptime_before, boot_info_before,
                restore_counts_before, get_warm_restart_states(duthost))

    request_epoch = _switch_epoch(duthost)
    req_out = _send_nsf_reboot_request(duthost)
    logger.info("rebootbackend Reboot(NSF) response on %s: %s", duthost.hostname, req_out)
    if "SWSS_RC_SUCCESS" not in req_out and "critical state" in req_out.lower():
        # rebootbackend's SystemStateHelper keeps its own copy of the table; if it predates the
        # bring-up above it can still hold INACTIVE. A restart re-reads the (now clean) table.
        logger.warning("rebootbackend on %s still reports critical state (%s); restarting %s and retrying once",
                       duthost.hostname, system_critical_state(duthost)[1], FRAMEWORK_CONTAINER)
        restart_framework(duthost)
        request_epoch = _switch_epoch(duthost)
        req_out = _send_nsf_reboot_request(duthost)
        logger.info("rebootbackend Reboot(NSF) retry response on %s: %s", duthost.hostname, req_out)
    if "SWSS_RC_SUCCESS" not in req_out and "non-retriable" in req_out.lower():
        raise RuntimeError(
            "rebootbackend on {} refuses NSF until the next cold boot: an earlier NSF attempt on this boot failed "
            "and latched it ({}). Cold-reboot the switch; the first failed NSF test has the reason.".format(
                duthost.hostname, req_out))
    if "SWSS_RC_SUCCESS" not in req_out:
        raise RuntimeError("rebootbackend rejected NSF reboot on {}: {}".format(duthost.hostname, req_out))

    # Wait for the kernel reboot to begin. A single failed probe is not enough (transient SSH/containerd
    # hiccups happen); require an uptime drop or two consecutive unreachable probes. rebootbackend gives up on
    # the freeze after REBOOTBACKEND_FREEZE_TIMEOUT s; its verdict is checked every 15 s so a failed attempt
    # is reported with its reason instead of as a timeout.
    request_time = time.time()
    down_deadline = request_time + reboot_start_timeout
    next_verdict_check = request_time + 15
    reboot_observed = False
    unreachable_probes = 0
    while time.time() < down_deadline:
        time.sleep(3)
        up = get_kernel_uptime(duthost)
        if up is None:
            unreachable_probes += 1
        else:
            unreachable_probes = 0
        if (up is not None and up < uptime_before) or unreachable_probes >= 2:
            reboot_observed = True
            logger.info("GPINS NSF reboot underway on %s after %.0f s (current uptime=%s)",
                        duthost.hostname, time.time() - request_time, up)
            break
        if up is not None and time.time() >= next_verdict_check:
            next_verdict_check = time.time() + 15
            verdict = rebootbackend_verdict(duthost, request_epoch)
            if verdict:
                raise RuntimeError(
                    "rebootbackend on {} accepted the NSF request but gave up {:.0f} s later without rebooting: {}. "
                    "It refuses NSF until the next cold boot and may leave apps frozen. {}".format(
                        duthost.hostname, time.time() - request_time, verdict, orchagent_pending_requests(duthost)))

    if not reboot_observed:
        raise RuntimeError("GPINS NSF reboot did not start within {}s on {} (uptime_before={:.0f}s); rebootbackend "
                           "verdict: {}; warm-restart states: {}".format(
                               reboot_start_timeout, duthost.hostname, uptime_before,
                               rebootbackend_verdict(duthost, request_epoch) or "none logged",
                               get_warm_restart_states(duthost)))

    # Wait for the switch to come back up and warm-boot reconciliation (orchagent & syncd) to complete.
    if not wait_until(reboot_timeout, 5, 10, _is_gpins_nsf_stack_ready, duthost, restore_counts_before):
        try:
            tasks = get_container_tasks(duthost)
        except Exception as exc:
            tasks = "unavailable ({})".format(exc)
        raise RuntimeError("DUT {} did not recover from GPINS NSF reboot within {}s (uptime now={}, pre-reboot "
                           "restore_counts={}, warm-restart states={}, container tasks={})".format(
                               duthost.hostname, reboot_timeout, get_kernel_uptime(duthost), restore_counts_before,
                               get_warm_restart_states(duthost), tasks))

    uptime_after = get_kernel_uptime(duthost)
    boot_info = redis_hgetall(duthost, "STATE_DB", BOOT_INFO_KEY)
    logger.info("DUT %s recovered from GPINS NSF reboot %.0f s after the request (uptime_after=%s s, "
                "BOOT_INFO %s -> %s, restore_counts %s -> %s, warm-restart states=%s)", duthost.hostname,
                time.time() - request_time, uptime_after, boot_info_before, boot_info, restore_counts_before,
                get_warm_restart_restore_counts(duthost), get_warm_restart_states(duthost))
    if uptime_after is None or uptime_after >= uptime_before:
        raise RuntimeError("rebootbackend accepted the NSF request on {} but the kernel did not reboot "
                           "(uptime before={:.0f}s, after={}s)".format(duthost.hostname, uptime_before, uptime_after))
    boot_type = boot_info.get("boot-type")
    if boot_type is None:
        logger.warning("STATE_DB %s has no boot-type on %s; cannot confirm warm boot: %s",
                       BOOT_INFO_KEY, duthost.hostname, boot_info)
    elif boot_type != "warm_boot":
        raise RuntimeError("DUT {} came back from a '{}' boot instead of a warm (NSF) boot: {}".format(
            duthost.hostname, boot_type, boot_info))
    return boot_info

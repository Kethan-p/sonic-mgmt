import os
import ast
import socket
import subprocess
import time
import scapy
# Packet Test Framework imports
import ptf
import ptf.packet as packet
import ptf.testutils as testutils
from ptf import config
from ptf.base_tests import BaseTest
from ptf.mask import Mask
from scapy.fields import MACField, ShortEnumField, FieldLenField, ShortField
from scapy.data import ETHER_ANY
from scapy.packet import NoPayload
from scapy.layers.dhcp6 import _DHCP6OptGuessPayload

IPv6 = scapy.layers.inet6.IPv6
DHCP6_Solicit = scapy.layers.dhcp6.DHCP6_Solicit
DHCP6_RelayForward = scapy.layers.dhcp6.DHCP6_RelayForward
DHCP6_Request = scapy.layers.dhcp6.DHCP6_Request
DHCP6_Advertise = scapy.layers.dhcp6.DHCP6_Advertise
DHCP6_Reply = scapy.layers.dhcp6.DHCP6_Reply
DHCP6_RelayReply = scapy.layers.dhcp6.DHCP6_RelayReply
DHCP6OptRelayMsg = scapy.layers.dhcp6.DHCP6OptRelayMsg
DHCP6OptUnknown = scapy.layers.dhcp6.DHCP6OptUnknown

DHCP6OptClientId = scapy.layers.dhcp6.DHCP6OptClientId
DHCP6OptOptReq = scapy.layers.dhcp6.DHCP6OptOptReq
DHCP6OptElapsedTime = scapy.layers.dhcp6.DHCP6OptElapsedTime
DHCP6OptIA_NA = scapy.layers.dhcp6.DHCP6OptIA_NA
DUID_LL = scapy.layers.dhcp6.DUID_LL
DHCP6OptIfaceId = scapy.layers.dhcp6.DHCP6OptIfaceId
DHCP6OptRemoteID = scapy.layers.dhcp6.DHCP6OptRemoteID
DHCP6OptServerId = scapy.layers.dhcp6.DHCP6OptServerId


class DataplaneBaseTest(BaseTest):
    def __init__(self):
        BaseTest.__init__(self)

    def setUp(self):
        self.dataplane = ptf.dataplane_instance
        self.dataplane.flush()
        if config["log_dir"] is not None:
            filename = os.path.join(config["log_dir"], str(self)) + ".pcap"
            self.dataplane.start_pcap(filename)

    def tearDown(self):
        if config["log_dir"] is not None:
            self.dataplane.stop_pcap()


"""
 This test simulates a new host booting up on the VLAN network of a ToR and
 requesting an IPv6 address via DHCPv6. Setup is as follows:
  - DHCP client is simulated by listening/sending on an interface connected to VLAN of ToR.
  - DHCP server is simulated by listening/sending on injected PTF interfaces which link
    ToR to leaves. This way we can listen for traffic sent from DHCP relay out to would-be DHCPv6 servers

 This test performs the following functionality:
   1.) Simulated client broadcasts a DHCPv6 SOLICIT message.
   2.) Verify DHCP relay running on ToR receives the DHCPv6 SOLICIT message and send a DHCPv6 RELAY-FORWARD
       message encapsulating the client DHCPv6 SOLICIT message and relays it to all of its known DHCP servers.
   3.) Simulate DHCPv6 RELAY-REPLY message send from a DHCP server to the ToR encapsulating DHCPv6 ADVERTISE message.
   4.) Verify DHCP relay receives the DHCPv6 RELAY-REPLY message decapsulate it and forwards DHCPv6 ADVERTISE
       message to our simulated client.
   5.) Simulated client broadcasts a DHCPv6 REQUEST message.
   6.) Verify DHCP relay running on ToR receives the DHCPv6 REQUEST message and send a DHCPv6 RELAY-FORWARD
       message encapsulating the client DHCPv6 REQUEST message and relays it to all of its known DHCP servers.
   7.) Simulate DHCPv6 RELAY-REPLY message send from a DHCP server to the ToR encapsulating DHCPv6 REPLY message.
   8.) Verify DHCP relay receives the DHCPv6 RELAY-REPLY message decapsulate it and forwards DHCPv6 REPLY
       message to our simulated client.

"""

dhcp6opts = {79: "OPTION_CLIENT_LINKLAYER_ADDR",  # RFC6939
             }


class _LLAddrField(MACField):
    pass

# "Client link-layer address type.  The link-layer type MUST be a valid hardware  # noqa: E501
# type assigned by the IANA, as described in [RFC0826]


class DHCP6OptClientLinkLayerAddr(_DHCP6OptGuessPayload):  # RFC6939
    name = "DHCP6 Option - Client Link Layer address"
    fields_desc = [ShortEnumField("optcode", 79, dhcp6opts),
                   FieldLenField("optlen", None, length_of="clladdr",
                                 adjust=lambda pkt, x: x + 2),
                   ShortField("lltype", 1),  # ethernet
                   _LLAddrField("clladdr", ETHER_ANY)]


class DHCPTest(DataplaneBaseTest):
    BROADCAST_MAC = '33:33:00:01:00:02'
    BROADCAST_IP = 'ff02::1:2'
    DHCP_CLIENT_PORT = 546
    DHCP_SERVER_PORT = 547

    def __init__(self):
        self.test_params = testutils.test_params_get()
        self.client_port_index = int(self.test_params['client_port_index'])
        self.is_dualtor = True if self.test_params['is_dualtor'] == 'True' else False
        self.client_link_local = self.generate_client_interace_ipv6_link_local_address(
            self.client_port_index)

        DataplaneBaseTest.__init__(self)

    def setUp(self):
        DataplaneBaseTest.setUp(self)
        self.hostname = self.test_params['hostname']

        # These are the interfaces we are injected into that link to out leaf switches
        self.server_port_indices = ast.literal_eval(
            self.test_params['leaf_port_indices'])
        self.num_dhcp_servers = int(self.test_params['num_dhcp_servers'])
        self.assertTrue(self.num_dhcp_servers > 0,
                        "Error: This test requires at least one DHCP server to be specified!")

        # We will simulate a responding DHCP server on the first interface in the provided set
        self.server_ip = self.test_params['server_ip']

        self.relay_iface_ip = self.test_params['relay_iface_ip']
        self.relay_iface_mac = self.test_params['relay_iface_mac']
        self.relay_link_local = self.test_params['relay_link_local']
        self.relay_linkaddr = '::'
        self.vlan_ip = self.test_params['vlan_ip']
        self.downstream_relay_ip = self.test_params['downstream_relay_ip']
        self.client_mac = self.dataplane.get_mac(0, self.client_port_index)
        self.uplink_mac = self.test_params['uplink_mac']
        self.loopback_ipv6 = self.test_params['loopback_ipv6']

        # --- GPINS relay-mode parameters (all optional; defaults == upstream behaviour) ---
        # link_addr: expected link-address in Relay-Forward. Upstream: the VLAN GUA.
        #   GPINS L3+LLA has no GUA on the relay interface -> '::'.
        self.link_addr = self._param_or('link_addr', self.vlan_ip)
        # relay_listen_ip: destination the server sends Relay-Reply to. Upstream:
        #   loopback on dual-ToR, VLAN GUA otherwise. GPINS L3+LLA: loopback.
        self.relay_listen_ip = self._param_or(
            'relay_listen_ip', self.loopback_ipv6 if self.is_dualtor else self.relay_iface_ip)
        # interface_id: expected Option 18 value as a string, e.g. "[Loopback0]:Ethernet1/33".
        #   None -> upstream behaviour (present only on dual-ToR, as the 16-byte VLAN GUA).
        self.interface_id = self._param_or('interface_id', None)
        # interface_id_gua: GUA the relay uses as its default Option 18 (GPINS L3+GUA with
        #   enable_interface_id=true and no interface_id), sent as the 16-byte address (ju3u1m1, 2026-10-26).
        self.interface_id_gua = self._param_or('interface_id_gua', None)
        # remote_id: expected Option 37 remote-id string. None -> not expected.
        self.remote_id = self._param_or('remote_id', None)
        # rfc6939_support: whether Option 79 (Client Link-Layer Address) should be
        #   checked in Relay-Forward (default True; False -> skip Option 79 check).
        rfc6939_param = self._param_or('rfc6939_support', 'True')
        self.rfc6939_support = str(rfc6939_param).lower() not in ('false', '0', 'no')
        # downstream_reply_src_ip: IPv6 source of the nested Relay-Reply the DUT
        #   forwards to downstream Relay A. Upstream: VLAN GUA. '' -> don't care
        #   (GPINS L3+LLA, where the relay has no GUA on the port).
        self.downstream_reply_src_ip = self._param_or('downstream_reply_src_ip', self.relay_iface_ip)

    def _param_or(self, name, default):
        value = self.test_params.get(name)
        if value is None or value == 'None':
            return default
        return value

    @staticmethod
    def _ipv6_equal(a, b):
        try:
            return socket.inet_pton(socket.AF_INET6, str(a)) == socket.inet_pton(socket.AF_INET6, str(b))
        except (OSError, ValueError, TypeError):
            return str(a) == str(b)

    def expected_iface_id_bytes(self):
        """Option 18 payload the DUT is expected to insert, or None if none is expected."""
        if self.interface_id is not None:
            return str(self.interface_id).encode()
        if self.interface_id_gua is not None:
            return socket.inet_pton(socket.AF_INET6, self.interface_id_gua)
        if self.is_dualtor:
            return socket.inet_pton(socket.AF_INET6, self.vlan_ip)
        return None

    def reply_iface_id_bytes(self):
        """Option 18 the simulated server copies into its Relay-Reply (RFC 8415 19.2), or None."""
        if self.interface_id is None and self.interface_id_gua is None:
            return None
        return self.expected_iface_id_bytes()

    def generate_client_interace_ipv6_link_local_address(self, client_port_index):
        # For DUALTOR Setup, flapping the link will disrupt ICMP HB communication and link health is impacted.
        # Skip this for DUALTOR.
        if not self.is_dualtor:
            # Shutdown and startup the client interface to generate a proper IPv6 link-local address
            command = "ifconfig eth{} down".format(client_port_index)
            proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            proc.communicate()
            command = "ifconfig eth{} up".format(client_port_index)
            proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            proc.communicate()

        command = "ip addr show eth{} | grep inet6 | grep 'scope link' | awk '{{print $2}}' | cut -d '/' -f1".\
                  format(client_port_index)
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        stdout, stderr = proc.communicate()

        return stdout.decode().strip() if isinstance(stdout, bytes) else stdout.strip()

    def tearDown(self):
        DataplaneBaseTest.tearDown(self)

    """
     Packet generation functions/wrappers
    """

    def create_dhcp_solicit_packet(self):
        solicit_packet = packet.Ether(
            src=self.client_mac, dst=self.BROADCAST_MAC)
        solicit_packet /= IPv6(src=self.client_link_local,
                               dst=self.BROADCAST_IP)
        solicit_packet /= packet.UDP(sport=self.DHCP_CLIENT_PORT,
                                     dport=self.DHCP_SERVER_PORT)
        solicit_packet /= DHCP6_Solicit(trid=12345)
        solicit_packet /= DHCP6OptClientId(
            duid=DUID_LL(lladdr=self.client_mac))
        solicit_packet /= DHCP6OptIA_NA()
        solicit_packet /= DHCP6OptOptReq(reqopts=[23, 24, 29])
        solicit_packet /= DHCP6OptElapsedTime(elapsedtime=0)

        return solicit_packet

    def _server_relay_reply(self, inner_message):
        """Relay-Reply as a DHCPv6 server would send it to this relay."""
        pkt = packet.Ether(dst=self.uplink_mac)
        pkt /= IPv6(src=self.server_ip, dst=self.relay_listen_ip)
        pkt /= packet.UDP(sport=self.DHCP_SERVER_PORT, dport=self.DHCP_SERVER_PORT)
        pkt /= DHCP6_RelayReply(msgtype=13, linkaddr=self.link_addr, peeraddr=self.client_link_local)
        # RFC 8415 19.2: the server copies Interface-ID back; GPINS L3+LLA relies
        # on it to find the egress port because link-address is '::'.
        iface_id = self.reply_iface_id_bytes()
        if iface_id is not None:
            pkt /= DHCP6OptIfaceId(ifaceid=iface_id)
        pkt /= DHCP6OptRelayMsg(message=[inner_message])
        return pkt

    def create_dhcp_advertise_packet(self):
        advertise_packet = packet.Ether(
            src=self.relay_iface_mac, dst=self.client_mac)
        advertise_packet /= IPv6(src=self.relay_link_local,
                                 dst=self.client_link_local)
        advertise_packet /= packet.UDP(sport=self.DHCP_SERVER_PORT,
                                       dport=self.DHCP_CLIENT_PORT)
        advertise_packet /= DHCP6_Advertise(trid=12345)

        return advertise_packet

    def create_dhcp_advertise_relay_reply_packet(self):
        return self._server_relay_reply(DHCP6_Advertise(trid=12345))

    def create_dhcp_request_packet(self):
        request_packet = packet.Ether(
            src=self.client_mac, dst=self.BROADCAST_MAC)
        request_packet /= IPv6(src=self.client_link_local,
                               dst=self.BROADCAST_IP)
        request_packet /= packet.UDP(sport=self.DHCP_CLIENT_PORT,
                                     dport=self.DHCP_SERVER_PORT)
        request_packet /= DHCP6_Request(trid=12345)

        return request_packet

    def create_dhcp_reply_packet(self):
        reply_packet = packet.Ether(
            src=self.relay_iface_mac, dst=self.client_mac)
        reply_packet /= IPv6(src=self.relay_link_local,
                             dst=self.client_link_local)
        reply_packet /= packet.UDP(sport=self.DHCP_SERVER_PORT,
                                   dport=self.DHCP_CLIENT_PORT)
        reply_packet /= DHCP6_Reply(trid=12345)

        return reply_packet

    def create_dhcp_reply_relay_reply_packet(self):
        return self._server_relay_reply(DHCP6_Reply(trid=12345))

    def create_downstream_relay_forward(self):
        """Create the opaque Relay-Forward received from downstream Relay A."""
        # Relay B must preserve this entire envelope. Its inner link, peer,
        # and client message describe topology behind Relay A and are not
        # behavior under test here.
        relay_forward = DHCP6_RelayForward(
            msgtype=12,
            hopcount=0,
            linkaddr=self.vlan_ip,
            peeraddr=self.client_link_local)
        relay_forward /= DHCP6OptRelayMsg(
            message=[DHCP6_Solicit(trid=12345)])

        return relay_forward

    def create_dhcp_relay_forward_packet(self):
        """Create the packet sent from immediate downstream Relay A."""
        relay_forward_packet = packet.Ether(
            src=self.client_mac, dst=self.relay_iface_mac)
        relay_forward_packet /= IPv6(
            src=self.downstream_relay_ip, dst=self.relay_iface_ip)
        relay_forward_packet /= packet.UDP(
            sport=self.DHCP_SERVER_PORT, dport=self.DHCP_SERVER_PORT)
        relay_forward_packet /= self.create_downstream_relay_forward()

        return relay_forward_packet

    def create_dhcp_relayed_relay_packet(self):
        """Create the nested Relay-Forward Relay B should send upstream."""
        relayed_relay_packet = packet.Ether(src=self.uplink_mac)
        relayed_relay_packet /= IPv6()
        relayed_relay_packet /= packet.UDP(
            sport=self.DHCP_SERVER_PORT, dport=self.DHCP_SERVER_PORT)
        relayed_relay_packet /= DHCP6_RelayForward(
            msgtype=12,
            hopcount=1,
            linkaddr=self.relay_linkaddr,
            peeraddr=self.downstream_relay_ip)
        relayed_relay_packet /= DHCP6OptRelayMsg(
            message=[self.create_downstream_relay_forward()])
        if self.is_dualtor:
            relayed_relay_packet /= DHCP6OptIfaceId(
                ifaceid=socket.inet_pton(socket.AF_INET6, self.vlan_ip))

        return relayed_relay_packet

    def create_dhcp_relay_relay_reply_packet(self):
        """Create the nested Relay-Reply sent to Relay B by its upstream."""
        relay_relay_reply_packet = packet.Ether(dst=self.uplink_mac)
        relay_relay_reply_packet /= IPv6(src=self.server_ip,
                                         dst=self.relay_listen_ip)
        relay_relay_reply_packet /= packet.UDP(
            sport=self.DHCP_SERVER_PORT, dport=self.DHCP_SERVER_PORT)
        relay_relay_reply_packet /= DHCP6_RelayReply(msgtype=13, hopcount=1, linkaddr=self.relay_linkaddr,
                                                     peeraddr=self.downstream_relay_ip)
        packet_inside = DHCP6_RelayReply(
            msgtype=13, linkaddr=self.link_addr, peeraddr=self.client_link_local)
        packet_inside /= DHCP6OptRelayMsg(message=[DHCP6_Reply(trid=12345)])
        relay_relay_reply_packet /= DHCP6OptServerId(duid=DUID_LL(lladdr="00:11:22:33:44:55"))
        iface_id = self.expected_iface_id_bytes()
        if iface_id is not None:
            relay_relay_reply_packet /= DHCP6OptIfaceId(ifaceid=iface_id)
        relay_relay_reply_packet /= DHCP6OptRelayMsg(message=[packet_inside])

        return relay_relay_reply_packet

    def create_dhcp_relay_reply_packet(self):
        """Create the Relay-Reply Relay B should forward to Relay A."""
        relay_reply_packet = packet.Ether(
            src=self.relay_iface_mac, dst=self.client_mac)
        # Source may be unknown (GPINS L3+LLA has no GUA on the port); the
        # verifier masks IPv6.src when downstream_reply_src_ip == ''.
        src_ip = self.downstream_reply_src_ip or self.relay_link_local
        relay_reply_packet /= IPv6(src=src_ip,
                                   dst=self.downstream_relay_ip)
        relay_reply_packet /= packet.UDP(sport=self.DHCP_SERVER_PORT,
                                         dport=self.DHCP_SERVER_PORT)
        relay_reply_packet /= DHCP6_RelayReply(
            msgtype=13, linkaddr=self.link_addr, peeraddr=self.client_link_local)
        relay_reply_packet /= DHCP6OptRelayMsg(
            message=[DHCP6_Reply(trid=12345)])

        return relay_reply_packet

    """
     Send/receive functions

    """

    # Simulate client connecting on VLAN and broadcasting a DHCPv6 SOLICIT message
    def client_send_solicit(self):
        # Form and send DHCPv6 SOLICIT packet
        solicit_packet = self.create_dhcp_solicit_packet()
        self.last_client_packet = solicit_packet
        testutils.send_packet(self, self.client_port_index, solicit_packet)

    @staticmethod
    def _dhcp6_options(layer):
        """Yield the DHCPv6 option layers chained after a DHCP6 message/relay header."""
        opt = layer.payload
        while opt is not None and not isinstance(opt, NoPayload):
            yield opt
            opt = opt.payload

    @staticmethod
    def _relay_msg_bytes(opt):
        """Raw bytes of the Relay-Message option payload (scapy may wrap it in a list)."""
        msg = opt.message
        if isinstance(msg, (list, tuple)):
            return b''.join(bytes(m) for m in msg)
        return bytes(msg)

    def _check_relay_forward(self, pkt, expected_inner):
        """Return None if pkt is the Relay-Forward we expect, else a short reason string."""
        if not pkt.haslayer(DHCP6_RelayForward):
            return "not a Relay-Forward"
        eth = pkt[packet.Ether]
        if eth.src.lower() != str(self.uplink_mac).lower():
            return "Ether.src %s != uplink mac %s" % (eth.src, self.uplink_mac)
        udp = pkt[packet.UDP]
        if udp.sport != self.DHCP_SERVER_PORT or udp.dport != self.DHCP_SERVER_PORT:
            return "UDP ports %d->%d" % (udp.sport, udp.dport)
        rf = pkt[DHCP6_RelayForward]
        if rf.msgtype != 12 or rf.hopcount != 0:
            return "msgtype/hopcount %s/%s" % (rf.msgtype, rf.hopcount)
        if not self._ipv6_equal(rf.linkaddr, self.link_addr):
            return "linkaddr %s != %s" % (rf.linkaddr, self.link_addr)
        if not self._ipv6_equal(rf.peeraddr, self.client_link_local):
            return "peeraddr %s != %s" % (rf.peeraddr, self.client_link_local)

        iface_ids, remote_ids, relay_msgs, has_clladdr = [], [], [], False
        for opt in self._dhcp6_options(rf):
            if isinstance(opt, DHCP6OptIfaceId):
                iface_ids.append(bytes(opt.ifaceid))
            elif isinstance(opt, DHCP6OptRemoteID):
                raw_opt_body = bytes(opt)[4:4 + opt.optlen]
                remote_ids.append((raw_opt_body, bytes(opt.remoteid)))
            elif isinstance(opt, DHCP6OptRelayMsg):
                relay_msgs.append(self._relay_msg_bytes(opt))
            elif isinstance(opt, DHCP6OptClientLinkLayerAddr) or getattr(opt, 'optcode', None) == 79:
                has_clladdr = True

        if relay_msgs != [expected_inner]:
            return "Relay-Message mismatch (%d found)" % len(relay_msgs)
        exp_iface_id = self.expected_iface_id_bytes()
        if exp_iface_id is None:
            if iface_ids:
                return "unexpected Interface-ID %r" % iface_ids
        elif iface_ids != [exp_iface_id]:
            return "Interface-ID %r != %r" % (iface_ids, exp_iface_id)
        if self.remote_id is not None:
            exp_remote = str(self.remote_id).encode()
            if len(remote_ids) != 1 or exp_remote not in remote_ids[0]:
                return "Remote-ID %r != %r" % (remote_ids, self.remote_id)
        if self.rfc6939_support and not has_clladdr:
            return "missing Client Link-Layer Address option (79)"
        return None

    def verify_relay_forward(self, client_packet, timeout=4.0):
        """Wait for the Relay-Forward encapsulating client_packet on any uplink.

        Fields are checked semantically (not byte-for-byte) so that option
        ordering or vendor formatting of Option 18/37 does not matter.
        """
        expected_inner = bytes(client_packet[packet.UDP].payload)
        reasons = []
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            result = testutils.dp_poll(self, device_number=0, timeout=remaining)
            if not isinstance(result, self.dataplane.PollSuccess):
                break
            if result.port not in self.server_port_indices:
                continue
            try:
                pkt = packet.Ether(result.packet)
            except Exception as exc:  # malformed frame: keep polling
                reasons.append("port %d: undecodable (%s)" % (result.port, exc))
                continue
            reason = self._check_relay_forward(pkt, expected_inner)
            if reason is None:
                return result.port
            reasons.append("port %d: %s" % (result.port, reason))
        self.fail("No matching Relay-Forward on uplinks %s within %.1fs. Seen: %s"
                  % (self.server_port_indices, timeout, reasons[-10:] or "nothing"))

    # Verify that the DHCP relay actually received and relayed the DHCPv6 SOLICIT message to all of
    # its known DHCP servers.
    def verify_relayed_solicit_relay_forward(self):
        self.verify_relay_forward(self.last_client_packet)

    # Simulate a DHCP server sending a DHCPv6 RELAY-REPLY encapsulating ADVERTISE packet message to client.
    # We do this by injecting a RELAY-REPLY encapsulating ADVERTISE message on the link connected to one
    # of our leaf switches.
    def server_send_advertise_relay_reply(self):
        # Form and send DHCPv6 RELAY-REPLY encapsulating ADVERTISE packet
        advertise_relay_reply_packet = self.create_dhcp_advertise_relay_reply_packet()
        advertise_relay_reply_packet.src = self.dataplane.get_mac(
            0, self.server_port_indices[0])
        testutils.send_packet(
            self, self.server_port_indices[0], advertise_relay_reply_packet)

    # Verify that the DHCPv6 ADVERTISE would be received by our simulated client
    def verify_relayed_advertise(self):
        # Create a packet resembling a DHCPv6 ADVERTISE packet
        advertise_packet = self.create_dhcp_advertise_packet()

        # Mask off fields we don't care about matching
        masked_packet = Mask(advertise_packet)
        masked_packet.set_do_not_care_scapy(IPv6, "fl")
        masked_packet.set_do_not_care_scapy(IPv6, "tc")
        masked_packet.set_do_not_care_scapy(IPv6, "hlim")
        # dual tor uses loopback0 ipv6 address as source
        masked_packet.set_do_not_care_scapy(packet.UDP, "chksum")
        masked_packet.set_do_not_care_scapy(packet.UDP, "len")

        # NOTE: verify_packet() will fail for us via an assert, so no need to check a return value here
        testutils.verify_packet(self, masked_packet, self.client_port_index)

    # Simulate our client sending a DHCPv6 REQUEST message
    def client_send_request(self):
        # Form and send DHCPv6 REQUEST packet
        request_packet = self.create_dhcp_request_packet()
        self.last_client_packet = request_packet
        testutils.send_packet(self, self.client_port_index, request_packet)

    # Verify that the DHCP relay actually received and relayed the DHCPv6 REQUEST message to all of
    # its known DHCP servers.
    def verify_relayed_request_relay_forward(self):
        self.verify_relay_forward(self.last_client_packet)

    # Simulate a DHCP server sending a DHCPv6 RELAY-REPLY encapsulating REPLY packet message to client.
    def server_send_reply_relay_reply(self):
        # Form and send DHCPv6 RELAY-REPLY encapsulating REPLY packet
        reply_relay_reply_packet = self.create_dhcp_reply_relay_reply_packet()
        reply_relay_reply_packet.src = self.dataplane.get_mac(
            0, self.server_port_indices[0])
        testutils.send_packet(
            self, self.server_port_indices[0], reply_relay_reply_packet)

    # Verify that the DHCPv6 REPLY would be received by our simulated client
    def verify_relayed_reply(self):
        # Create a packet resembling a DHCPv6 REPLY packet
        reply_packet = self.create_dhcp_reply_packet()

        # Mask off fields we don't care about matching
        masked_packet = Mask(reply_packet)
        masked_packet.set_do_not_care_scapy(IPv6, "fl")
        masked_packet.set_do_not_care_scapy(IPv6, "tc")
        masked_packet.set_do_not_care_scapy(IPv6, "hlim")
        #  dual tor uses loopback0 ipv6 address as source
        masked_packet.set_do_not_care_scapy(packet.UDP, "chksum")
        masked_packet.set_do_not_care_scapy(packet.UDP, "len")

        # NOTE: verify_packet() will fail for us via an assert, so no need to check a return value here
        testutils.verify_packet(self, masked_packet, self.client_port_index)

    def downstream_relay_send_relay_forward(self):
        """Inject a Relay-Forward from Relay B's immediate downstream."""
        relay_forward_packet = self.create_dhcp_relay_forward_packet()
        testutils.send_packet(
            self, self.client_port_index, relay_forward_packet)

    def verify_relayed_relay_forward(self):
        """Verify Relay B adds one envelope toward its immediate upstream."""
        relayed_relay_forward_packet = self.create_dhcp_relayed_relay_packet()

        masked_packet = Mask(relayed_relay_forward_packet)
        masked_packet.set_do_not_care_scapy(packet.Ether, "dst")
        masked_packet.set_do_not_care_scapy(IPv6, "src")
        masked_packet.set_do_not_care_scapy(IPv6, "dst")
        masked_packet.set_do_not_care_scapy(IPv6, "fl")
        masked_packet.set_do_not_care_scapy(IPv6, "tc")
        masked_packet.set_do_not_care_scapy(IPv6, "plen")
        masked_packet.set_do_not_care_scapy(IPv6, "nh")
        masked_packet.set_do_not_care_scapy(packet.UDP, "chksum")
        masked_packet.set_do_not_care_scapy(packet.UDP, "len")

        relayed_relay_forward_count = testutils.count_matched_packets_all_ports(
            self, masked_packet, self.server_port_indices, timeout=4.0)
        self.assertTrue(
            relayed_relay_forward_count >= 1,
            "Failed: Relayed Relay-Forward count of %d"
            % relayed_relay_forward_count)

    def server_send_relay_relay_reply(self):
        """Inject the nested Relay-Reply from Relay B's upstream."""
        relay_relay_reply_packet = self.create_dhcp_relay_relay_reply_packet()
        relay_relay_reply_packet.src = self.dataplane.get_mac(
            0, self.server_port_indices[0])
        testutils.send_packet(
            self, self.server_port_indices[0], relay_relay_reply_packet)

    def verify_relay_relay_reply(self):
        """Verify Relay B removes its outer header and forwards the enclosed Relay-Reply unchanged."""
        relay_reply_packet = self.create_dhcp_relay_reply_packet()

        # Mask off fields we don't care about matching
        masked_packet = Mask(relay_reply_packet)
        masked_packet.set_do_not_care_scapy(IPv6, "fl")
        masked_packet.set_do_not_care_scapy(IPv6, "tc")
        masked_packet.set_do_not_care_scapy(IPv6, "hlim")
        if not self.downstream_reply_src_ip:
            # No GUA on the relay port (GPINS L3+LLA): source address is implementation-defined.
            masked_packet.set_do_not_care_scapy(IPv6, "src")
        masked_packet.set_do_not_care_scapy(packet.UDP, "chksum")
        masked_packet.set_do_not_care_scapy(packet.UDP, "len")

        # NOTE: verify_packet() will fail for us via an assert, so no need to check a return value here
        testutils.verify_packet(self, masked_packet, self.client_port_index)

    def runTest(self):
        self.client_send_solicit()
        self.verify_relayed_solicit_relay_forward()

        self.server_send_advertise_relay_reply()
        self.verify_relayed_advertise()

        self.client_send_request()
        self.verify_relayed_request_relay_forward()

        self.server_send_reply_relay_reply()
        self.verify_relayed_reply()

        # Relay-to-relay topology: Client <-> Relay A (PTF) <-> Relay B (DUT) <-> Server.
        # This is different from all previous tests where our DUT is the Relay A.
        # This tests the RFC 8415 section 19.1.2 forward path. PTF simulates
        # Relay A unicasting a Relay-Forward from its GUA to Relay B's GUA.
        # Relay B should encapsulate it in another Relay-Forward and send it
        # toward the server. dhcp6relay currently does not accept this unicast
        # Relay-Forward, so the code is retained but its calls are commented out.
        # This is a feature gap for a topology SONiC does not currently use.
        # self.downstream_relay_send_relay_forward()
        # self.verify_relayed_relay_forward()
        # This explicitly tests the RFC 8415 section 19.3 return path, not the
        # normal t0/m0/mx first-hop exchange; the regular path above uses an
        # end-client link-local peer. PTF simulates downstream Relay A with a GUA,
        # so the outer Relay-Reply uses linkaddr=:: and peeraddr=<Relay A GUA>,
        # matching the relay-to-relay model implemented by dhcp6relay and validated
        # by dhcpmon. The mgmt test configures PTF's arp_responder so DUT Relay B
        # can resolve Relay A. A link-local relay-to-relay hop is allowed by
        # RFC8415, but current SONiC dhcp6relay/dhcpmon profiles assume GUA. They
        # also assume first-hop downstream VLANs, which is fundamentally incompatible
        # with making the DUT a second-or-later relay; this only models GUA return.
        # This is a hypothetical test; there is no current SONiC topology where
        # we use the DUT this way.
        self.server_send_relay_relay_reply()
        self.verify_relay_relay_reply()


class DHCPNoRelayTest(DHCPTest):
    """Negative test: a client Solicit on this port must NOT be relayed.

    Used for interfaces without a DHCP_RELAY entry (GPINS replacement for the
    upstream "VLAN without dhcpv6_servers is not bound" socket check).
    """

    def runTest(self):
        self.client_send_solicit()
        expected_inner = bytes(self.last_client_packet[packet.UDP].payload)
        deadline = time.time() + 4.0
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            result = testutils.dp_poll(self, device_number=0, timeout=remaining)
            if not isinstance(result, self.dataplane.PollSuccess):
                break
            if result.port not in self.server_port_indices:
                continue
            try:
                pkt = packet.Ether(result.packet)
            except Exception:
                continue
            if not pkt.haslayer(DHCP6_RelayForward):
                continue
            rf = pkt[DHCP6_RelayForward]
            inner = [self._relay_msg_bytes(o) for o in self._dhcp6_options(rf) if isinstance(o, DHCP6OptRelayMsg)]
            self.assertTrue(expected_inner not in inner,
                            "Solicit from unconfigured port %d was relayed on uplink port %d"
                            % (self.client_port_index, result.port))

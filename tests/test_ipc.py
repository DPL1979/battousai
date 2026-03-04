"""
test_ipc.py — Tests for battousai.ipc (IPCManager, Mailbox, Message, BulletinBoard)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest

from battousai.ipc import IPCManager, Mailbox, Message, MessageType, BROADCAST_ALL


def _make_msg(sender, recipient, msg_type=MessageType.TASK, payload=None, timestamp=0):
    """Helper to create a Message with all required fields."""
    return Message(
        sender_id=sender,
        recipient_id=recipient,
        message_type=msg_type,
        payload=payload or {},
        timestamp=timestamp,
    )


class TestMailbox(unittest.TestCase):

    def setUp(self):
        self.mailbox = Mailbox(agent_id="agent_0001", max_size=10)

    def test_mailbox_starts_empty(self):
        msgs = self.mailbox.receive_all(current_tick=0)
        self.assertEqual(len(msgs), 0)

    def test_mailbox_deliver_and_receive(self):
        msg = _make_msg("sender_0001", "agent_0001", payload={"task": "do_work"})
        self.mailbox.deliver(msg)
        msgs = self.mailbox.receive_all(current_tick=0)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].payload, {"task": "do_work"})

    def test_mailbox_respects_max_size(self):
        """Delivering more than max_size messages should not raise but cap the queue."""
        for i in range(15):
            msg = _make_msg("sender_0001", "agent_0001", MessageType.CUSTOM, {"i": i})
            self.mailbox.deliver(msg)
        # Should not exceed max_size
        self.assertLessEqual(len(self.mailbox._queue), 10)

    def test_mailbox_receive_all_clears_queue(self):
        msg = _make_msg("s_0001", "agent_0001", MessageType.STATUS)
        self.mailbox.deliver(msg)
        self.mailbox.receive_all(current_tick=0)
        # Second receive should be empty
        msgs2 = self.mailbox.receive_all(current_tick=0)
        self.assertEqual(len(msgs2), 0)


class TestMessageType(unittest.TestCase):

    def test_all_expected_message_types_exist(self):
        expected = ["TASK", "RESULT", "STATUS", "QUERY", "REPLY",
                    "BROADCAST", "HEARTBEAT", "ERROR", "CUSTOM"]
        names = [mt.name for mt in MessageType]
        for e in expected:
            self.assertIn(e, names)


class TestIPCManagerRegistration(unittest.TestCase):

    def setUp(self):
        self.ipc = IPCManager()

    def test_register_agent_creates_mailbox(self):
        self.ipc.register_agent("agent_0001")
        mailbox = self.ipc.get_mailbox("agent_0001")
        self.assertIsNotNone(mailbox)

    def test_unregister_agent_removes_mailbox(self):
        self.ipc.register_agent("agent_0001")
        self.ipc.unregister_agent("agent_0001")
        mailbox = self.ipc.get_mailbox("agent_0001")
        self.assertIsNone(mailbox)

    def test_get_mailbox_for_unknown_agent_returns_none(self):
        mailbox = self.ipc.get_mailbox("no_such_agent")
        self.assertIsNone(mailbox)


class TestIPCManagerUnicast(unittest.TestCase):

    def setUp(self):
        self.ipc = IPCManager()
        self.ipc.register_agent("sender_0001")
        self.ipc.register_agent("recipient_0001")

    def test_send_message_delivers_to_recipient_mailbox(self):
        msg = _make_msg("sender_0001", "recipient_0001", payload={"job": "process"})
        self.ipc.send(msg)
        mailbox = self.ipc.get_mailbox("recipient_0001")
        msgs = mailbox.receive_all(current_tick=0)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].payload["job"], "process")

    def test_send_to_unknown_recipient_does_not_raise(self):
        msg = _make_msg("sender_0001", "ghost_9999")
        # Should silently fail or return error, not raise
        try:
            self.ipc.send(msg)
        except Exception as e:
            self.fail(f"send() raised unexpectedly: {e}")

    def test_multiple_messages_accumulate_in_mailbox(self):
        for i in range(3):
            msg = _make_msg("sender_0001", "recipient_0001", MessageType.CUSTOM, {"n": i})
            self.ipc.send(msg)
        mailbox = self.ipc.get_mailbox("recipient_0001")
        msgs = mailbox.receive_all(current_tick=0)
        self.assertEqual(len(msgs), 3)


class TestIPCManagerBroadcast(unittest.TestCase):

    def setUp(self):
        self.ipc = IPCManager()
        self.ipc.register_agent("sender_0001")
        self.ipc.register_agent("recv_a")
        self.ipc.register_agent("recv_b")

    def test_broadcast_reaches_all_registered_agents(self):
        msg = _make_msg("sender_0001", BROADCAST_ALL, MessageType.BROADCAST,
                        {"alert": "system_update"})
        self.ipc.send(msg)
        for agent_id in ["recv_a", "recv_b"]:
            mailbox = self.ipc.get_mailbox(agent_id)
            msgs = mailbox.receive_all(current_tick=0)
            self.assertGreater(len(msgs), 0, f"Agent {agent_id} got no broadcast")


class TestIPCManagerPubSub(unittest.TestCase):
    """
    The IPC bulletin board is a key-value store (last value wins).
    Subscribers are tracked informally; the board does NOT push messages
    into mailboxes — subscribers poll via board_read().
    """

    def setUp(self):
        self.ipc = IPCManager()
        self.ipc.register_agent("publisher_0001")
        self.ipc.register_agent("subscriber_a")
        self.ipc.register_agent("subscriber_b")
        self.ipc.register_agent("non_subscriber")

    def test_subscribe_and_publish_reaches_subscriber(self):
        """After publish, board_read returns the published value."""
        self.ipc.subscribe("news", "subscriber_a")
        self.ipc.publish("news", {"headline": "Battousai launched"}, "publisher_0001", tick=1)
        value = self.ipc.board_read("news")
        self.assertIsNotNone(value)
        self.assertEqual(value.get("headline"), "Battousai launched")

    def test_non_subscriber_board_read_still_works(self):
        """Board is global; non-subscriber can also read any published topic."""
        self.ipc.publish("alerts", {"level": "critical"}, "publisher_0001", tick=1)
        # Even non-subscribers can read the board (it's a shared KV store)
        value = self.ipc.board_read("alerts")
        self.assertIsNotNone(value)

    def test_multiple_subscribers_all_can_read_published_topic(self):
        """Two subscribers both see the published value via board_read."""
        self.ipc.subscribe("alerts", "subscriber_a")
        self.ipc.subscribe("alerts", "subscriber_b")
        self.ipc.publish("alerts", {"level": "critical"}, "publisher_0001", tick=1)
        for agent_id in ["subscriber_a", "subscriber_b"]:
            value = self.ipc.board_read("alerts")
            self.assertIsNotNone(value)

    def test_publish_overwrites_previous_value(self):
        self.ipc.publish("status", "v1", "publisher_0001", tick=1)
        self.ipc.publish("status", "v2", "publisher_0001", tick=2)
        self.assertEqual(self.ipc.board_read("status"), "v2")

    def test_board_read_unknown_topic_returns_none(self):
        result = self.ipc.board_read("no_such_topic")
        self.assertIsNone(result)


class TestIPCManagerCreateMessage(unittest.TestCase):

    def setUp(self):
        self.ipc = IPCManager()
        self.ipc.register_agent("sender")
        self.ipc.register_agent("receiver")

    def test_create_message_and_delivers(self):
        """create_message() factory creates and sends atomically."""
        msg = self.ipc.create_message(
            sender_id="sender",
            recipient_id="receiver",
            message_type=MessageType.QUERY,
            payload={"q": "status"},
            timestamp=5,
        )
        self.assertIsInstance(msg, Message)
        mb = self.ipc.get_mailbox("receiver")
        msgs = mb.receive_all(0)
        self.assertEqual(len(msgs), 1)

    def test_stats_tracks_sent_count(self):
        msg = _make_msg("sender", "receiver")
        self.ipc.send(msg)
        stats = self.ipc.stats()
        self.assertGreaterEqual(stats["total_sent"], 1)


if __name__ == "__main__":
    unittest.main()

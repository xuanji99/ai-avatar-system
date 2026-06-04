"""
Tests for the real-time WebSocket pipeline's concurrency contract.

These exercise `ConnectionManager` directly with a fake socket so we don't
need a live DB, LLM, or GPU. The properties under test are the ones that
make barge-in actually work:

  * `handle_text_input` dispatches without blocking (returns before the
    turn finishes) — otherwise the WS receive loop can never observe an
    interrupt.
  * A second input cancels the first in-flight turn.
  * An explicit interrupt cancels the turn and notifies the client.
  * Input validation rejects empty / oversized messages up front.
"""

import asyncio

import pytest

from app.websocket import (
    _CLAUSE_RE,
    _MAX_CHUNK_CHARS,
    _MIN_FIRST_CHUNK_LEN,
    _MIN_SENTENCE_LEN,
    _SENTENCE_RE,
    MAX_TEXT_INPUT_LEN,
    ConnectionManager,
    _drain_chunks,
)

# ── chunker (first-frame latency) ───────────────────────────────────────────


def test_drain_chunks_emits_at_clause_for_first_fragment():
    """A clause boundary ships the opening fragment before the sentence ends."""
    buf = "Sure thing, let me look that up for you right now."
    chunks, rest = _drain_chunks(buf, _CLAUSE_RE, _MIN_FIRST_CHUNK_LEN, _MAX_CHUNK_CHARS)
    # "Sure thing," is >= 12 chars → emitted at the comma, not the period.
    assert chunks
    assert chunks[0].startswith("Sure thing,")


def test_drain_chunks_never_drops_text():
    """Short leading fragments merge forward rather than being discarded."""
    buf = "Hi, the answer is 42 and that is final."
    chunks, rest = _drain_chunks(buf, _CLAUSE_RE, _MIN_FIRST_CHUNK_LEN, _MAX_CHUNK_CHARS)
    reassembled = " ".join(chunks)
    if rest.strip():
        reassembled = (reassembled + " " + rest).strip()
    for word in ["Hi,", "answer", "42", "final."]:
        assert word in reassembled


def test_drain_chunks_force_flush_runon():
    """A long run-on with no punctuation is cut at a space, not held forever."""
    buf = "word " * 60  # 300 chars, no sentence punctuation
    chunks, rest = _drain_chunks(buf, _SENTENCE_RE, _MIN_SENTENCE_LEN, _MAX_CHUNK_CHARS)
    assert chunks  # something was force-flushed
    assert all(len(c) <= _MAX_CHUNK_CHARS for c in chunks)


def test_drain_chunks_holds_incomplete_buffer():
    """With no boundary and under the cap, nothing is emitted yet."""
    chunks, rest = _drain_chunks(
        "partial thought with no end", _SENTENCE_RE, _MIN_SENTENCE_LEN, _MAX_CHUNK_CHARS
    )
    assert chunks == []
    assert rest == "partial thought with no end"


class FakeWebSocket:
    """Minimal stand-in that records everything sent to the client."""

    def __init__(self):
        self.sent: list[dict] = []
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def send_json(self, message):
        self.sent.append(message)


def _wire_session(manager: ConnectionManager, session_id: str = "s1", user_id: str = "u1"):
    """Attach a fake connected session without going through the DB-backed connect()."""
    ws = FakeWebSocket()
    manager.active_connections[session_id] = ws
    manager._send_locks[session_id] = asyncio.Lock()
    manager.session_data[session_id] = {
        "messages": [],
        "user_id": user_id,
        "language": "en",
        "system_prompt": None,
        "voice_wav": None,
        "avatar_image_local": None,
    }
    return ws


@pytest.mark.asyncio
async def test_handle_text_input_is_non_blocking():
    """Dispatch must return immediately, before the turn completes."""
    m = ConnectionManager()
    _wire_session(m)

    started = asyncio.Event()

    async def slow_turn(session_id, text):
        started.set()
        await asyncio.sleep(5)  # simulate a long LLM+TTS+animation turn

    m._handle_text_input_inner = slow_turn  # type: ignore[assignment]

    # If dispatch blocked on the turn, this would take ~5s and time out.
    await asyncio.wait_for(m.handle_text_input("s1", "hello there"), timeout=0.5)

    assert "s1" in m._active_turns
    await asyncio.wait_for(started.wait(), timeout=1)  # the turn really started

    # cleanup
    await m.interrupt_active_turn("s1")


@pytest.mark.asyncio
async def test_second_input_interrupts_first():
    """A new turn cancels the previous in-flight turn (barge-in)."""
    m = ConnectionManager()
    _wire_session(m)

    async def slow_turn(session_id, text):
        await asyncio.sleep(5)

    m._handle_text_input_inner = slow_turn  # type: ignore[assignment]

    await m.handle_text_input("s1", "first message")
    first_task = m._active_turns["s1"]

    await m.handle_text_input("s1", "second message")
    await asyncio.sleep(0.05)  # let the cancellation settle

    assert first_task.cancelled() or first_task.done()
    # The new turn is now the active one.
    assert m._active_turns["s1"] is not first_task

    await m.interrupt_active_turn("s1")


@pytest.mark.asyncio
async def test_explicit_interrupt_notifies_client():
    """interrupt_active_turn cancels and emits an `interrupted` event."""
    m = ConnectionManager()
    ws = _wire_session(m)

    async def slow_turn(session_id, text):
        await asyncio.sleep(5)

    m._handle_text_input_inner = slow_turn  # type: ignore[assignment]
    await m.handle_text_input("s1", "talk to me")

    interrupted = await m.interrupt_active_turn("s1")
    assert interrupted is True
    assert "s1" not in m._active_turns
    assert any(msg["type"] == "interrupted" for msg in ws.sent)


@pytest.mark.asyncio
async def test_interrupt_with_no_active_turn_is_noop():
    m = ConnectionManager()
    _wire_session(m)
    assert await m.interrupt_active_turn("s1") is False


@pytest.mark.asyncio
async def test_empty_text_rejected():
    m = ConnectionManager()
    ws = _wire_session(m)
    await m.handle_text_input("s1", "   ")
    assert any(msg["type"] == "error" for msg in ws.sent)
    assert "s1" not in m._active_turns  # no turn spawned


@pytest.mark.asyncio
async def test_oversized_text_rejected():
    m = ConnectionManager()
    ws = _wire_session(m)
    await m.handle_text_input("s1", "x" * (MAX_TEXT_INPUT_LEN + 1))
    assert any(
        "too long" in msg.get("message", "").lower() for msg in ws.sent if msg["type"] == "error"
    )
    assert "s1" not in m._active_turns


@pytest.mark.asyncio
async def test_set_language_coerces_unknown():
    m = ConnectionManager()
    _wire_session(m)
    await m.set_language("s1", "klingon")
    assert m.session_data["s1"]["language"] == "en"
    await m.set_language("s1", "fr")
    assert m.session_data["s1"]["language"] == "fr"


@pytest.mark.asyncio
async def test_set_voice_rejects_cross_tenant(monkeypatch):
    """A voice owned by another user must not attach to this session."""
    m = ConnectionManager()
    _wire_session(m, user_id="owner-A")

    async def fake_entry(voice_id):
        return {"id": voice_id, "user_id": "owner-B", "wav_path": "/tmp/x.wav"}

    monkeypatch.setattr(m, "_get_voice_entry", fake_entry)
    ok = await m.set_voice_by_id("s1", "some-voice")
    assert ok is False
    assert m.session_data["s1"]["voice_wav"] is None


@pytest.mark.asyncio
async def test_audio_transcribed_in_session_language(monkeypatch):
    """STT must use the session's selected language, not always English."""
    import base64

    from app.services import stt as stt_module

    m = ConnectionManager()
    _wire_session(m)
    m.session_data["s1"]["language"] = "fr"

    captured = {}

    async def fake_transcribe(audio, language="en"):
        captured["language"] = language
        return ""  # empty → handler returns before spawning a turn

    monkeypatch.setattr(stt_module.stt_service, "transcribe", fake_transcribe)
    # _handle_audio_inner reads the module-level stt_service via the ws module
    from app import websocket as wsmod

    monkeypatch.setattr(wsmod.stt_service, "transcribe", fake_transcribe)

    await m._handle_audio_inner("s1", base64.b64encode(b"x" * 2000).decode())
    assert captured.get("language") == "fr"

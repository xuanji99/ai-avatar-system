"""
Tests for the conversation + message endpoints, including ownership scoping.

Verifies that the auth/ownership guards added during the security pass
actually hold — a user can only see and mutate their own conversations
and messages.
"""

import pytest
from httpx import AsyncClient

from app.models import Avatar, Conversation, Message, Session


async def _seed_session(db_session, user_id: str, status: str = "active") -> Session:
    avatar = Avatar(
        user_id=user_id,
        name="Test Avatar",
        image_url="http://x/i.jpg",
        thumbnail_url="http://x/t.jpg",
        s3_key="avatars/x/image.jpg",
        status="ready",
    )
    db_session.add(avatar)
    await db_session.commit()
    await db_session.refresh(avatar)

    session = Session(user_id=user_id, avatar_id=avatar.id, status=status)
    db_session.add(session)
    await db_session.commit()
    await db_session.refresh(session)
    return session


@pytest.mark.asyncio
async def test_list_conversations_scoped_to_user(
    client: AsyncClient, db_session, test_user, auth_headers
):
    """A user only sees conversations belonging to their own sessions."""
    # Own session + conversation
    own = await _seed_session(db_session, test_user.id)
    db_session.add(Conversation(session_id=own.id, title="Mine", message_count=0))

    # Another user's session + conversation
    other = await _seed_session(db_session, "someone-else")
    db_session.add(Conversation(session_id=other.id, title="Theirs", message_count=0))
    await db_session.commit()

    resp = await client.get("/api/v1/conversations/", headers=auth_headers)
    assert resp.status_code == 200
    titles = [c["title"] for c in resp.json()]
    assert "Mine" in titles
    assert "Theirs" not in titles


@pytest.mark.asyncio
async def test_conversation_message_count_is_live(
    client: AsyncClient, db_session, test_user, auth_headers
):
    """message_count reflects the live message total, not the stale stored 0."""
    session = await _seed_session(db_session, test_user.id)
    # Conversation row seeded with the stale default count of 0...
    db_session.add(Conversation(session_id=session.id, title="Chat", message_count=0))
    # ...but the session actually has 3 messages.
    db_session.add(Message(session_id=session.id, role="user", content="a", content_type="text"))
    db_session.add(
        Message(session_id=session.id, role="assistant", content="b", content_type="text")
    )
    db_session.add(Message(session_id=session.id, role="user", content="c", content_type="text"))
    await db_session.commit()

    resp = await client.get("/api/v1/conversations/", headers=auth_headers)
    assert resp.status_code == 200
    mine = [c for c in resp.json() if c["title"] == "Chat"][0]
    assert mine["message_count"] == 3  # live count, not the stored 0


@pytest.mark.asyncio
async def test_rename_conversation(client: AsyncClient, db_session, test_user, auth_headers):
    session = await _seed_session(db_session, test_user.id)
    convo = Conversation(session_id=session.id, title="Old", message_count=0)
    db_session.add(convo)
    await db_session.commit()
    await db_session.refresh(convo)

    resp = await client.patch(
        f"/api/v1/conversations/{convo.id}/rename",
        json={"title": "New Title"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["title"] == "New Title"


@pytest.mark.asyncio
async def test_cannot_rename_others_conversation(
    client: AsyncClient, db_session, test_user, auth_headers
):
    other = await _seed_session(db_session, "someone-else")
    convo = Conversation(session_id=other.id, title="Theirs", message_count=0)
    db_session.add(convo)
    await db_session.commit()
    await db_session.refresh(convo)

    resp = await client.patch(
        f"/api/v1/conversations/{convo.id}/rename",
        json={"title": "Hacked"},
        headers=auth_headers,
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_message_edit_and_delete(client: AsyncClient, db_session, test_user, auth_headers):
    session = await _seed_session(db_session, test_user.id)
    msg = Message(session_id=session.id, role="user", content="original", content_type="text")
    db_session.add(msg)
    await db_session.commit()
    await db_session.refresh(msg)

    # Edit
    resp = await client.patch(
        f"/api/v1/messages/{msg.id}",
        json={"content": "edited"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["content"] == "edited"

    # Delete
    resp = await client.delete(f"/api/v1/messages/{msg.id}", headers=auth_headers)
    assert resp.status_code == 204

    # Gone
    resp = await client.get(f"/api/v1/messages/{msg.id}", headers=auth_headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_session_export(client: AsyncClient, db_session, test_user, auth_headers):
    session = await _seed_session(db_session, test_user.id)
    db_session.add(Message(session_id=session.id, role="user", content="hi", content_type="text"))
    db_session.add(
        Message(session_id=session.id, role="assistant", content="hello", content_type="text")
    )
    await db_session.commit()

    resp = await client.get(f"/api/v1/sessions/{session.id}/export", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["session"]["id"] == session.id
    assert len(body["messages"]) == 2
    assert body["truncated"] is False
    assert "attachment" in resp.headers.get("content-disposition", "")

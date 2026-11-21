"""
Tests for the multi-agent voice assistant.

Tests verify:
- GatekeeperAgent routes to ProfileAgent when caller asks about the owner
- GatekeeperAgent routes to WebDemoAgent when caller wants a general interaction
- WebDemoAgent responds to a web search question
- WebDemoAgent refuses a third query and offers to hand off to the owner
- Language detection heuristic (_detect_language) works for Hindi and Hinglish
- Detected language is preserved and used in on_enter greeting after handoff

Run with: uv run pytest tests/ -v
"""

import os
import sys

import pytest
from dotenv import load_dotenv
from livekit.agents import AgentSession, inference

# Load credentials from .env.local (same as the agent entrypoint)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env.local"))

# Ensure the src package is importable before importing from agent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import (  # noqa: E402
    OWNER_NAME,
    GatekeeperAgent,
    ProfileAgent,
    UserData,
    WebDemoAgent,
)

# ---------------------------------------------------------------------------
# Helper: shared LLM for judge calls (uses LiveKit Inference)
# ---------------------------------------------------------------------------

_JUDGE_MODEL = "google/gemini-2.0-flash-001"

# Skip all tests in this module when LiveKit credentials are absent.
# Set LIVEKIT_API_KEY (and LIVEKIT_URL, LIVEKIT_API_SECRET) in .env.local
# or as environment variables before running tests.
_has_credentials = bool(os.getenv("LIVEKIT_API_KEY"))
pytestmark = pytest.mark.skipif(
    not _has_credentials,
    reason="LIVEKIT_API_KEY not set — skipping live LLM tests. Set credentials in .env.local to run.",
)


# ---------------------------------------------------------------------------
# GatekeeperAgent tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gatekeeper_greets_and_asks_intent() -> None:
    """GatekeeperAgent should greet and ask about the caller's intent."""
    async with (
        inference.LLM(model=_JUDGE_MODEL) as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(GatekeeperAgent())
        result = await session.run(user_input="Hello")
        await result.expect.next_event(type="message").judge(
            llm,
            intent=f"Should greet the caller and ask whether they want to learn about {OWNER_NAME} or try the AI demo.",
        )


@pytest.mark.asyncio
async def test_gatekeeper_routes_to_profile_on_profile_intent() -> None:
    """GatekeeperAgent should call transfer_to_profile when caller wants info about the owner."""
    async with (
        inference.LLM(model=_JUDGE_MODEL) as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(GatekeeperAgent())
        result = await session.run(
            user_input=f"I'm a recruiter and I'd like to know more about {OWNER_NAME}'s background."
        )
        # Expect the tool call to transfer to the owner profile
        result.expect.next_event(type="function_call").assert_match(
            name="transfer_to_profile"
        )


@pytest.mark.asyncio
async def test_gatekeeper_routes_to_web_demo_on_general_intent() -> None:
    """GatekeeperAgent should call transfer_to_web_demo when caller wants a general interaction."""
    async with (
        inference.LLM(model=_JUDGE_MODEL) as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(GatekeeperAgent())
        result = await session.run(
            user_input="I just want to try out the AI and see what it can do."
        )
        result.expect.next_event(type="function_call").assert_match(
            name="transfer_to_web_demo"
        )


# ---------------------------------------------------------------------------
# WebDemoAgent tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_web_demo_answers_first_search_question() -> None:
    """WebDemoAgent should perform a web search for the first question."""
    async with (
        inference.LLM(model=_JUDGE_MODEL) as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(WebDemoAgent())
        result = await session.run(user_input="What is the capital of France?")
        # Should produce an answer (either direct or after a search tool call)
        await result.expect.last_event(type="message").judge(
            llm,
            intent="Should answer that the capital of France is Paris.",
        )


@pytest.mark.asyncio
async def test_web_demo_refuses_third_search_and_redirects() -> None:
    """WebDemoAgent should refuse a 3rd question and offer to switch to the owner profile."""
    async with (
        inference.LLM(model=_JUDGE_MODEL) as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(WebDemoAgent())

        # Two allowed questions
        await session.run(user_input="What is the capital of France?")
        await session.run(user_input="Who invented the telephone?")

        # Third question — should be refused
        result = await session.run(user_input="What is the speed of light?")
        await result.expect.last_event(type="message").judge(
            llm,
            intent=(
                "Should decline to answer more questions and offer to either "
                f"talk about {OWNER_NAME} or end the call. Should NOT perform another search."
            ),
        )


@pytest.mark.asyncio
async def test_web_demo_offers_profile_handoff_after_limit() -> None:
    """After reaching the search limit, WebDemoAgent should offer the transfer_to_profile tool."""
    async with (
        inference.LLM(model=_JUDGE_MODEL) as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(WebDemoAgent())

        # Exhaust the limit
        await session.run(user_input="What is the capital of France?")
        await session.run(user_input="Who invented the telephone?")
        await session.run(user_input="What is the speed of light?")

        # Caller now asks to hear about the owner — should hand off
        result = await session.run(
            user_input=f"Alright, tell me about {OWNER_NAME} then."
        )
        result.expect.next_event(type="function_call").assert_match(
            name="transfer_to_profile"
        )


# ---------------------------------------------------------------------------
# Language preservation across handoff — live LLM tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hindi_preserved_on_profile_handoff() -> None:
    """ProfileAgent on_enter greeting should be in Hindi when userdata.detected_language='Hindi'."""
    async with (
        inference.LLM(model=_JUDGE_MODEL) as llm,
        AgentSession[UserData](
            llm=llm, userdata=UserData(detected_language="Hindi")
        ) as session,
    ):
        await session.start(ProfileAgent())
        # on_enter fires automatically; give the agent an empty turn to capture its greeting
        result = await session.run(user_input="")
        await result.expect.last_event(type="message").judge(
            llm,
            intent=(
                f"Should greet the caller and ask what they want to know about {OWNER_NAME},"
                " and the response should be entirely in Hindi (Devanagari or romanised Hindi)."
                " It must NOT be in English."
            ),
        )


@pytest.mark.asyncio
async def test_hinglish_preserved_on_web_demo_handoff() -> None:
    """WebDemoAgent on_enter greeting should be in Hinglish when userdata.detected_language='Hinglish'."""
    async with (
        inference.LLM(model=_JUDGE_MODEL) as llm,
        AgentSession[UserData](
            llm=llm, userdata=UserData(detected_language="Hinglish")
        ) as session,
    ):
        await session.start(WebDemoAgent())
        result = await session.run(user_input="")
        await result.expect.last_event(type="message").judge(
            llm,
            intent=(
                "Should tell the caller it can answer up to two web search questions"
                " and ask what they'd like to know."
                " The response should be in Hindi or Hinglish (a mix of Hindi and English),"
                " not purely in formal English."
            ),
        )


@pytest.mark.asyncio
async def test_english_preserved_on_handoff() -> None:
    """ProfileAgent on_enter greeting should be in English when userdata.detected_language='English'."""
    async with (
        inference.LLM(model=_JUDGE_MODEL) as llm,
        AgentSession[UserData](
            llm=llm, userdata=UserData(detected_language="English")
        ) as session,
    ):
        await session.start(ProfileAgent())
        result = await session.run(user_input="")
        await result.expect.last_event(type="message").judge(
            llm,
            intent=(
                f"Should greet the caller and ask what they want to know about {OWNER_NAME}."
                " The response should be in English."
            ),
        )

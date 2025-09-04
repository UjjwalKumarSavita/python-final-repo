"""
AutoGen AgentChat (0.7.x) team: Planner → Summarizer → Critic

Expose two helpers:
- autogen_summarize_async(...)  -> async str  (use inside FastAPI endpoints)
- autogen_summarize(...)        -> sync  str  (use in scripts/CLI)

Requires:
- OPENAI_API_KEY (and optionally OPENAI_BASE_URL / OPENAI_API_VERSION)
- OPENAI_MODEL (default gpt-4o-mini)
"""
from __future__ import annotations
import asyncio
import os
import textwrap
from typing import Optional

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
# from autogen_agentchat.groupchat import GroupChat, GroupChatManager
from autogen_agentchat.groupchat import GroupChat, GroupChatManager
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.conditions import MaxMessageTermination


def _user_prompt(doc_text: str, target_words: int) -> str:
    return textwrap.dedent(f"""
    Summarize the document via plan → write → critique.

    <DOC>
    {doc_text}
    </DOC>

    Target length: ~{target_words} words.
    """).strip()


def _sys_planner(target_words: int) -> str:
    return textwrap.dedent(f"""
    You are Planner. Output 3–6 bullet points that MUST be covered in the final summary
    to reach about {target_words} words. Do not write the summary itself.
    """).strip()


def _sys_summarizer(target_words: int) -> str:
    return textwrap.dedent(f"""
    You are Summarizer. Write a clear, self-contained summary (~{target_words} words).
    Preserve key entities, numbers, and dates. Use concise paragraphs; avoid filler.
    """).strip()


def _sys_critic() -> str:
    return textwrap.dedent("""
    You are Critic. Review the latest summary for clarity, correctness, coverage, and length.
    If acceptable, reply exactly: APPROVED
    Otherwise reply: REVISE: <one-paragraph guidance>.
    """).strip()


def _client(temperature: float, seed: Optional[int]) -> OpenAIChatCompletionClient:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for AutoGen AgentChat.")
    return OpenAIChatCompletionClient(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        api_version=os.getenv("OPENAI_API_VERSION") or None,
        temperature=float(temperature),
        extra_body={"seed": seed} if seed is not None else None,
    )


async def autogen_summarize_async(
    doc_text: str,
    *,
    target_words: int = 350,
    temperature: float = 0.3,
    seed: Optional[int] = None,
) -> str:
    client = _client(temperature, seed)

    planner = AssistantAgent("Planner", system_message=_sys_planner(target_words), model_client=client)
    summarizer = AssistantAgent("Summarizer", system_message=_sys_summarizer(target_words), model_client=client)
    critic = AssistantAgent("Critic", system_message=_sys_critic(), model_client=client)

    initial = TextMessage(content=_user_prompt(doc_text, target_words), source="user")

    gc = GroupChat(
        agents=[planner, summarizer, critic],
        messages=[initial],
        termination_condition=MaxMessageTermination(max_turns=8),
    )
    manager = GroupChatManager(groupchat=gc)
    await manager.run()

    last_summary = ""
    for msg in reversed(manager.groupchat.messages):
        role = getattr(msg, "source", "") or getattr(msg, "name", "")
        content = (getattr(msg, "content", "") or "").strip()
        if not content:
            continue
        if role == "Critic" and content.startswith("APPROVED"):
            idx = manager.groupchat.messages.index(msg)
            for prev in reversed(manager.groupchat.messages[:idx]):
                if getattr(prev, "source", "") == "Summarizer":
                    return (prev.content or "").strip()
            return last_summary or "Summary approved but not found."
        if role == "Summarizer" and not last_summary:
            last_summary = content

    return last_summary or "Summary not produced."


def autogen_summarize(
    doc_text: str,
    *,
    target_words: int = 350,
    temperature: float = 0.3,
    seed: Optional[int] = None,
) -> str:
    # For scripts/CLI use: sync wrapper around the async API
    return asyncio.run(autogen_summarize_async(doc_text, target_words=target_words, temperature=temperature, seed=seed))

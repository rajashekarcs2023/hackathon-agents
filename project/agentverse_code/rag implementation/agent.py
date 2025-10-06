#agent address : agent1qwv3zyqeu5f3uh3yd6ns9c7ua6r6zh8x9ct8s589d0c4wc2y0ltggre6cqw

import os
from datetime import datetime
from uuid import uuid4

from uagents import Agent, Context, Protocol, Model
from uagents.experimental.quota import QuotaProtocol, RateLimit
from uagents_core.models import ErrorMessage
from uagents_core.contrib.protocols.chat import (
    chat_protocol_spec,
    ChatMessage,
    ChatAcknowledgement,
    TextContent,
    EndSessionContent,
    StartSessionContent
)

from vector_store import query_vector_db
from ai import get_completion

AGENT_NAME = os.getenv("AGENT_NAME", "Fetch Docs Assistant")
AGENT_SEED = os.getenv("AGENT_SEED", "fetch‑docs‑assistant‑seed")
PORT       = int(os.getenv("PORT", "8000"))


agent = Agent(
    name=AGENT_NAME,
    seed=AGENT_SEED,
    port=PORT,
    endpoint=[f"http://localhost:{PORT}/submit"],
)

quota_proto = QuotaProtocol(
    storage_reference=agent.storage,
    name="Docs‑QA",
    version="0.1.0",
    default_rate_limit=RateLimit(window_size_minutes=5, max_requests=15),
)

class DocumentationRequest(Model):
    text: str

class DocumentationResponse(Model):
    text: str

@quota_proto.on_message(
    model=DocumentationRequest,
    replies={DocumentationResponse, ErrorMessage},
)
async def docs_handler(ctx: Context, sender: str, msg: DocumentationRequest):
    ctx.logger.info(f"[Docs‑QA] {msg.text}")
    matches = query_vector_db(msg.text)
    if not matches:
        await ctx.send(sender, ErrorMessage(error="No relevant context found."))
        return

    context = "\n\n".join(
        f"Source: {m['metadata'].get('source','')}\n\nContent: {m['metadata'].get('text','')}"
        for m in matches
    )
    answer = get_completion(context=context, query=msg.text)
    await ctx.send(sender, DocumentationResponse(text=answer))


chat_proto = Protocol(spec=chat_protocol_spec)

@chat_proto.on_message(model=ChatAcknowledgement)
async def ack_handler(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"[ACK] from {sender} for msg {msg.acknowledged_msg_id}")

@chat_proto.on_message(model=ChatMessage)
async def chat_handler(ctx: Context, sender: str, msg: ChatMessage):

    if all(isinstance(c, StartSessionContent) for c in msg.content):
        return

    user_text = " ".join(
        (getattr(c, "text", "") or getattr(c, "markdown", ""))
        for c in msg.content
    ).strip()

    if not user_text:               
        return                     

    matches = query_vector_db(user_text)
    if matches:
        context = "\n\n".join(
            f"Source: {m['metadata'].get('source','')}\n\nContent: {m['metadata'].get('text','')}"
            for m in matches
        )
        answer = get_completion(context=context, query=user_text)
    else:
        answer = "I couldn't find relevant information in the current documentation."

    await ctx.send(
        sender,
        ChatMessage(
            timestamp=datetime.utcnow(),
            msg_id=uuid4(),
            content=[TextContent(type="text", text=answer)],
        ),
    )

agent.include(quota_proto, publish_manifest=True)
agent.include(chat_proto,  publish_manifest=True)

if __name__ == "__main__":
    agent.run()

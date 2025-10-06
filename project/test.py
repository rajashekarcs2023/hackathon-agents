# mailbox_tester.py
from uagents import Agent, Context, Model
from uagents_core.contrib.protocols.chat import ChatMessage, TextContent, chat_protocol_spec
from datetime import datetime
from uuid import uuid4

class DocumentationRequest(Model):
    text: str
class DocumentationResponse(Model):
    text: str

SEED = "tester tester tester tester tester tester tester tester tester tester tester"
tester = Agent(
    name="tester",
    seed=SEED,
    port=8010,
    mailbox=True,                 
)

TARGET = "agent1qwv3zyqeu5f3uh3yd6ns9c7ua6r6zh8x9ct8s589d0c4wc2y0ltggre6cqw"

@tester.on_event("startup")
async def startup(ctx: Context):
    await ctx.send(
        TARGET,
        ChatMessage(
            timestamp=datetime.utcnow(),
            msg_id=uuid4(),
            content=[TextContent(type="text", text="What is the FET token?")],
        ),
    )

@tester.on_message(model=ChatMessage)
async def receive(ctx: Context, sender: str, msg: ChatMessage):
    text = next((c.text for c in msg.content if isinstance(c, TextContent)), "")
    print("ANSWER:", text)

if __name__ == "__main__":
    tester.run()


user_text = " ".join(
    getattr(item, "text", "") for item in msg.content
).strip()

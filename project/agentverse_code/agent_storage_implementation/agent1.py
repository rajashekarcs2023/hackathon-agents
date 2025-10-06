#Agent address : agent1qvgrfkghs0qzqqgu65sfezjv8x84xgwhg06rn0p4gl0eur6mxp05v9hfd9a

import os
import json
import requests
from datetime import datetime
from uuid import uuid4
from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)

# Create the agent
agent = Agent()

# Initialize the chat protocol
chat_proto = Protocol(spec=chat_protocol_spec)

# Static information storage - organized by categories (easily extensible)
MHACKS_INFO = {
    "prizes": {
        "best_use_fetchai": {
            "amount": "$1250",
            "description": "Cash Prize + Internship Interview Opportunity",
            "criteria": "Demonstrates the fullest use of the Fetch.ai tech stack. Teams must register agents on Agentverse, enable chat protocol, and integrate ASI:One as reasoning engine.",
            "requirements": ["Register agents on Agentverse", "Enable chat protocol", "Integrate ASI:One", "End-to-end implementation"]
        },
        "best_deployment": {
            "amount": "$750", 
            "description": "Cash Prize + Internship Interview Opportunity",
            "criteria": "Publishes highest number of useful, discoverable, and well-documented agents on Agentverse.",
            "requirements": ["Multiple agents", "Clear documentation", "Discoverable on Agentverse", "Useful functionality"]
        },
        "best_use_asione": {
            "amount": "$500",
            "description": "Cash Prize + Internship Interview Opportunity", 
            "criteria": "Most effective application of ASI:One as core reasoning and decision-making engine.",
            "requirements": ["ASI:One integration", "Smart interactions", "Reasoning capabilities"]
        }
    },
    
    "judging_criteria": {
        "functionality_technical": {
            "weight": "25%",
            "title": "Functionality & Technical Implementation",
            "description": "Does the agent system work as intended? Are the agents properly communicating and reasoning in real time?"
        },
        "fetchai_usage": {
            "weight": "20%",
            "title": "Use of Fetch.ai Technology", 
            "description": "Are agents registered on Agentverse? Is the Chat Protocol implemented for ASI:One discoverability?"
        },
        "innovation_creativity": {
            "weight": "20%",
            "title": "Innovation & Creativity",
            "description": "How original or creative is the solution? Is it solving a problem in a new or unconventional way?"
        },
        "real_world_impact": {
            "weight": "20%",
            "title": "Real-World Impact & Usefulness",
            "description": "Does the solution solve a meaningful problem? How useful would this be to an end user?"
        },
        "user_experience": {
            "weight": "15%",
            "title": "User Experience & Presentation",
            "description": "Is the solution presented clearly with a well-structured demo? Is there a smooth and intuitive user experience?"
        }
    },

    "tech_stack": {
        "uagents": {
            "description": "Python library for building autonomous agents",
            "purpose": "Create agents that can communicate and coordinate tasks",
            "installation": "pip install uagents",
            "documentation": "https://docs.fetch.ai/uAgents",
            "innovation_lab_guide": "https://innovationlab.fetch.ai/resources/docs/agent-creation/uagent-creation"
        },
        "agentverse": {
            "description": "Open marketplace for AI Agents", 
            "purpose": "Publish and discover agents built with any framework",
            "registration_required": True,
            "url": "https://agentverse.ai",
            "features": ["Agent discovery", "Publishing marketplace", "Integration with ASI:One"]
        },
        "asione": {
            "description": "World's first agentic LLM and discovery layer",
            "purpose": "Routes user queries to appropriate agents for execution",
            "integration_required": True,
            "api_url": "https://api.asi1.ai/v1/chat/completions",
            "quickstart_guide": "https://innovationlab.fetch.ai/resources/docs/asione/asi-one-quickstart",
            "api_key_required": True
        }
    },

    "requirements": {
        "mandatory": [
            "All agents must be categorized under Innovation Lab",
            "Include badge: ![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)",
            "Include badge: ![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)",
            "Public GitHub repository required",
            "README.md with agent names and addresses",
            "Demo video (3-5 minutes)"
        ],
        "technical": {
            "chat_protocol": "Must implement Chat Protocol for ASI:One discoverability",
            "agentverse_registration": "Agents must be registered on Agentverse",
            "manifest_publishing": "Use publish_manifest=True when including chat protocol"
        },
        "submission": {
            "platform": "Must submit to Devpost with GitHub link",
            "documentation": "README.md must include agent details and addresses",
            "demo": "3-5 minute demo video showing functionality"
        }
    },

    "code_examples": {
        "chat_protocol_agent": """from datetime import datetime
from uuid import uuid4
from uagents import Agent, Context, Protocol
from uagents.setup import fund_agent_if_low
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)

agent = Agent()
chat_proto = Protocol(spec=chat_protocol_spec)

def create_text_chat(text: str) -> ChatMessage:
    content = [TextContent(type="text", text=text)]
    return ChatMessage(
        timestamp=datetime.utcnow(),
        msg_id=uuid4(),
        content=content,
    )

@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    # Always send acknowledgement
    await ctx.send(sender, ChatAcknowledgement(
        timestamp=datetime.utcnow(), 
        acknowledged_msg_id=msg.msg_id
    ))
    
    for item in msg.content:
        if isinstance(item, StartSessionContent):
            ctx.logger.info(f"Session started with {sender}")
        elif isinstance(item, TextContent):
            ctx.logger.info(f"Message: {item.text}")
            # Add your agent logic here
            response = create_text_chat("Hello from Agent")
            await ctx.send(sender, response)
        elif isinstance(item, EndSessionContent):
            ctx.logger.info(f"Session ended with {sender}")

@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"Message acknowledged by {sender}")

# This makes the agent discoverable by ASI:One
agent.include(chat_proto, publish_manifest=True)

if __name__ == "__main__": 
    agent.run()""",

        "weather_agent_complete": """# Complete Weather Agent Example - agents.py
from uagents import Agent, Context, Protocol, Model
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
    StartSessionContent,
    EndSessionContent,
)
from datetime import datetime
from uuid import uuid4
from typing import Any, Dict
import requests

class StructuredOutputPrompt(Model):
    prompt: str
    output_schema: Dict[str, Any]

class StructuredOutputResponse(Model):
    output: Dict[str, Any]

class WeatherRequest(Model):
    location: str

class WeatherResponse(Model):
    weather: str

def get_weather(location: str):
    \"\"\"Return current weather for a location string.\"\"\"
    if not location or not location.strip():
        raise ValueError("location is required")
    
    # Geocode the location
    geo_params = {"name": location, "count": 1, "language": "en", "format": "json"}
    gr = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params=geo_params, timeout=60
    )
    gr.raise_for_status()
    g = gr.json()
    
    if not g.get("results"):
        raise RuntimeError(f"No geocoding match for: {location}")
    
    r0 = g["results"][0]
    latitude = r0["latitude"]
    longitude = r0["longitude"]
    display = ", ".join([v for v in [r0.get("name"), r0.get("admin1"), r0.get("country")] if v])
    
    # Get current weather
    wx_params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m"
    }
    wr = requests.get("https://api.open-meteo.com/v1/forecast", params=wx_params, timeout=60)
    wr.raise_for_status()
    data = wr.json()
    current = data.get("current", {})
    
    temp = current.get("temperature_2m")
    app = current.get("apparent_temperature") 
    wind = current.get("wind_speed_10m")
    rh = current.get("relative_humidity_2m")
    
    parts = [f"Weather for {display}"]
    if temp is not None: parts.append(f"temp {temp}°C")
    if app is not None: parts.append(f"feels like {app}°C")
    if rh is not None: parts.append(f"RH {rh}%")
    if wind is not None: parts.append(f"wind {wind} km/h")
    
    return {"weather": ", ".join(parts)}

AI_AGENT_ADDRESS = "agent1qtlpfshtlcxekgrfcpmv7m9zpajuwu7d5jfyachvpa4u3dkt6k0uwwp2lct"

agent = Agent()
chat_proto = Protocol(spec=chat_protocol_spec)
struct_output_client_proto = Protocol(name="StructuredOutputClientProtocol", version="0.1.0")

def create_text_chat(text: str, end_session: bool = False) -> ChatMessage:
    content = [TextContent(type="text", text=text)]
    if end_session:
        content.append(EndSessionContent(type="end-session"))
    return ChatMessage(timestamp=datetime.utcnow(), msg_id=uuid4(), content=content)

@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    ctx.storage.set(str(ctx.session), sender)
    await ctx.send(sender, ChatAcknowledgement(
        timestamp=datetime.utcnow(), acknowledged_msg_id=msg.msg_id
    ))

    for item in msg.content:
        if isinstance(item, StartSessionContent):
            ctx.logger.info(f"Session started with {sender}")
        elif isinstance(item, TextContent):
            ctx.logger.info(f"Processing: {item.text}")
            # Send to AI agent for structured output
            await ctx.send(AI_AGENT_ADDRESS, StructuredOutputPrompt(
                prompt=item.text, output_schema=WeatherRequest.schema()
            ))

@struct_output_client_proto.on_message(StructuredOutputResponse)
async def handle_structured_output_response(ctx: Context, sender: str, msg: StructuredOutputResponse):
    session_sender = ctx.storage.get(str(ctx.session))
    if not session_sender:
        return

    try:
        location = msg.output.get("location") if isinstance(msg.output, dict) else None
        if not location or "<UNKNOWN>" in str(msg.output):
            raise ValueError("No valid location found")
            
        weather = get_weather(location)
        reply = weather.get("weather", f"Weather for {location}: (no data)")
        await ctx.send(session_sender, create_text_chat(reply))
        
    except Exception as err:
        ctx.logger.error(f"Error: {err}")
        await ctx.send(session_sender, create_text_chat(
            "Sorry, I couldn't process your request. Please try again."
        ))

agent.include(chat_proto, publish_manifest=True)
agent.include(struct_output_client_proto, publish_manifest=True)

if __name__ == "__main__":
    agent.run()""",

        "agent_storage": """# Store data persistently
ctx.storage.set("user_preferences", {"theme": "dark", "language": "python"})
ctx.storage.set("query_count", 42)

# Retrieve stored data
preferences = ctx.storage.get("user_preferences") or {}
count = ctx.storage.get("query_count") or 0

# Initialize on startup
@agent.on_event("startup")
async def initialize(ctx: Context):
    ctx.storage.set("initialized", True)
    ctx.logger.info("Agent initialized!")"""
    },

    "resources": {
        "documentation": {
            "innovation_lab_main": "https://innovationlab.fetch.ai/projects",
            "uagent_creation": "https://innovationlab.fetch.ai/resources/docs/agent-creation/uagent-creation", 
            "agent_communication": "https://innovationlab.fetch.ai/resources/docs/agent-communication/uagent-uagent-communication",
            "asi_compatible": "https://innovationlab.fetch.ai/resources/docs/examples/chat-protocol/asi-compatible-uagents",
            "asi_one_quickstart": "https://innovationlab.fetch.ai/resources/docs/asione/asi-one-quickstart",
            "chat_protocol_tutorial": "https://docs.asi1.ai/documentation/tutorials/agent-chat-protocol",
            "official_docs": "https://docs.fetch.ai/"
        },
        "github_repos": {
            "innovation_lab_examples": "https://github.com/fetchai/innovation-lab-examples",
            "uagents_main": "https://github.com/fetchai/uAgents",
            "uagents_examples": "https://github.com/fetchai/uAgents/tree/main/examples"
        },
        "api_keys": {
            "asi_one": {
                "description": "Required for ASI:One API integration",
                "how_to_get": "Follow the ASI:One quickstart guide",
                "environment_variable": "ASI_ONE_API_KEY"
            }
        }
    },

    "troubleshooting": {
        "agent_not_discoverable": {
            "symptoms": ["Agent not showing up in ASI:One", "Cannot find agent on Agentverse"],
            "solutions": [
                "Ensure chat protocol is included with publish_manifest=True",
                "Check that agent is registered on Agentverse", 
                "Verify agent is running and accessible at specified endpoint",
                "Confirm ASI:One integration is properly implemented",
                "Check Innovation Lab and Hackathon badges are in README"
            ]
        },
        "chat_protocol_errors": {
            "symptoms": ["Messages not being received", "Chat protocol not working"],
            "solutions": [
                "Import all required components from uagents_core.contrib.protocols.chat",
                "Implement both ChatMessage and ChatAcknowledgement handlers",
                "Always send acknowledgements for received messages",
                "Use proper timestamp and message ID formats"
            ]
        },
        "agentverse_registration": {
            "symptoms": ["Cannot register on Agentverse", "Agent not publishing"],
            "solutions": [
                "Ensure agent has proper endpoint configuration",
                "Check network connectivity and firewall settings", 
                "Verify agent manifest is properly published",
                "Confirm required badges are included in README.md"
            ]
        },
        "api_key_issues": {
            "symptoms": ["API authentication errors", "Cannot access ASI:One"],
            "solutions": [
                "Set ASI_ONE_API_KEY environment variable",
                "Follow ASI:One quickstart guide for API key setup",
                "Check API key permissions and validity",
                "Ensure proper API endpoint usage"
            ]
        }
    },

    "inspiration": {
        "categories": {
            "productivity": "Tools that make daily tasks faster and smoother. Automations for schoolwork, small businesses, or workflows like CRM updates, email handling, social media coordination.",
            "finance": "Agents that improve personal or corporate finances. Expense trackers, credit assessment, portfolio optimization, anything that helps save, invest, or manage money.",
            "education": "Agents that help people learn, stay updated, understand complex topics. Interactive study aids, AI explainers, research companions.",
            "wildcard": "Creative ideas that don't fit the above categories. As long as it uses the Fetch.ai stack and delivers real value."
        },
        "past_projects": {
            "description": "Browse past hackathon winning projects for inspiration",
            "link": "https://innovationlab.fetch.ai/projects",
            "note": "Look for innovative uses of agent coordination, real-world problem solving, and creative ASI:One integrations"
        }
    },

    "submission_checklist": {
        "pre_submission": [
            "Agent registered on Agentverse",
            "Chat protocol implemented with publish_manifest=True", 
            "ASI:One integration working and tested",
            "All required badges in README.md",
            "Public GitHub repository created and accessible"
        ],
        "submission_items": [
            "Demo video recorded (3-5 minutes)",
            "README.md includes agent names and addresses", 
            "Code is well-documented and functional",
            "Devpost submission completed with GitHub link"
        ],
        "testing": [
            "Agent responds through ASI:One chat interface",
            "All features work as demonstrated",
            "Agent handles edge cases gracefully",
            "Demo video shows clear value proposition"
        ]
    },

    "tips": {
        "winning_strategies": [
            "Focus on real-world problems that affect many people",
            "Demonstrate clear agent coordination and intelligence", 
            "Show innovative use of ASI:One for reasoning",
            "Create polished, professional demos",
            "Document your code thoroughly"
        ],
        "common_mistakes": [
            "Not implementing chat protocol correctly",
            "Forgetting to publish manifest to Agentverse",
            "Missing required badges in README",
            "Creating agents that don't solve real problems",
            "Poor demo presentation or explanation"
        ],
        "presentation": [
            "Start with the problem you're solving",
            "Show the agent in action with real scenarios",
            "Explain how ASI:One integration adds intelligence",
            "Demonstrate practical value for end users",
            "Keep demo under 5 minutes and well-paced"
        ]
    }
}

@agent.on_event("startup")
async def initialize_storage(ctx: Context):
    """Initialize storage on startup"""
    ctx.storage.set("mhacks_data", MHACKS_INFO)
    ctx.storage.set("query_count", 0)
    ctx.logger.info("✅ MHacks data initialized in storage")

def create_text_chat(text: str) -> ChatMessage:
    """Create a text chat message"""
    content = [TextContent(type="text", text=text)]
    return ChatMessage(
        timestamp=datetime.utcnow(),
        msg_id=uuid4(),
        content=content,
    )

async def classify_query_with_openai(user_query: str) -> list:
    """Use OpenAI to classify what categories of information the query needs"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return ["prizes", "tech_stack", "requirements"]
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    classification_prompt = f"""Analyze this user query and determine which MHacks Fetch.ai categories are relevant:

Available categories:
- prizes: Prize information, amounts, criteria, rewards
- judging_criteria: How projects are judged, scoring weights, evaluation
- tech_stack: Technical info about uAgents, Agentverse, ASI:One
- requirements: Submission requirements, badges, mandatory elements
- code_examples: Code templates, examples, implementation guides
- resources: Documentation links, GitHub repos, API guides
- troubleshooting: Common problems, debugging, error solutions
- inspiration: Project ideas, past winners, categories
- submission_checklist: Step-by-step submission process
- tips: Winning strategies, common mistakes, presentation advice

User query: "{user_query}"

Return ONLY a JSON array like: ["prizes", "tech_stack"]"""

    body = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": classification_prompt}],
        "max_tokens": 30,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(url, headers=headers, json=body)
        result = response.json()
        categories = json.loads(result["choices"][0]["message"]["content"].strip())
        return categories
    except:
        return ["prizes", "tech_stack", "requirements"]

def get_relevant_data(ctx: Context, categories: list) -> dict:
    """Get relevant data based on categories"""
    mhacks_data = ctx.storage.get("mhacks_data") or MHACKS_INFO
    relevant_data = {}
    
    for category in categories:
        if category in mhacks_data:
            relevant_data[category] = mhacks_data[category]
    
    return relevant_data

async def query_openai(user_query: str, relevant_data: dict) -> str:
    """Query OpenAI with relevant context"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    system_prompt = f"""You are a helpful support agent for MHacks participants working with Fetch.ai technology.

MHacks Fetch.ai Information:
{json.dumps(relevant_data, indent=2)}

Instructions: Be helpful and specific. Use emojis. Include code examples when relevant. Focus on practical advice for hackathon participants."""

    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    response = requests.post(url, headers=headers, json=body)
    result = response.json()
    return result["choices"][0]["message"]["content"]

# Handle incoming chat messages
@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    # Send acknowledgement
    await ctx.send(sender, ChatAcknowledgement(
        timestamp=datetime.utcnow(), 
        acknowledged_msg_id=msg.msg_id
    ))
    
    # Check if this message has both StartSessionContent and TextContent
    has_start_session = any(isinstance(item, StartSessionContent) for item in msg.content)
    has_text_content = any(isinstance(item, TextContent) for item in msg.content)
    
    # Process each content item
    for item in msg.content:
        if isinstance(item, StartSessionContent) and not has_text_content:
            welcome_response = create_text_chat(
                """👋 **Welcome to MHacks Fetch.ai Support!**

🏆 **Prizes** - $1250, $750, and $500 available
💻 **Code Examples** - uAgents templates and integration  
📋 **Requirements** - Submission guidelines and badges
🔧 **Tech Stack** - uAgents, Agentverse, ASI:One help

**What would you like to know?** 🚀"""
            )
            await ctx.send(sender, welcome_response)
            
        elif isinstance(item, TextContent):
            # Classify query and get relevant data
            categories = await classify_query_with_openai(item.text)
            relevant_data = get_relevant_data(ctx, categories)
            
            # Update query count
            count = ctx.storage.get("query_count") or 0
            ctx.storage.set("query_count", count + 1)
            
            # Query OpenAI
            if has_start_session:
                enhanced_query = f"This is the user's first message. Include a brief welcome with: {item.text}"
                response_text = await query_openai(enhanced_query, relevant_data)
            else:
                response_text = await query_openai(item.text, relevant_data)
                
            response_message = create_text_chat(response_text)
            await ctx.send(sender, response_message)
            
        elif isinstance(item, EndSessionContent):
            goodbye_response = create_text_chat(
                """🚀 **Good luck with your MHacks project!**

✅ Register agents on Agentverse  
✅ Implement Chat Protocol
✅ Integrate ASI:One
✅ Submit to Devpost with demo video

**You've got this!** 🌟"""
            )
            await ctx.send(sender, goodbye_response)

# Handle acknowledgements  
@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass

# Include the chat protocol and publish manifest
agent.include(chat_proto, publish_manifest=True)

if __name__ == "__main__":
    agent.run()
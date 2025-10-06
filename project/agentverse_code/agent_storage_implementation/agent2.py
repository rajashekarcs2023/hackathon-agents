# Agent address : agent1q0f9qrvtn9h6njyjv0t5x4pkr459mew9mgwg89kam36gs299d0lmqdrvepu

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
# Complete MHACKS_INFO with Agentverse MCP Server fix
MHACKS_INFO = {
    "agentverse_mcp_server": {
        "description": "Agentverse MCP Server allows you to use the Agentverse API in your MCP clients",
        "overview": "Model Context Protocol (MCP) is an open standard that lets AI systems interact with external data sources and tools over a secure, two-way channel. Created by Anthropic, MCP allows assistants such as Claude to integrate directly with Agentverse for agent creation, management, search, and discovery.",
        "compatible_clients": ["Cursor", "Claude Desktop", "Claude Code", "OpenAI Playground", "Cline"],
        
        "remote_servers": {
            "main_server": {
                "url": "https://mcp.agentverse.ai/sse",
                "description": "Production server with the full Agentverse toolset for advanced workflows (storage, quotas, analytics, and more)"
            },
            "lite_server": {
                "url": "https://mcp-lite.agentverse.ai/mcp", 
                "description": "Minimal server exposing core tools to create, update, start/stop agents, and search marketplace. Optimized for clients with tool-count limits"
            }
        },
        
        "features": [
            "Agents & Hosting API: Create/update agents, upload code (JSON array), start/stop, and fetch details/logs",
            "Marketplace Search API: Search and discover agents; fetch public agent profiles", 
            "Storage API: Get/set/delete per-agent key-value storage for lightweight state",
            "Secrets API (User-level): List/create/delete user secrets available to your agents",
            "Almanac API (Main MCP): Register and query agents on the on-chain Almanac (network-aware ops)",
            "Mailbox API (Main MCP): Manage mailboxes, quotas, and message metrics for Local/Mailbox agents",
            "Service & Health (Main MCP): Health checks and transport endpoints (SSE/HTTP) for diagnostics"
        ],
        
        "setup_instructions": {
            "cursor": {
                "steps": [
                    "Open Cursor Settings, and go to Tools and Integrations tab",
                    "Click + New MCP Server to open your mcp.json",
                    "Paste configuration into mcp.json with your AGENTVERSE_API_TOKEN",
                    "Save file and go back to Tools and Integrations tab"
                ],
                "config": {
                    "mcpServers": {
                        "agentverse-lite": {
                            "type": "http",
                            "url": "https://mcp-lite.agentverse.ai/mcp",
                            "timeout": 180000,
                            "env": {
                                "AGENTVERSE_API_TOKEN": "Your Agentverse API Token"
                            }
                        }
                    }
                }
            },
            "claude_desktop": {
                "steps": [
                    "Open Claude Desktop, and go to Settings tab",
                    "Click on Connectors and add Custom Connector",
                    "Enter details for Agentverse MCP server",
                    "Restart Claude Desktop"
                ]
            },
            "openai_playground": {
                "steps": [
                    "Open https://platform.openai.com/playground",
                    "Click + Create and in tools click +add and select MCP Server",
                    "Click +Server and fill in Agentverse MCP details",
                    "Start chatting with the Agentverse MCP"
                ]
            }
        },
        
        "api_token_required": True,
        "token_source": "Create one in Agentverse API Keys (https://agentverse.ai/profile/api-keys)",
        
        "rules_file": {
            "name": "av-mcp.mdc",
            "description": "Initialization prompt and Vibe Coding rules for Agentverse MCP",
            "download_url": "https://asset.cloudinary.com/fetch-ai/5794112484d9e37b472c2000840ca33f",
            "features": [
                "Protocol correctness: Enforces Agent Chat Protocol invariants, strict ACK rhythm, and session/stream semantics",
                "Version targeting: Aligns with latest uagents behavior", 
                "Scaffolds and layouts: Standard hosted/ and mailbox_or_local/ structures with README guidance",
                "Hosted allowlist: Curated imports for Hosted agents; recommend Mailbox/Local when deps aren't supported",
                "Storage & media: ExternalStorage usage patterns; image analysis/generation; tmp URL staging for video/audio",
                "Rate limits & ACL: QuotaProtocol examples with per-sender quotas and optional ACL bypass",
                "MCP reliability: Create/update/start, JSON-stringified code uploads, retry guidance, raw JWT token requirement",
                "Secrets policy: Hosted ignores repo .env; configure secrets in Agentverse. Mailbox/Local require AGENT_NAME, AGENT_SEED, PORT/AGENT_PORT",
                "Templates & checklists: Ready-to-run examples and a final preflight checklist plus required README badges"
            ]
        },
        
        "usage_example": "Create a uAgent Hosted on Agentverse using the available AV tools and rules. Make it behave as an expert assistant on Supercars. Also create a separate .env with ASI_ONE_API_KEY and save it in the agent project with update agent code. The agent must use the Agent Chat Protocol so it's ASI1-compatible.",
        
        "example_workflow": [
            "Use Cursor with AV MCP Rules to create agents",
            "Agent gets created and hosted on Agentverse automatically", 
            "Chat with agent immediately in Agentverse",
            "Test agent in ASI:One (Agentic mode) for end-to-end verification"
        ]
    },

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
        },
        "mcp": {
            "description": "Model Context Protocol - open standard for AI systems to interact with external tools",
            "purpose": "Enables agents to access real-world data, external APIs, and tools in a standardized way",
            "created_by": "Anthropic",
            "note": "This section covers MCP integration approaches. For direct Agentverse MCP Server usage, see agentverse_mcp_server section",
            "key_features": [
                "Standardized tool and service access protocol",
                "Dynamic tool discovery at runtime",
                "Multiple transport methods (stdio, SSE, custom)",
                "Type safety with tool schemas",
                "Eliminates need for custom integration logic"
            ],
            "benefits_with_uagents": [
                "Standardized access to external capabilities",
                "Dynamic discovery and orchestration via ASI:One",
                "Plug-and-play extensibility",
                "Ecosystem growth and modularity",
                "Agent-to-agent tool sharing"
            ],
            "integration_approaches": {
                "langgraph_mcp_adapter": {
                    "description": "LangGraph agent uses langchain_mcp_adapters to connect to MCP servers",
                    "process": "LangGraph → MCP tools → uagents_adapter → Agentverse registration → ASI:One discovery",
                    "use_case": "Leverage LangGraph workflows while exposing as agentic services"
                },
                "remote_mcp_servers": {
                    "description": "uAgent client connects directly to remote MCP servers (e.g., Smithery.ai)",
                    "process": "uAgent bridge → Remote MCP servers → Agentverse registration → ASI:One access",
                    "use_case": "Quickly expose existing remote MCP services to agent ecosystem"
                },
                "mcp_server_on_agentverse": {
                    "description": "FastMCP server wrapped as uAgent using MCPServerAdapter",
                    "process": "FastMCP server → MCPServerAdapter → uAgent → Agentverse registration",
                    "use_case": "Deploy Python-based FastMCP servers with minimal integration effort"
                }
            },
            "common_use_cases": [
                "Connecting research agents to medical databases (PubMed, clinical trials)",
                "Enabling travel assistants to access real-time listings (Airbnb)",
                "Allowing productivity agents to interact with calendars, emails, web search",
                "Financial agents accessing market data and trading APIs",
                "Healthcare agents connecting to medical calculators and databases"
            ],
            "agentverse_integration": {
                "main_server": "https://mcp.agentverse.ai/sse",
                "lite_server": "https://mcp-lite.agentverse.ai/mcp",
                "differences": {
                    "main": "Full Agentverse toolset for advanced workflows (storage, quotas, analytics)",
                    "lite": "Minimal server optimized for clients with tool-count limits"
                }
            },
            "compatible_clients": ["Cursor", "Claude Desktop", "Claude Code", "OpenAI Playground", "Cline"],
            "required_token": "AGENTVERSE_API_TOKEN"
        },
        "adapters": {
            "uagents_adapter": {
                "description": "Package to integrate existing LangChain/LangGraph/CrewAI agents with uAgents ecosystem",
                "installation": {
                    "base": "pip install uagents-adapter",
                    "langchain_support": "pip install \"uagents-adapter[langchain]\"",
                    "crewai_support": "pip install \"uagents-adapter[crewai]\"",
                    "all_frameworks": "pip install \"uagents-adapter[langchain,crewai]\""
                },
                "purpose": "Wrap existing AI frameworks as uAgents for Agentverse compatibility",
                "supported_frameworks": ["LangChain", "LangGraph", "CrewAI"],
                "benefits": [
                    "Leverage existing AI agent frameworks",
                    "Advanced orchestration capabilities", 
                    "Multi-agent collaboration (CrewAI)",
                    "Access to extensive tool ecosystems",
                    "Seamless Agentverse integration"
                ],
                "required_apis": ["OPENAI_API_KEY", "AGENTVERSE_API_KEY", "Framework-specific APIs"],
                "use_cases": {
                    "langchain": "Single sophisticated agents with tool chains",
                    "langgraph": "Complex reasoning flows and state management",
                    "crewai": "Multi-agent collaboration and specialized teams"
                }
            }
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

        "langgraph_adapter": """# LangGraph Adapter Example - agent.py
import os
import time
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import chat_agent_executor
from langchain_core.messages import HumanMessage

from uagents_adapter import LangchainRegisterTool, cleanup_uagent

# Load environment variables
load_dotenv()

# Required API keys
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
API_TOKEN = os.environ["AGENTVERSE_API_KEY"]

if not API_TOKEN:
    raise ValueError("Please set AGENTVERSE_API_KEY environment variable")

# Set up tools and LLM
tools = [TavilySearchResults(max_results=3)]
model = ChatOpenAI(temperature=0)

# Create LangGraph-based executor
app = chat_agent_executor.create_tool_calling_executor(model, tools)

# Wrap LangGraph agent into a function for uAgent
def langgraph_agent_func(query):
    # Handle input if it's a dict with 'input' key
    if isinstance(query, dict) and 'input' in query:
        query = query['input']
    
    messages = {"messages": [HumanMessage(content=query)]}
    final = None
    for output in app.stream(messages):
        final = list(output.values())[0]  # Get latest
    return final["messages"][-1].content if final else "No response"

# Register the LangGraph agent via uAgent
tool = LangchainRegisterTool()
agent_info = tool.invoke(
    {
        "agent_obj": langgraph_agent_func,
        "name": "langgraph_tavily_agent",
        "port": 8080,
        "description": "A LangGraph-based Tavily-powered search agent",
        "api_token": API_TOKEN,
        "mailbox": True
    }
)

print(f"✅ Registered LangGraph agent: {agent_info}")

# Keep the agent alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("🛑 Shutting down LangGraph agent...")
    cleanup_uagent("langgraph_tavily_agent")
    print("✅ Agent stopped.")

# Requirements for this example:
# pip install uagents-adapter[langchain] langchain-openai langchain-community 
# pip install langgraph python-dotenv""",

        "crewai_adapter": """# CrewAI Adapter Example - Trip Planner
import os
from crewai import Crew
from dotenv import load_dotenv
from uagents_adapter import CrewaiRegisterTool
from trip_agents import TripAgents
from trip_tasks import TripTasks

class TripCrew:
    def __init__(self, origin, cities, date_range, interests):
        self.cities = cities
        self.origin = origin
        self.interests = interests
        self.date_range = date_range

    def run(self):
        agents = TripAgents()
        tasks = TripTasks()

        # Create specialized agents
        city_selector_agent = agents.city_selection_agent()
        local_expert_agent = agents.local_expert()
        travel_concierge_agent = agents.travel_concierge()

        # Define tasks for each agent
        identify_task = tasks.identify_task(city_selector_agent, self.origin, self.cities, self.interests, self.date_range)
        gather_task = tasks.gather_task(local_expert_agent, self.origin, self.interests, self.date_range)
        plan_task = tasks.plan_task(travel_concierge_agent, self.origin, self.interests, self.date_range)

        # Create and run the crew
        crew = Crew(
            agents=[city_selector_agent, local_expert_agent, travel_concierge_agent],
            tasks=[identify_task, gather_task, plan_task],
            verbose=True,
        )

        result = crew.kickoff()
        return result

    def kickoff(self, inputs=None):
        \"\"\"Compatibility method for uAgents integration.\"\"\"
        if inputs:
            self.origin = inputs.get("origin", self.origin)
            self.cities = inputs.get("cities", self.cities)
            self.date_range = inputs.get("date_range", self.date_range)
            self.interests = inputs.get("interests", self.interests)
        return self.run()

def main():
    load_dotenv()
    api_key = os.getenv("AGENTVERSE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key or not openai_api_key:
        print("Error: Missing required API keys")
        return

    # Create trip crew instance
    trip_crew = TripCrew("", "", "", "")

    # Register with uAgents
    register_tool = CrewaiRegisterTool()
    
    query_params = {
        "origin": {"type": "str", "required": True},
        "cities": {"type": "str", "required": True}, 
        "date_range": {"type": "str", "required": True},
        "interests": {"type": "str", "required": True},
    }

    result = register_tool.run(
        tool_input={
            "crew_obj": trip_crew,
            "name": "Trip Planner Crew AI Agent",
            "port": 8080,
            "description": "A CrewAI agent that helps plan trips based on preferences",
            "api_token": api_key,
            "mailbox": True,
            "query_params": query_params,
            "example_query": "Plan a trip from New York to Paris in June, interested in art and history.",
        }
    )

    print(f"CrewAI agent registered: {result}")
    
    # Keep running
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    main()

# Installation:
# pip install "uagents-adapter[crewai]" crewai[tools] python-dotenv
# 
# Environment variables needed:
# OPENAI_API_KEY=your_openai_key
# AGENTVERSE_API_KEY=your_agentverse_key""",

        "mcp_langgraph_integration": """# MCP + LangGraph Agent Integration
# LangGraph agent with MCP tools, wrapped as uAgent

from langgraph.prebuilt import chat_agent_executor
from langchain_openai import ChatOpenAI
from langchain_mcp import MCPTool
from uagents_adapter import LangchainRegisterTool
import os

# Setup LangGraph with MCP tools
def create_mcp_langgraph_agent():
    # Connect to MCP servers
    mcp_tools = [
        MCPTool(server_config={
            "command": "uvx",
            "args": ["mcp-server-pubmed"],
            "env": {}
        }),
        MCPTool(server_config={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-brave-search"],
            "env": {"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY")}
        })
    ]
    
    # Create LangGraph agent with MCP tools
    model = ChatOpenAI(temperature=0)
    agent = chat_agent_executor.create_tool_calling_executor(model, mcp_tools)
    
    return agent

# Wrap as uAgent function
def langgraph_mcp_func(query):
    agent = create_mcp_langgraph_agent()
    messages = {"messages": [{"role": "user", "content": query}]}
    
    result = None
    for output in agent.stream(messages):
        result = list(output.values())[0]
    
    return result["messages"][-1].content if result else "No response"

# Register with uAgents
tool = LangchainRegisterTool()
agent_info = tool.invoke({
    "agent_obj": langgraph_mcp_func,
    "name": "research_mcp_agent",
    "port": 8080,
    "description": "Research agent with access to PubMed and web search via MCP",
    "api_token": os.getenv("AGENTVERSE_API_KEY"),
    "mailbox": True
})

# Benefits:
# - Access to multiple MCP tools in one agent
# - LangGraph workflow capabilities
# - Discoverable via ASI:One
# - Tool selection handled by LangGraph""",

        "mcp_remote_servers": """# Connect uAgent to Remote MCP Servers
# Direct connection to Smithery.ai and other remote MCP servers

from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import *
import requests
import json

agent = Agent()
chat_proto = Protocol(spec=chat_protocol_spec)

# MCP client for remote servers
class MCPClient:
    def __init__(self, server_url):
        self.server_url = server_url
        self.session = requests.Session()
    
    async def call_tool(self, tool_name, arguments):
        payload = {
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        response = self.session.post(f"{self.server_url}/mcp", json=payload)
        return response.json()
    
    async def list_tools(self):
        payload = {"method": "tools/list"}
        response = self.session.post(f"{self.server_url}/mcp", json=payload)
        return response.json()

# Connect to multiple MCP servers
mcp_clients = {
    "pubmed": MCPClient("https://pubmed.smithery.ai"),
    "calculator": MCPClient("https://calculator.smithery.ai"),
    "weather": MCPClient("https://weather.smithery.ai")
}

@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(sender, ChatAcknowledgement(
        timestamp=datetime.utcnow(), acknowledged_msg_id=msg.msg_id
    ))
    
    for item in msg.content:
        if isinstance(item, TextContent):
            query = item.text.lower()
            
            # Route to appropriate MCP server based on query
            if "research" in query or "pubmed" in query:
                result = await mcp_clients["pubmed"].call_tool("search", {"query": item.text})
            elif "calculate" in query or "math" in query:
                result = await mcp_clients["calculator"].call_tool("calculate", {"expression": item.text})
            elif "weather" in query:
                result = await mcp_clients["weather"].call_tool("current_weather", {"location": item.text})
            else:
                result = {"content": "I can help with research, calculations, or weather. What would you like?"}
            
            response = create_text_chat(str(result.get("content", "No result")))
            await ctx.send(sender, response)

agent.include(chat_proto, publish_manifest=True)

# Benefits:
# - Access to existing remote MCP services
# - No need to host MCP servers
# - Quick integration with Smithery.ai ecosystem
# - Multiple tool access in one agent""",

        "mcp_server_agentverse": """# Create MCP Server on Agentverse using MCPServerAdapter
# Deploy FastMCP server as uAgent

from fastmcp import FastMCP
from uagents_adapter import MCPServerAdapter
import os

# Create FastMCP server
mcp = FastMCP("MHacks Calculator")

@mcp.tool()
def calculate(expression: str) -> str:
    \"\"\"Calculate mathematical expressions safely.\"\"\"
    try:
        # Safe evaluation of mathematical expressions
        allowed_chars = set('0123456789+-*/().,')
        if not all(c in allowed_chars for c in expression.replace(' ', '')):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    \"\"\"Convert temperature between Celsius, Fahrenheit, and Kelvin.\"\"\"
    # Conversion logic
    if from_unit.lower() == "celsius":
        if to_unit.lower() == "fahrenheit":
            result = (value * 9/5) + 32
        elif to_unit.lower() == "kelvin":
            result = value + 273.15
        else:
            result = value
    elif from_unit.lower() == "fahrenheit":
        if to_unit.lower() == "celsius":
            result = (value - 32) * 5/9
        elif to_unit.lower() == "kelvin":
            result = (value - 32) * 5/9 + 273.15
        else:
            result = value
    else:  # Kelvin
        if to_unit.lower() == "celsius":
            result = value - 273.15
        elif to_unit.lower() == "fahrenheit":
            result = (value - 273.15) * 9/5 + 32
        else:
            result = value
    
    return f"{value}° {from_unit} = {result}° {to_unit}"

# Create and register MCP server as uAgent
def main():
    api_token = os.getenv("AGENTVERSE_API_KEY")
    
    adapter = MCPServerAdapter()
    result = adapter.register_server(
        mcp_server=mcp,
        name="mhacks_calculator_mcp",
        description="Calculator and temperature converter via MCP",
        port=8080,
        api_token=api_token,
        mailbox=True
    )
    
    print(f"MCP Server registered: {result}")
    
    # Keep server running
    mcp.run(port=8080)

if __name__ == "__main__":
    main()

# Benefits:
# - FastMCP server simplicity
# - Automatic tool schema generation
# - ASI:One LLM handles tool selection
# - Minimal integration effort""",

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
        "image_analysis_agent": "https://innovationlab.fetch.ai/resources/docs/examples/chat-protocol/image-analysis-agent",
        "image_generation_agent": "https://innovationlab.fetch.ai/resources/docs/examples/chat-protocol/image-generation-agent",
        "asi_one_quickstart": "https://innovationlab.fetch.ai/resources/docs/asione/asi-one-quickstart",
        "chat_protocol_tutorial": "https://docs.asi1.ai/documentation/tutorials/agent-chat-protocol",
        "langgraph_adapter": "https://innovationlab.fetch.ai/resources/docs/examples/adapters/langgraph-adapter-example",
        "crewai_adapter": "https://innovationlab.fetch.ai/resources/docs/examples/adapters/crewai-adapter-example",
        "agentverse_mcp": "https://docs.agentverse.ai/documentation/advanced-usages/agentverse-mcp",
        "mcp_integration": "https://innovationlab.fetch.ai/resources/docs/examples/mcp-integration/",
        "official_docs": "https://docs.fetch.ai/"
    },
    
    # NEW: Specific link guidance for different query types
    "link_guidance": {
        "basic_uagent_questions": {
            "description": "How to create uAgents, basic setup, first agent",
            "primary_link": "https://innovationlab.fetch.ai/resources/docs/agent-creation/uagent-creation",
            "content_overview": "Complete guide on creating uAgents with hosted, local, and mailbox options"
        },
        "agent_communication": {
            "description": "How agents communicate, connect to frontend, agent-to-agent communication",
            "primary_link": "https://innovationlab.fetch.ai/resources/docs/agent-communication/uagent-uagent-communication"
        },
        "asi_compatible": {
            "description": "ASI:One compatible agents, chat protocol implementation",
            "primary_link": "https://innovationlab.fetch.ai/resources/docs/examples/chat-protocol/asi-compatible-uagents"
        },
        "image_analysis": {
            "description": "Image analysis agents, computer vision with uAgents",
            "primary_link": "https://innovationlab.fetch.ai/resources/docs/examples/chat-protocol/image-analysis-agent"
        },
        "image_generation": {
            "description": "Image generation agents, AI image creation with uAgents",
            "primary_link": "https://innovationlab.fetch.ai/resources/docs/examples/chat-protocol/image-generation-agent"
        }
    },
    
    "github_repos": {
        "innovation_lab_examples": "https://github.com/fetchai/innovation-lab-examples",
        "uagents_main": "https://github.com/fetchai/uAgents",
        "uagents_examples": "https://github.com/fetchai/uAgents/tree/main/examples",
        "mcp_agents_examples": "https://github.com/fetchai/innovation-lab-examples/tree/main/mcp-agents"
    },
        "mcp_integration": {
            "description": "Model Context Protocol integration with uAgents for external tool access",
            "integration_approaches": {
                "langgraph_mcp": "LangGraph agent with MCP tools, wrapped as uAgent",
                "remote_mcp_servers": "uAgent connecting to remote MCP servers (Smithery.ai)",
                "mcp_server_agentverse": "FastMCP server deployed as uAgent using MCPServerAdapter"
            },
            "remote_servers": ["Smithery.ai PubMed", "Medical calculators", "Weather APIs", "Financial data"],
            "compatible_clients": ["Cursor", "Claude Desktop", "Claude Code", "OpenAI Playground", "Cline"],
            "required_dependencies": ["fastmcp", "langchain_mcp", "uagents_adapter"],
            "benefits": [
                "Standardized external tool access",
                "Dynamic tool discovery and orchestration",
                "Plug-and-play extensibility",
                "Agent-to-agent tool sharing",
                "Rich ecosystem growth"
            ]
        },
        "adapters": {
            "uagents_adapter": {
                "package_installation": {
                    "base": "pip install uagents-adapter",
                    "langchain_support": "pip install \"uagents-adapter[langchain]\"",
                    "crewai_support": "pip install \"uagents-adapter[crewai]\"",
                    "all_frameworks": "pip install \"uagents-adapter[langchain,crewai]\""
                },
                "framework_dependencies": {
                    "langchain": [
                        "langchain-openai",
                        "langchain-community", 
                        "langgraph"
                    ],
                    "crewai": [
                        "crewai[tools]",
                        "langchain_openai"
                    ],
                    "common": [
                        "uagents",
                        "python-dotenv"
                    ]
                },
                "environment_variables": [
                    "OPENAI_API_KEY=your_openai_key",
                    "AGENTVERSE_API_KEY=your_agentverse_key",
                    "TAVILY_API_KEY=your_tavily_key (for search agents)",
                    "AGENT_SEED=your_agent_seed_phrase (optional)"
                ],
                "use_cases": {
                    "langchain": "Single sophisticated agents with tool chains and complex reasoning",
                    "langgraph": "State management, multi-turn conversations, directed reasoning flows",
                    "crewai": "Multi-agent collaboration, specialized teams, complex task orchestration"
                },
                "examples": {
                    "langchain": "Search agents, document processors, API integrators",
                    "crewai": "Trip planners, research teams, content creation crews"
                }
            }
        },
        "api_keys": {
            "asi_one": {
                "description": "Required for ASI:One API integration",
                "how_to_get": "Follow the ASI:One quickstart guide",
                "environment_variable": "ASI_ONE_API_KEY"
            },
            "agentverse": {
                "description": "Required for LangGraph/CrewAI adapter integration and MCP",
                "how_to_get": "Get from Agentverse dashboard - Profile > API Keys",
                "environment_variable": "AGENTVERSE_API_KEY",
                "uses": ["Adapter integrations", "MCP server authentication", "Agent hosting"]
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
    
    # Minimal fallback ONLY for API unavailability (not for classification logic)
    def emergency_fallback(query: str) -> list:
        """Only used when API is completely unavailable"""
        return ["prizes", "tech_stack", "judging_criteria", "requirements"]  # Return multiple categories to be safe
    
    # If no OpenAI API key, use emergency fallback
    if not api_key:
        return emergency_fallback(user_query)
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # ROBUST LLM CLASSIFICATION PROMPT
    classification_prompt = f"""You are a classification expert. Analyze this MHacks Fetch.ai query and return the most relevant categories.

CATEGORIES WITH CLEAR EXAMPLES:

agentverse_mcp_server:
- Setting up MCP clients (Cursor, Claude Desktop, OpenAI Playground) to connect to Agentverse
- Examples: "how to use agentverse mcp", "setup cursor with agentverse", "claude desktop mcp", "mcp.json configuration"

code_examples: 
- User wants actual code implementations, templates, or examples
- Examples: "code example", "how to implement", "template", "langgraph adapter code", "crewai adapter", "chat protocol code"

judging_criteria:
- How projects are evaluated, scored, or judged
- Examples: "judging criteria", "how are projects judged", "scoring", "evaluation criteria", "weightage", "what do judges look for"

prizes:
- Prize amounts, descriptions, rewards
- Examples: "prizes", "how much money", "cash prizes", "rewards", "prize breakdown"

tech_stack:
- Technical information about frameworks and architecture
- Examples: "uagents", "agentverse", "asi:one", "technology stack", "mcp integration approaches"

requirements:
- Submission rules, mandatory elements, badges
- Examples: "requirements", "submission", "badges", "devpost", "mandatory elements"

resources:
- Documentation links, installation guides, GitHub repos
- Examples: "documentation", "how to install", "github", "resources", "guides"

troubleshooting:
- Problems, errors, debugging help
- Examples: "error", "not working", "problem", "debug", "issue", "troubleshoot"

CLASSIFICATION RULES:
1. Focus on USER INTENT, not just keywords
2. If query matches multiple categories, return the most relevant 1-2 categories
3. "judging criteria" queries should ALWAYS return ["judging_criteria"]
4. "agentverse mcp" queries should return ["agentverse_mcp_server"]
5. "code" or "example" queries should include ["code_examples"]

User query: "{user_query}"

Return ONLY a JSON array like: ["judging_criteria"] or ["code_examples", "tech_stack"]"""

    body = {
        "model": "gpt-4o-mini",  # Fixed: Changed from gpt-4o
        "messages": [{"role": "user", "content": classification_prompt}],
        "max_tokens": 50,        # Fixed: Changed from 1000
        "temperature": 0.1
    }
    
    try:
        response = requests.post(url, headers=headers, json=body, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        content = result["choices"][0]["message"]["content"].strip()
        
        # Parse JSON response
        try:
            categories = json.loads(content)
            if isinstance(categories, list) and all(isinstance(cat, str) for cat in categories):
                # Validate categories exist in our data
                valid_categories = ["agentverse_mcp_server", "prizes", "judging_criteria", "tech_stack", 
                                 "requirements", "code_examples", "resources", "troubleshooting", 
                                 "inspiration", "submission_checklist", "tips"]
                
                filtered_categories = [cat for cat in categories if cat in valid_categories]
                if filtered_categories:
                    return filtered_categories
                    
        except json.JSONDecodeError:
            pass
            
        # If JSON parsing fails, try to extract category names from response
        content_lower = content.lower()
        if "judging_criteria" in content_lower or "criteria" in content_lower:
            return ["judging_criteria"]
        if "agentverse_mcp" in content_lower:
            return ["agentverse_mcp_server"]
        if "code" in content_lower:
            return ["code_examples", "tech_stack"]
        if "prize" in content_lower:
            return ["prizes"]
            
        # Last resort - return multiple categories
        return ["prizes", "tech_stack", "judging_criteria"]
        
    except (requests.RequestException, KeyError, IndexError) as e:
        # Network/API error - use emergency fallback
        return emergency_fallback(user_query)

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
    
    system_prompt = f"""You are a helpful MHacks support agent. Use ONLY the provided data to answer questions.

MHacks Fetch.ai Information:
{json.dumps(relevant_data, indent=2)}

CRITICAL RULES - NO EXCEPTIONS:
1. NEVER provide links that are not in the data above
2. ONLY use links from the "resources" section provided
3. If asked for documentation, use ONLY these specific links:
   - Basic uAgent creation: https://innovationlab.fetch.ai/resources/docs/agent-creation/uagent-creation
   - Agent communication: https://innovationlab.fetch.ai/resources/docs/agent-communication/uagent-uagent-communication
   - ASI:One compatible agents: https://innovationlab.fetch.ai/resources/docs/examples/chat-protocol/asi-compatible-uagents
   - Image analysis agents: https://innovationlab.fetch.ai/resources/docs/examples/chat-protocol/image-analysis-agent
   - Image generation agents: https://innovationlab.fetch.ai/resources/docs/examples/chat-protocol/image-generation-agent

4. NEVER mention links like "fetch.ai/docs" or any links not in the provided data
5. Use exact code from "code_examples" without modification
6. For judging criteria: Use exact weights and descriptions provided
7. If information isn't in the data, say so and point to the innovation lab main page

SPECIFIC CODE EXAMPLE SELECTION:
- Basic LangChain agent deployment: Use "langgraph_adapter" example (Tavily search agent)
- LangChain + MCP integration: Use "mcp_langgraph_integration" example (PubMed + web search)
- CrewAI agent deployment on agentverse, or any crewai agent: Use "crewai_adapter" example (trip planner)
- Basic uAgent: Use "chat_protocol_agent" example

When users ask "langchain agent on agentverse" or similar basic questions, use the "langgraph_adapter" example, NOT the MCP integration version.

LINK USAGE RULES:
- "How to create uAgent" → Use uagent_creation link
- "Agent communication" → Use agent_communication link  
- "ASI:One agent" → Use asi_compatible link
- "Image analysis" → Use image_analysis_agent link
- "Image generation" → Use image_generation_agent link
- General questions → Use innovation_lab_main link

FORBIDDEN:
- Never invent or hallucinate documentation links
- Never use fetch.ai/docs URLs (they don't exist in this context)
- Never modify provided code examples
- Never make up API methods or classes"""

    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        "max_tokens": 1200,
        "temperature": 0.1  # Lower temperature to reduce hallucination
    }
    
    try:
        response = requests.post(url, headers=headers, json=body, timeout=15)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except (requests.RequestException, KeyError, IndexError) as e:
        return f"I'm experiencing technical difficulties. Please check the Innovation Lab documentation at https://innovationlab.fetch.ai/resources/docs/ for detailed information."

def get_relevant_data(ctx: Context, categories: list, user_query: str = "") -> dict:
    """Get relevant data based on categories with query-aware enhancements"""
    mhacks_data = ctx.storage.get("mhacks_data") or MHACKS_INFO
    relevant_data = {}
    
    # Add classified categories
    for category in categories:
        if category in mhacks_data:
            relevant_data[category] = mhacks_data[category]
    
    # Query-aware enhancements to catch missed classifications
    if user_query:
        query_lower = user_query.lower()
        
        # LangChain/LangGraph specific boost
        if ("langchain" in query_lower or "langgraph" in query_lower) and "code_examples" not in relevant_data:
            relevant_data["code_examples"] = mhacks_data.get("code_examples", {})
            relevant_data["tech_stack"] = mhacks_data.get("tech_stack", {})
            relevant_data["resources"] = mhacks_data.get("resources", {})
        
        # CrewAI specific boost  
        if "crewai" in query_lower and "code_examples" not in relevant_data:
            relevant_data["code_examples"] = mhacks_data.get("code_examples", {})
            relevant_data["tech_stack"] = mhacks_data.get("tech_stack", {})
        
        # MCP specific boost
        if "mcp" in query_lower and "agentverse_mcp_server" not in relevant_data:
            relevant_data["agentverse_mcp_server"] = mhacks_data.get("agentverse_mcp_server", {})
            relevant_data["code_examples"] = mhacks_data.get("code_examples", {})
    
    return relevant_data

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
                """yo! 👋 I'm your MHacks support agent with all the intel on prizes ($1250-$500), judging criteria, code examples, requirements & troubleshooting 🤖

ready to help you ship something legendary! what's up? 🚀"""
            )
            await ctx.send(sender, welcome_response)
            
        elif isinstance(item, TextContent):
            # Classify query and get relevant data
            categories = await classify_query_with_openai(item.text)
            relevant_data = get_relevant_data(ctx, categories, item.text)  # Fixed: Added user_query parameter
            
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
import os, asyncio, aiohttp, json
from typing import List, Dict, Any, AsyncIterator
from pydantic import BaseModel, SecretStr
from langchain.callbacks.base import AsyncCallbackHandler
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import ConfigurableField
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pathlib import Path

# Get the project root directory (where .env should be)
ROOT_DIR = Path(__file__).parent.parent
ENV_PATH = ROOT_DIR / '.env'

# Debug: Print paths and check if .env exists
print("\n=== Environment Debug Info ===")
print(f"Project root directory: {ROOT_DIR}")
print(f"Looking for .env at: {ENV_PATH}")
print(f".env file exists: {ENV_PATH.exists()}")

if ENV_PATH.exists():
    print("\n=== .env File Contents ===")
    try:
        with open(ENV_PATH, 'r') as f:
            # Print first few characters of each line to avoid exposing full API keys
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    masked_value = value[:4] + '...' if value else 'None'
                    print(f"{key}={masked_value}")
    except Exception as e:
        print(f"Error reading .env file: {e}")

# Load environment variables from .env file
print("\n=== Loading Environment Variables ===")
load_dotenv(dotenv_path=ENV_PATH)

# Print all environment variables (masked)
print("\n=== Environment Variables ===")
for key in ['OPENAI_API_KEY', 'SERPAPI_API_KEY']:
    value = os.environ.get(key)
    if value:
        masked_value = value[:4] + '...' if value else 'None'
        print(f"{key}={masked_value}")
    else:
        print(f"{key} not found in environment")

# --- API KEYS ---
try:
    OPENAI_API_KEY = SecretStr(os.environ["OPENAI_API_KEY"])
    print("\nOpenAI API key loaded successfully")
except KeyError as e:
    print(f"\nERROR: {str(e)}")
    print("Please ensure you have a .env file in the project root with OPENAI_API_KEY")
    raise

try:
    SERPAPI_API_KEY = SecretStr(os.environ["SERPAPI_API_KEY"])
    print("SerpAPI key loaded successfully")
except KeyError as e:
    print(f"\nERROR: {str(e)}")
    print("Please ensure you have a .env file in the project root with SERPAPI_API_KEY")
    raise

# --- LLM ---
print("\n=== Initializing LLM ===")
llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0,
    streaming=True,
    api_key=OPENAI_API_KEY
).configurable_fields(callbacks=ConfigurableField(id="callbacks", name="callbacks", description="Stream"))

# --- Prompt ---
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful assistant that uses tools to answer questions. "
        "For any question that requires real-time or factual information, use the serpapi tool to search for information. "
        "For mathematical questions, use the appropriate math tools (add, subtract, multiply, exponentiate). "
        "Always use tools when appropriate, and only use final_answer when you have a complete answer. "
        "When using serpapi, make sure to include relevant information from the search results in your final answer."
    )),
    MessagesPlaceholder(variable_name="messages"),
])

# --- SerpAPI Result ---
class Article(BaseModel):
    title: str
    source: str
    link: str
    snippet: str

    @classmethod
    def from_serpapi_result(cls, r: dict):
        return cls(**r)

# --- Tool Result Interface ---
class ToolResult(BaseModel):
    """Base class for all tool results."""
    tool_name: str
    args: Dict[str, Any]
    result: Any
    call_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary format."""
        return {
            "tool": self.tool_name,
            "args": self.args,
            "result": self.result,
            "call_id": self.call_id
        }

class MathResult(ToolResult):
    """Result type for mathematical operations."""
    operation: str
    operands: List[float]
    result: float

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "operation": self.operation,
            "operands": self.operands,
            "result": self.result
        })
        return base_dict

class SerpAPIResult(ToolResult):
    """Result type for SerpAPI search."""
    articles: List[Dict[str, str]]

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["result"] = self.articles
        return base_dict

# --- Tools ---
@tool
async def add(x: float, y: float, call_id: str = "") -> MathResult:
    """Add x and y together."""
    result = x + y
    return MathResult(
        tool_name="add",
        args={"x": x, "y": y},
        operation="addition",
        operands=[x, y],
        result=result,
        call_id=call_id
    )

@tool
async def subtract(x: float, y: float, call_id: str = "") -> MathResult:
    """Subtract y from x."""
    result = x - y
    return MathResult(
        tool_name="subtract",
        args={"x": x, "y": y},
        operation="subtraction",
        operands=[x, y],
        result=result,
        call_id=call_id
    )

@tool
async def multiply(x: float, y: float, call_id: str = "") -> MathResult:
    """Multiply x and y."""
    result = x * y
    return MathResult(
        tool_name="multiply",
        args={"x": x, "y": y},
        operation="multiplication",
        operands=[x, y],
        result=result,
        call_id=call_id
    )

@tool
async def exponentiate(x: float, y: float, call_id: str = "") -> MathResult:
    """Raise x to the power of y."""
    result = x ** y
    return MathResult(
        tool_name="exponentiate",
        args={"x": x, "y": y},
        operation="exponentiation",
        operands=[x, y],
        result=result,
        call_id=call_id
    )

@tool
async def serpapi(query: str, call_id: str = "") -> SerpAPIResult:
    """Search Google using SerpAPI."""
    params = {
        "api_key": SERPAPI_API_KEY.get_secret_value(),
        "engine": "google",
        "q": query
    }
    async with aiohttp.ClientSession() as session:
        async with session.get("https://serpapi.com/search", params=params) as res:
            data = await res.json()
    
    articles = []
    for r in data.get("organic_results", []):
        try:
            article = {
                "title": r.get("title", ""),
                "source": r.get("source", ""),
                "link": r.get("link", ""),
                "snippet": r.get("snippet", "")
            }
            articles.append(article)
        except Exception as e:
            print(f"[ERROR] Failed to process article: {e}")
            continue
    
    return SerpAPIResult(
        tool_name="serpapi",
        args={"query": query},
        articles=articles,
        result=articles,
        call_id=call_id
    )

class FinalAnswer(BaseModel):
    answer: str
    tools_used: List[str]

@tool
async def final_answer(answer: str, tools_used: List[str]) -> FinalAnswer:
    """Return the final answer to the user."""
    return FinalAnswer(answer=answer, tools_used=tools_used)

tools = [add, subtract, multiply, exponentiate, serpapi, final_answer]
name2tool = {t.name: t.coroutine for t in tools}

# --- Callback Handler ---
class QueueCallbackHandler(AsyncCallbackHandler):
    """Callback handler that puts messages into a queue."""
    
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.content_buffer = ""
        self.tool_results = {}  # Track tool results by call ID
    
    async def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        """Handle tool start event."""
        tool_name = serialized.get("name", "")
        args = json.loads(input_str) if isinstance(input_str, str) else input_str
        tool_call_id = serialized.get("id", "")
        
        # Initialize tool result tracking
        self.tool_results[tool_call_id] = {
            "name": tool_name,
            "args": args,
            "result": None
        }
        
        # Send tool start message
        tool_start_msg = f'<step><step_name>tool_start</step_name>{{"tool": "{tool_name}", "args": {json.dumps(args)}}}</step>\n'
        await self.queue.put(tool_start_msg)
    
    async def on_tool_end(self, output: Any, **kwargs):
        """Handle tool end event."""
        tool_call_id = kwargs.get("tool_call_id", "")
        tool_info = self.tool_results.get(tool_call_id, {})
        
        try:
            # Convert output to ToolResult if it isn't already
            if not isinstance(output, ToolResult):
                output = ToolResult(
                    tool_name=tool_info["name"],
                    args=tool_info["args"],
                    result=output,
                    call_id=tool_call_id
                )
            else:
                # Ensure the call_id is set correctly
                output.call_id = tool_call_id
            
            # Get the serialized result
            result_dict = output.to_dict()
            
            # Send the tool result
            tool_result_msg = f'<step><step_name>tool_result</step_name>{json.dumps(result_dict)}</step>\n'
            await self.queue.put(tool_result_msg)
            
        except Exception as e:
            print(f"[ERROR] Failed to format tool result: {e}")
            error_msg = f'<step><step_name>error</step_name>{{"error": "Failed to format tool result"}}</step>\n'
            await self.queue.put(error_msg)

# --- Memory ---
chat_map = {}
def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_map:
        chat_map[session_id] = InMemoryChatMessageHistory()
    return chat_map[session_id]

# --- Tool Call Reconstruction Helper ---
class ToolCallAccumulator:
    def __init__(self):
        self.tool_calls: Dict[int, Dict[str, Any]] = {}
    
    def add_chunk(self, chunk_data: Dict[str, Any]):
        """Add a tool call chunk and reconstruct the complete tool call"""
        index = chunk_data.get('index', 0)
        
        if index not in self.tool_calls:
            self.tool_calls[index] = {
                'name': '',
                'id': '',
                'args_str': '',
                'type': 'tool_call'
            }
        
        tool_call = self.tool_calls[index]
        
        # Update name and id from the first chunk
        if chunk_data.get('name'):
            tool_call['name'] = chunk_data['name']
        if chunk_data.get('id'):
            tool_call['id'] = chunk_data['id']
        
        # Accumulate arguments string
        if chunk_data.get('args'):
            tool_call['args_str'] += chunk_data['args']
    
    def get_complete_tool_calls(self) -> List[Dict[str, Any]]:
        """Get all complete tool calls with parsed arguments"""
        complete_calls = []
        
        for tool_call in self.tool_calls.values():
            if tool_call['name'] and tool_call['id']:
                try:
                    # Parse the accumulated args string as JSON
                    args = json.loads(tool_call['args_str']) if tool_call['args_str'] else {}
                    complete_calls.append({
                        'name': tool_call['name'],
                        'id': tool_call['id'],
                        'args': args,
                        'type': tool_call['type']
                    })
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Failed to parse tool call args: {tool_call['args_str']}, error: {e}")
                    # Still add the tool call with empty args to avoid errors
                    complete_calls.append({
                        'name': tool_call['name'],
                        'id': tool_call['id'],
                        'args': {},
                        'type': tool_call['type']
                    })
        
        return complete_calls

# --- Agent Executor ---
class CustomAgentExecutor:
    def __init__(self, max_iterations=3):
        # Simplified chain that just takes messages
        chain = prompt | llm.bind_tools(tools)
        self.agent = chain
        self.max_iterations = max_iterations

    def _get_valid_message_history(self, messages):
        """
        Extract only valid message sequences from history.
        OpenAI requires that ToolMessages immediately follow AIMessages with tool_calls.
        """
        valid_messages = []
        i = 0
        
        while i < len(messages):
            msg = messages[i]
            
            if isinstance(msg, HumanMessage):
                valid_messages.append(msg)
                i += 1
            elif isinstance(msg, AIMessage):
                # Check if this AI message has tool_calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    # This AI message has tool calls, so we need to find the corresponding ToolMessages
                    valid_messages.append(msg)
                    i += 1
                    
                    # Collect all the tool call IDs we need responses for
                    expected_tool_call_ids = {tc.get('id') for tc in msg.tool_calls if tc.get('id')}
                    found_tool_call_ids = set()
                    
                    # Look for consecutive ToolMessages that respond to these tool calls
                    while i < len(messages) and isinstance(messages[i], ToolMessage):
                        tool_msg = messages[i]
                        if hasattr(tool_msg, 'tool_call_id') and tool_msg.tool_call_id in expected_tool_call_ids:
                            valid_messages.append(tool_msg)
                            found_tool_call_ids.add(tool_msg.tool_call_id)
                        i += 1
                    
                    # If we didn't find all expected tool responses, we might have a broken sequence
                    # but we'll keep what we have
                    if len(found_tool_call_ids) != len(expected_tool_call_ids):
                        print(f"[WARNING] Incomplete tool call sequence. Expected {len(expected_tool_call_ids)}, found {len(found_tool_call_ids)}")
                
                else:
                    # Regular AI message without tool calls
                    valid_messages.append(msg)
                    i += 1
            else:
                # Skip any orphaned ToolMessages or other message types
                print(f"[WARNING] Skipping orphaned or invalid message: {type(msg).__name__}")
                i += 1
        
        return valid_messages

    async def stream_invoke(self, input: str, session_id: str):
        print(f"[DEBUG] Starting stream_invoke with input: {input}")
        print(f"[DEBUG] Session ID: {session_id}")

        # Get chat history
        history = get_chat_history(session_id)
        messages = history.messages
        print(f"[DEBUG] History messages count: {len(messages)}")

        # Get valid message history
        working_messages = self._get_valid_message_history(messages)
        print(f"[DEBUG] Valid history messages count: {len(working_messages)}")

        # Add user message
        working_messages.append(HumanMessage(content=input))
        print(f"[DEBUG] Added user message: {input}")
        print(f"[DEBUG] Working messages count: {len(working_messages)}")

        # Create queue for streaming
        queue = asyncio.Queue()
        
        async def run_agent():
            try:
                iteration = 1
                while iteration <= self.max_iterations:
                    print(f"\n[DEBUG] Iteration {iteration}")
                    print(f"[DEBUG] Working messages count: {len(working_messages)}")
                    
                    # Debug print messages
                    for i, msg in enumerate(working_messages):
                        print(f"[DEBUG] Message {i}: {type(msg).__name__}")

                    print("[DEBUG] Sending input payload to agent")
                    print("[DEBUG] Starting to stream agent response")

                    # Create callback handler for this iteration
                    callback_handler = QueueCallbackHandler(queue)

                    # Run agent with callbacks
                    response = await self.agent.ainvoke(
                        {"messages": working_messages},
                        config={"callbacks": [callback_handler]}
                    )
                    
                    print(f"[DEBUG] Got response from agent: {type(response)}")
                    print(f"[DEBUG] Response content: {response.content if hasattr(response, 'content') else 'No content'}")
                    print(f"[DEBUG] Has tool_calls: {hasattr(response, 'tool_calls')}")
                    if hasattr(response, 'tool_calls'):
                        print(f"[DEBUG] Tool calls: {response.tool_calls}")

                    # Process response
                    if isinstance(response, AIMessage):
                        # Add AI message to both working messages and history
                        working_messages.append(response)
                        history.add_message(response)
                        
                        # Check for tool calls
                        if hasattr(response, 'tool_calls') and response.tool_calls:
                            print(f"[DEBUG] Processing {len(response.tool_calls)} tool calls")
                            # Track tools used
                            tools_used = []
                            # Process tool calls
                            for tool_call in response.tool_calls:
                                # Handle both dictionary and object tool calls
                                tool_name = tool_call.get('name') if isinstance(tool_call, dict) else tool_call.name
                                args = tool_call.get('args') if isinstance(tool_call, dict) else tool_call.args
                                tool_call_id = tool_call.get('id') if isinstance(tool_call, dict) else tool_call.id

                                print(f"[DEBUG] Processing tool call: {tool_name}")
                                print(f"[DEBUG] Tool args: {args}")
                                print(f"[DEBUG] Tool call ID: {tool_call_id}")

                                try:
                                    # Send tool start message
                                    tool_start_msg = f'<step><step_name>tool_start</step_name>{{"tool": "{tool_name}", "args": {json.dumps(args)}}}</step>\n'
                                    print(f"[DEBUG] Sending tool start message: {tool_start_msg}")
                                    await queue.put(tool_start_msg)
                                    
                                    # Execute tool
                                    print(f"[DEBUG] Executing tool: {tool_name}")
                                    result = await name2tool[tool_name](**args)
                                    print(f"[DEBUG] Tool result: {result}")
                                    tools_used.append(tool_name)
                                    
                                    # Send tool result to frontend
                                    print(f"[DEBUG] Sending tool result to frontend: {tool_name}")
                                    # Format the result as a proper JSON string
                                    tool_result = {
                                        "tool": tool_name,
                                        "result": str(result),
                                        "args": args
                                    }
                                    result_json = json.dumps(tool_result)
                                    print(f"[DEBUG] Tool result JSON: {result_json}")
                                    tool_result_msg = f'<step><step_name>tool_result</step_name>{result_json}</step>\n'
                                    print(f"[DEBUG] Sending tool result message: {tool_result_msg}")
                                    await queue.put(tool_result_msg)
                                    
                                    # Add tool message to both working messages and history
                                    tool_msg = ToolMessage(
                                        content=str(result),
                                        tool_call_id=tool_call_id
                                    )
                                    working_messages.append(tool_msg)
                                    history.add_message(tool_msg)
                                    
                                except Exception as e:
                                    error_msg = f"Error executing {tool_name}: {str(e)}"
                                    print(f"[ERROR] {error_msg}")
                                    print(f"[ERROR] Full error: {str(e)}")
                                    
                                    # Add error message to both working messages and history
                                    tool_msg = ToolMessage(
                                        content=error_msg,
                                        tool_call_id=tool_call_id
                                    )
                                    working_messages.append(tool_msg)
                                    history.add_message(tool_msg)
                                    error_step = f'<step><step_name>error</step_name>{{"error": "{error_msg}"}}</step>\n'
                                    print(f"[DEBUG] Sending error message: {error_step}")
                                    await queue.put(error_step)
                            
                            # After processing all tool calls, send final answer with tools used
                            final_content = response.content
                            if not final_content and tools_used:
                                # If no content but we have tool results, use the last tool result
                                final_content = f"The result is {result}"
                            print(f"[DEBUG] Sending final answer with tools used: {tools_used}")
                            final_answer_msg = f'<step><step_name>final_answer</step_name>{{"answer": "{final_content}", "tools_used": {json.dumps(tools_used)}}}</step>\n'
                            print(f"[DEBUG] Sending final answer message: {final_answer_msg}")
                            await queue.put(final_answer_msg)
                            print("[DEBUG] Sending DONE message")
                            await queue.put("<<DONE>>\n")
                            break
                        else:
                            # No tool calls, send content directly
                            final_content = response.content
                            print(f"[DEBUG] No valid tool calls, sending final answer: {final_content}")
                            # Format as a final_answer step with empty tools list
                            final_answer_msg = f'<step><step_name>final_answer</step_name>{{"answer": "{final_content}", "tools_used": []}}</step>\n'
                            print(f"[DEBUG] Sending final answer message: {final_answer_msg}")
                            await queue.put(final_answer_msg)
                            print("[DEBUG] Sending DONE message")
                            await queue.put("<<DONE>>\n")
                            break
                    
                    iteration += 1

                if iteration > self.max_iterations:
                    print("[DEBUG] Max iterations reached")
                    await queue.put(f'<step><step_name>final_answer</step_name>{{"answer": "Max iterations reached without final answer", "tools_used": []}}</step>\n')
                    await queue.put("<<DONE>>\n")

            except Exception as e:
                print(f"[ERROR] Agent Error: {e}")
                print(f"[ERROR] Full error: {str(e)}")
                await queue.put(f'<step><step_name>error</step_name>{{"error": "{str(e)}"}}</step>\n')
                await queue.put("<<DONE>>\n")

        # Start agent in background
        asyncio.create_task(run_agent())

        # Return an async iterator that yields from the queue
        async def queue_iterator():
            while True:
                item = await queue.get()
                if item == "<<DONE>>":
                    break
                yield item
        
        return queue_iterator()

agent_executor = CustomAgentExecutor()
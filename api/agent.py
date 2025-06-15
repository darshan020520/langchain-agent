import os, asyncio, aiohttp, json
from typing import List, Dict, Any
from pydantic import BaseModel, SecretStr
from langchain.callbacks.base import AsyncCallbackHandler
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import ConfigurableField
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# --- API KEYS ---
OPENAI_API_KEY = SecretStr(os.environ["OPENAI_API_KEY"])
SERPAPI_API_KEY = SecretStr(os.environ["SERPAPI_API_KEY"])

# --- LLM ---
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    streaming=True,
    api_key=OPENAI_API_KEY
).configurable_fields(callbacks=ConfigurableField(id="callbacks", name="callbacks", description="Stream"))

# --- Prompt ---
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful assistant. Use tools to answer the current query. "
        "Always include valid tool arguments. Use `final_answer` only to finish."
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

# --- Tools ---
@tool
async def add(x: float, y: float) -> float:
    """Add x and y together."""
    return x + y

@tool
async def subtract(x: float, y: float) -> float:
    """Subtract y from x."""
    return x - y

@tool
async def multiply(x: float, y: float) -> float:
    """Multiply x and y."""
    return x * y

@tool
async def exponentiate(x: float, y: float) -> float:
    """Raise x to the power of y."""
    return x ** y

@tool
async def serpapi(query: str) -> List[Article]:
    """Search Google using SerpAPI."""
    params = {
        "api_key": SERPAPI_API_KEY.get_secret_value(),
        "engine": "google",
        "q": query
    }
    async with aiohttp.ClientSession() as session:
        async with session.get("https://serpapi.com/search", params=params) as res:
            data = await res.json()
    return [Article.from_serpapi_result(r) for r in data.get("organic_results", [])]

class FinalAnswer(BaseModel):
    answer: str
    tools_used: List[str]

@tool
async def final_answer(answer: str, tools_used: List[str]) -> FinalAnswer:
    """Return the final answer to the user."""
    return FinalAnswer(answer=answer, tools_used=tools_used)

tools = [add, subtract, multiply, exponentiate, serpapi, final_answer]
name2tool = {t.name: t.coroutine for t in tools}

# --- Callback for Streaming ---
class QueueCallbackHandler(AsyncCallbackHandler):
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def __aiter__(self):
        while True:
            item = await self.queue.get()
            if item == "<<DONE>>":
                break
            yield item

    async def on_llm_new_token(self, token: str, **kwargs):
        if token.strip():
            await self.queue.put(token)

    async def on_llm_end(self, *args, **kwargs):
        await self.queue.put("<<DONE>>")

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
        queue = asyncio.Queue()
        callback = QueueCallbackHandler(queue)
        history = get_chat_history(session_id)
        
        # Get only valid message sequences from history
        valid_history = self._get_valid_message_history(history.messages)
        
        # Add user message
        current_user_message = HumanMessage(content=input)
        
        # We'll work with a working copy of messages for the conversation
        working_messages = valid_history + [current_user_message]

        async def run_agent():
            try:
                for iteration in range(self.max_iterations):
                    print(f"[DEBUG] Iteration {iteration + 1}")
                    print(f"[DEBUG] Working messages count: {len(working_messages)}")
                    
                    # Print message types for debugging
                    for i, msg in enumerate(working_messages):
                        msg_type = type(msg).__name__
                        has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
                        print(f"[DEBUG] Message {i}: {msg_type} {'(with tool_calls)' if has_tool_calls else ''}")
                    
                    # Prepare input for the agent
                    input_payload = {"messages": working_messages}
                    
                    # Collect chunks and build complete message
                    chunks = []
                    content_parts = []
                    tool_call_accumulator = ToolCallAccumulator()
                    
                    async for chunk in self.agent.astream(input_payload, config={
                        "callbacks": [callback]
                    }):
                        chunks.append(chunk)
                        
                        # Collect content for streaming
                        if hasattr(chunk, "content") and chunk.content:
                            content_parts.append(chunk.content)
                        
                        # Process tool call chunks
                        if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                            for tcc in chunk.tool_call_chunks:
                                print(f"[DEBUG] Tool call chunk: {tcc}")
                                tool_call_accumulator.add_chunk(tcc)
                    
                    # Get reconstructed tool calls
                    collected_tool_calls = tool_call_accumulator.get_complete_tool_calls()
                    print(f"[DEBUG] Reconstructed {len(collected_tool_calls)} tool calls")
                    
                    # Get the final message
                    final_message = chunks[-1] if chunks else None
                    final_content = "".join(content_parts)
                    
                    # Check finish reason to determine if we have tool calls
                    has_tool_calls = False
                    if final_message and hasattr(final_message, "response_metadata"):
                        finish_reason = final_message.response_metadata.get("finish_reason")
                        has_tool_calls = finish_reason == "tool_calls"
                        print(f"[DEBUG] Finish reason: {finish_reason}, Has tool calls: {has_tool_calls}")
                    
                    # If no tool calls, we're done
                    if not has_tool_calls or not collected_tool_calls:
                        if final_content:
                            ai_message = AIMessage(content=final_content)
                            working_messages.append(ai_message)
                            # Add user message and AI response to history
                            history.add_message(current_user_message)
                            history.add_message(ai_message)
                        elif final_message and hasattr(final_message, "content"):
                            ai_message = AIMessage(content=final_message.content or "")
                            working_messages.append(ai_message)
                            # Add user message and AI response to history
                            history.add_message(current_user_message)
                            history.add_message(ai_message)
                        break

                    # Process tool calls
                    tool_messages = []
                    valid_tool_calls = []
                    
                    print(f"[DEBUG] About to process {len(collected_tool_calls)} tool calls")
                    
                    for call in collected_tool_calls:
                        # Extract tool information
                        tool_name = call.get("name")
                        tool_call_id = call.get("id")
                        args = call.get("args", {})
                        
                        # Skip invalid tool calls
                        if not tool_name or not tool_call_id:
                            print(f"[DEBUG] Skipping invalid tool call: name={tool_name}, id={tool_call_id}")
                            continue
                            
                        valid_tool_calls.append(call)
                        
                        try:
                            # Execute tool
                            if tool_name in name2tool:
                                print(f"[DEBUG] Executing {tool_name} with args: {args}")
                                result = await name2tool[tool_name](**args)
                                print(f"[DEBUG] Tool {tool_name} returned: {result}")
                                
                                # Create tool message
                                tool_msg = ToolMessage(
                                    content=str(result), 
                                    tool_call_id=tool_call_id
                                )
                                tool_messages.append(tool_msg)
                                
                                await queue.put(f"\n[Tool: {tool_name}] Result: {result}")
                                
                                # Check for final answer
                                if tool_name == "final_answer":
                                    final_ans = args.get('answer', str(result))
                                    await queue.put(f"\n✅ Final Answer: {final_ans}")
                                    
                                    # Add AI message with tool calls
                                    formatted_tool_calls = []
                                    for call in valid_tool_calls:
                                        formatted_call = {
                                            "name": call["name"],
                                            "args": call["args"],
                                            "id": call["id"],
                                            "type": "tool_call"
                                        }
                                        formatted_tool_calls.append(formatted_call)
                                    
                                    ai_msg = AIMessage(content=final_content, tool_calls=formatted_tool_calls)
                                    working_messages.append(ai_msg)
                                    
                                    # Add tool messages
                                    for tm in tool_messages:
                                        working_messages.append(tm)
                                    
                                    # Add all messages to history
                                    history.add_message(current_user_message)
                                    history.add_message(ai_msg)
                                    for tm in tool_messages:
                                        history.add_message(tm)
                                    
                                    await queue.put("<<DONE>>")
                                    return callback
                            else:
                                error_msg = f"Unknown tool: {tool_name}"
                                tool_msg = ToolMessage(
                                    content=error_msg,
                                    tool_call_id=tool_call_id
                                )
                                tool_messages.append(tool_msg)
                                await queue.put(f"\n❌ {error_msg}")
                                
                        except Exception as e:
                            error_msg = f"Tool {tool_name} failed: {str(e)}"
                            print(f"[ERROR] {error_msg}")
                            
                            tool_msg = ToolMessage(
                                content=error_msg,
                                tool_call_id=tool_call_id
                            )
                            tool_messages.append(tool_msg)
                            await queue.put(f"\n❌ {error_msg}")

                    # Add messages to working_messages for next iteration
                    if valid_tool_calls and tool_messages:
                        # Create AI message with tool calls
                        formatted_tool_calls = []
                        for call in valid_tool_calls:
                            formatted_call = {
                                "name": call["name"],
                                "args": call["args"],
                                "id": call["id"],
                                "type": "tool_call"
                            }
                            formatted_tool_calls.append(formatted_call)
                        
                        ai_msg = AIMessage(content=final_content or "", tool_calls=formatted_tool_calls)
                        working_messages.append(ai_msg)
                        
                        # Add corresponding tool messages immediately after
                        for tool_msg in tool_messages:
                            working_messages.append(tool_msg)
                        
                        print(f"[DEBUG] Added AI message + {len(tool_messages)} tool messages to working_messages")
                        
                        # Continue to next iteration
                        continue
                        
                    else:
                        # No valid tool calls - we're done
                        ai_msg = AIMessage(content=final_content)
                        working_messages.append(ai_msg)
                        # Add user message and AI response to history
                        history.add_message(current_user_message)
                        history.add_message(ai_msg)
                        break

                # If we reach max iterations, add accumulated messages to history
                # Add user message first, then any remaining messages
                if current_user_message not in history.messages:
                    history.add_message(current_user_message)
                
                # Add any remaining working messages that aren't in history
                history_len = len(history.messages)
                for msg in working_messages[len(valid_history) + 1:]:  # Skip the messages we already know about
                    if msg not in history.messages:
                        history.add_message(msg)
                
                await queue.put("\n⚠️ Max iterations reached without final answer")
                await queue.put("<<DONE>>")

            except Exception as e:
                print(f"[ERROR] Agent Error: {e}")
                await queue.put(f"\n💥 Agent Error: {e}")
                await queue.put("<<DONE>>")

        await run_agent()
        return callback

agent_executor = CustomAgentExecutor()
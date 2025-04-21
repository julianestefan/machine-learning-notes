# Agentic Systems in LangChain

Agentic systems represent the evolution of LLM applications from simple text generators to autonomous systems that can reason, plan, and take actions to accomplish specific tasks. This document explores how to build, customize, and deploy agentic systems using LangChain and LangGraph.

## 1. Introduction to Agents

Agents are AI systems that can interpret user requests, make decisions, and take actions to achieve goals. Unlike simpler LLM applications, agents can:

- **Reason** about problems and break them down
- **Plan** a sequence of steps to solve a task
- **Use tools** to interact with external systems and data sources
- **Learn** from past interactions and improve over time
- **Reflect** on their performance and adapt strategies

### The ReAct Framework

Most agents in LangChain follow the ReAct (**Rea**soning and **Act**ion) framework, which establishes a cycle of:

1. **Reasoning**: Analyzing the current state and deciding what to do
2. **Acting**: Executing a chosen tool or action
3. **Observing**: Processing the results of the action
4. **Reasoning Again**: Planning the next step based on observations

```python
# A simplified representation of the ReAct process
def react_process(query):
    # Initial reasoning about the query
    thought = "I need to understand what the user is asking and determine the tools I need."
    
    # Choose and execute an action
    action = "Use tool X to get information about Y"
    observation = execute_tool(action)
    
    # Reason about the observation
    thought = "Based on the information from tool X, I now need to..."
    
    # Further actions and reasoning...
    return final_response
```

## 2. Building Blocks of Agents

### Tools

Tools are functions that agents can call to interact with the world or perform specific tasks. They are defined with a clear purpose, input/output specification, and description to guide the agent.

```python
# Import necessary libraries
from langchain.tools import tool

# Define a simple tool
@tool
def count_letters(word: str, letter: str) -> int:
    """Counts how many times a specific letter appears in a word.
    
    Args:
        word: The word to analyze
        letter: The letter to count occurrences of
        
    Returns:
        int: The number of occurrences of the letter in the word
    """
    return word.lower().count(letter.lower())
```

### LLM Decision Making

The language model is the "brain" of the agent, making decisions about:
- Which tools to use
- How to interpret results
- When to request more information
- When a task is complete

```python
# Import required libraries
from langchain_openai import ChatOpenAI

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)
```

### Agent State Management

Agents need to maintain state to track conversation history, intermediate results, and progress toward goals.

```python
# Import necessary libraries
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage

# Define a state structure for the agent
class AgentState(TypedDict):
    # Track conversation history
    messages: List[BaseMessage]
    # Track intermediate results
    results: dict
    # Track current goal
    current_goal: str
```

## 3. Creating Basic Agents

Creating a simple agent involves defining tools, connecting them to a language model, and establishing the interaction flow.

```python
# Import necessary libraries
from langchain.agents import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import math

# Define a tool for our agent
@tool
def calculate_area(sides: str) -> float:
    """Calculate the area of a rectangle.
    
    Args:
        sides: A comma-separated string with two numbers representing length and width
              Example: "5, 3" or "10.5, 7.2"
    
    Returns:
        float: The area of the rectangle
    """
    dimensions = sides.split(',')
    length = float(dimensions[0].strip())
    width = float(dimensions[1].strip())
    return length * width

# Initialize the language model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create the ReAct agent
agent = create_react_agent(
    llm=model,
    tools=[calculate_area],
    handle_parsing_errors=True
)

# Create an input query
query = "What is the area of a rectangle with sides 7.5 and 12?"

# Execute the agent
response = agent.invoke({"messages": [HumanMessage(content=query)]})

# Print the agent's response
print(response["messages"][-1].content)
```

## 4. Tools Integration

Agents become more powerful by integrating various tools to extend their capabilities.

### Common Tool Types

| Tool Category | Description | Example Use Cases |
|---------------|-------------|------------------|
| Information Retrieval | Access external data sources | Web search, database queries |
| Computation | Perform calculations | Math operations, statistics |
| Code Execution | Run code snippets | Data analysis, automation |
| API Interaction | Connect to external services | Weather data, stock prices |
| File Operations | Work with documents | Read/write files, extract text |

### Building Custom Tool Sets

```python
# Import necessary libraries
from langchain.tools import tool
from langchain.agents import create_react_agent
from langchain_openai import ChatOpenAI
import requests

# Define multiple tools
@tool
def check_palindrome(text: str) -> bool:
    """Check if a string is a palindrome (reads the same backward as forward).
    
    Args:
        text: The string to check
        
    Returns:
        bool: True if the text is a palindrome, False otherwise
    """
    # Remove spaces and convert to lowercase
    cleaned_text = ''.join(text.lower().split())
    return cleaned_text == cleaned_text[::-1]

@tool
def fetch_historical_event(date: str) -> str:
    """Find a significant historical event that occurred on a specific date.
    
    Args:
        date: Date in format "DD Month" (e.g., "20 July" or "4 October")
        
    Returns:
        str: Description of a historical event from that date
    """
    # This would typically call an API, but we'll simulate a response
    events = {
        "20 july": "On July 20, 1969, Apollo 11 astronauts Neil Armstrong and Buzz Aldrin became the first humans to walk on the Moon.",
        "9 november": "On November 9, 1989, the Berlin Wall fell, marking a significant moment in the end of the Cold War.",
        "8 may": "On May 8, 1945, Victory in Europe Day (V-E Day) marked the formal end of World War II in Europe."
    }
    
    date_key = date.lower()
    return events.get(date_key, f"No significant event found for {date}")

# Initialize the language model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create an agent with multiple tools
agent = create_react_agent(
    llm=model,
    tools=[check_palindrome, fetch_historical_event],
    handle_parsing_errors=True
)

# Process a query
response = agent.invoke({
    "messages": [HumanMessage(content="Is 'madam' a palindrome?")]
})

print(f"Agent response: {response['messages'][-1].content}")
```

### Tool Selection Logic

Agents need to decide when and which tools to use. This can be implemented using:

1. **LLM-based selection**: The model decides based on the query
2. **Rule-based routing**: Predefined rules direct queries to specific tools
3. **Hybrid approaches**: Combining rules with LLM judgment

```python
# Example of a hybrid tool selection function
def select_appropriate_tool(query, available_tools):
    # Rule-based pre-filtering
    if "palindrome" in query.lower():
        candidates = [tool for tool in available_tools if "palindrome" in tool.name]
    elif any(date_word in query.lower() for date_word in ["date", "when", "year", "historical"]):
        candidates = [tool for tool in available_tools if "historical" in tool.name]
    else:
        candidates = available_tools
    
    # Let the LLM make the final decision if multiple candidates remain
    if len(candidates) > 1:
        return llm_based_tool_selection(query, candidates)
    elif candidates:
        return candidates[0]
    else:
        return None  # No suitable tool found
```

## 5. Graph-Based Agents with LangGraph

LangGraph provides a powerful framework for building complex, stateful agents using directed graphs.

### Graph Architecture Fundamentals

In LangGraph, applications are modeled as directed graphs where:

- **Nodes**: Functions, tools, or processing steps
- **Edges**: Rules determining how information flows between nodes

```python
# Import necessary libraries
import os
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

# Define the state structure
class State(TypedDict):
    messages: Annotated[list, "add_messages"]

# Initialize StateGraph with our State type
graph_builder = StateGraph(State)

# Define chatbot function to respond with the model
def chatbot(state: State):
    """Process the current state and generate a response using the LLM"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Add chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)

# Define the conversation flow: START → chatbot → END
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph to prepare for execution
graph = graph_builder.compile()
```

### Stateful Conversations

LangGraph maintains two types of state for agents:

- **Graph State**: Manages different workflows and tasks
- **Agent State**: Tracks conversation history, context, and progress

### Visualizing Agent Workflows

LangGraph provides tools to visualize agent architecture, which helps in debugging and documentation.

```python
# Import necessary libraries
from IPython.display import Image

# Generate and display the diagram of the graph
graph_visualization = graph.get_graph().draw_mermaid_png()
display(Image(graph_visualization))
```

A simple graph visualization might look like:

```
[START] --> [chatbot] --> [END]
```

More complex agents with tool integration might have loops and branches:

```
[START] --> [chatbot] --> [tool_decision]
[tool_decision] --> [tool1]
[tool_decision] --> [tool2]
[tool_decision] --> [END]
[tool1] --> [chatbot]
[tool2] --> [chatbot]
```

## 6. Advanced Agent Capabilities

### Conditional Branching with Tool Nodes

More sophisticated agents use conditional branching to decide whether to use tools or provide direct responses.

```python
# Import necessary libraries
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from typing import Annotated, TypedDict, List

# Define the state structure
class MessagesState(TypedDict):
    messages: List[BaseMessage]

# Initialize the LLM and tools
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [check_palindrome, fetch_historical_event]
llm_with_tools = llm.bind_tools(tools)

# Create a tool node
tool_node = ToolNode(tools)

# Initialize the graph
workflow = StateGraph(MessagesState)

# Define the model function
def call_model(state: MessagesState):
    last_message = state["messages"][-1]
    
    # Handle tool responses
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return {"messages": [AIMessage(content=last_message.tool_calls[0]["response"])]}
    
    # Generate new responses
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Define the branching logic
def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    
    # Check if the last message includes tool calls
    if last_message.tool_calls:
        return "tools"
    
    # End the conversation if no tool calls are present
    return END

# Add nodes and edges
workflow.add_node("chatbot", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges("chatbot", should_continue, ["tools", END])
workflow.add_edge("tools", "chatbot")

# Compile the workflow
app = workflow.compile()
```

### Multi-turn Conversations

Agents can maintain context across multiple interactions, enabling natural conversation flows:

```python
# Function to handle multi-turn conversation
def user_agent_conversation(queries):
    # Set up memory for persistence
    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()
    app_with_memory = workflow.compile(checkpointer=memory)
    
    conversation_id = memory.new_checkpoint()
    
    for i, query in enumerate(queries):
        print(f"User: {query}")
        
        # First query starts fresh
        if i == 0:
            inputs = {"messages": [HumanMessage(content=query)]}
        # Subsequent queries use conversation history
        else:
            inputs = {"messages": [HumanMessage(content=query)]}
            
        # Stream the response
        print("Agent: ", end="")
        for event in app_with_memory.stream(inputs, {"configurable": {"thread_id": conversation_id}}):
            for node_name, node_outputs in event.items():
                if "messages" in node_outputs:
                    for msg in node_outputs["messages"]:
                        if not isinstance(msg, HumanMessage) and hasattr(msg, "content"):
                            print(msg.content, end="", flush=True)
        print("\n")

# Example multi-turn conversation
queries = [
    "Is 'radar' a palindrome?",
    "What about 'hello world'?",
    "Tell me about what happened on May 8th, 1945"
]

user_agent_conversation(queries)
```

## 7. Building Conversational Agents

Conversational agents focus on natural dialogue and can be enhanced with streaming responses for better user experience.

### Streaming Responses

```python
# Import necessary libraries
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

# Set up a streaming-enabled LLM
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

# Define the State structure
class State(TypedDict):
    messages: Annotated[list, "add_messages"]

# Initialize and configure the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", lambda state: {"messages": [llm.invoke(state["messages"])]})
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# Define a function to execute the chatbot with streaming
def stream_conversation(user_input: str):
    """Stream the chatbot's response to a user input"""
    print(f"User: {user_input}")
    print("Agent: ", end="", flush=True)
    
    # Stream events from the graph
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for node_name, node_updates in event.items():
            if "messages" in node_updates:
                for message in node_updates["messages"]:
                    if hasattr(message, "content"):
                        print(message.content, end="", flush=True)
    print()  # Add a new line after the response completes

# Example usage
user_query = "Tell me about artificial intelligence in simple terms."
stream_conversation(user_query)
```

### Tool Integration in Chatbots

Adding tools to conversational agents allows them to access external information while maintaining natural dialogue.

```python
# Import necessary libraries
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

# Initialize Wikipedia tool
api_wrapper = WikipediaAPIWrapper(top_k_results=1)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
tools = [wikipedia_tool]

# Create LLM with tools
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# Set up the graph with the tool
tool_node = ToolNode(tools=[wikipedia_tool])

# Initialize state graph
graph_builder = StateGraph(State)

# Define chatbot function
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Add nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

# Set up conditional routing
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {
        "tool_call": "tools",
        "no_tool_call": END
    }
)

# Connect tools back to chatbot
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile the graph
graph = graph_builder.compile()
```

## 8. Evaluating and Debugging Agents

Ensuring agents perform well requires systematic evaluation and effective debugging strategies.

### Evaluation Metrics

| Metric | Description | How to Measure |
|--------|-------------|----------------|
| Task Completion | Whether the agent successfully completes assigned tasks | Success rate on benchmark tasks |
| Time Efficiency | How quickly the agent completes tasks | Average time to completion |
| Tool Usage | How effectively the agent uses available tools | Appropriate tool selection rate |
| Response Quality | How accurate and relevant the responses are | Human evaluation scores |
| Conversational Flow | How natural and coherent the conversation is | User satisfaction ratings |

### Debugging Techniques

1. **Verbose Output**: Enable detailed logging of agent reasoning steps
2. **Graph Visualization**: Examine the flow of information through the agent
3. **Step-by-Step Execution**: Run the agent one node at a time to pinpoint issues
4. **State Inspection**: Examine the agent's state at different points in execution

```python
# Example of enabling verbose output for debugging
def debug_agent_execution(query, agent, verbose=True):
    """Execute an agent with detailed debugging output"""
    print(f"Input query: {query}")
    
    # Track the agent's reasoning process
    if verbose:
        print("\n=== Agent Reasoning Process ===")
    
    # Execute with verbose output
    response = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        {"callbacks": [VerboseCallback()] if verbose else []}
    )
    
    if verbose:
        print("\n=== Final Response ===")
    
    print(response["messages"][-1].content)
    return response

# Custom callback for detailed logging
class VerboseCallback:
    def on_llm_start(self, *args, **kwargs):
        print("🧠 LLM thinking...")
    
    def on_tool_start(self, tool_name, tool_input, **kwargs):
        print(f"🔧 Using tool: {tool_name} with input: {tool_input}")
    
    def on_tool_end(self, output, **kwargs):
        print(f"📊 Tool result: {output}")
    
    def on_chain_end(self, outputs, **kwargs):
        print(f"⛓️ Chain complete: {outputs}")
```

## 9. Best Practices

### Design Principles

1. **Single Responsibility**: Design each tool to do one thing well
2. **Clear Documentation**: Provide comprehensive descriptions for every tool
3. **Error Handling**: Make tools resilient to unexpected inputs
4. **Stateless Design**: Keep tools stateless unless persistence is required
5. **Progressive Disclosure**: Start with simple flows, add complexity incrementally

### Performance Optimization

1. **Caching**: Cache expensive operations and tool results
2. **Parallelization**: Execute independent tools concurrently
3. **Response Streaming**: Use streaming for better user experience
4. **Efficient Memory Management**: Clean up unused state to prevent memory leaks

### Security Considerations

1. **Input Validation**: Validate all inputs before processing
2. **Sandboxing**: Isolate tool execution environments
3. **Rate Limiting**: Implement rate limits for external API calls
4. **Permission Controls**: Restrict tools to appropriate access levels
5. **Audit Logging**: Track all actions taken by the agent

### Real-world Applications

| Application Type | Description | Key Agent Components |
|------------------|-------------|---------------------|
| Customer Support | Automated support agent handling common queries | Knowledge base retrieval, escalation logic |
| Research Assistant | Agent that helps find and synthesize information | Web search, document analysis, summarization |
| Creative Collaborator | Agent that assists in creative processes | Idea generation, reference gathering, feedback |
| Coding Assistant | Agent that helps with programming tasks | Code generation, debugging, documentation |
| Data Analyst | Agent that automates data analysis workflows | Data retrieval, visualization, statistical tools |

By combining the principles, techniques, and best practices covered in this guide, you can create powerful agentic systems that augment human capabilities and automate complex tasks across a wide range of domains. 
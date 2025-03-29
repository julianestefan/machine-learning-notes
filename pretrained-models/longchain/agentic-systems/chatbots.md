# Building Intelligent Chatbots with LangGraph

LangGraph is a powerful framework for building complex, stateful applications with LLMs. This document covers how to create sophisticated chatbots using LangGraph's graph-based architecture.

## Graph-Based Architecture Fundamentals

LangGraph models applications as directed graphs where:

- **Nodes**: Represent functions, actions, or processing steps in your application flow
- **Edges**: Define the rules and pathways connecting these nodes, controlling how information flows

This architecture enables complex conversational workflows that can handle diverse user inputs and maintain context over time.

### State Management

LangGraph maintains two types of state:

- **Graph State**: Organizes different tasks and workflows in the application
- **Agent State**: Tracks an agent's progress through the graph, including conversation history and context

```python
# Import necessary libraries
import os
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from IPython.display import Image

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"  # Replace with your actual API key

# Define the language model
llm = ChatOpenAI(model="gpt-4o-min")

# Define the State structure to track conversation
class State(TypedDict):
    # Define messages with metadata
    messages: Annotated[list, "add_messages"]

# Initialize StateGraph with our State type
graph_builder = StateGraph(State)

# Define chatbot function to respond with the model
def chatbot(state: State):
    """Process the current state and generate a response using the LLM"""
    # The LLM takes the current messages and generates a response
    response = llm.invoke(state["messages"])
    # Return the updated messages including the new response
    return {"messages": [response]}

# Add chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)

# Define the conversation flow: START → chatbot → END
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph to prepare for execution
graph = graph_builder.compile()
```

## Streaming Responses

LangGraph supports streaming responses, which provides a more interactive user experience by showing responses as they're generated rather than waiting for the complete response.

```python
# Import necessary libraries
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"  # Replace with your actual API key

# Define the language model with streaming enabled
llm = ChatOpenAI(model="gpt-4o-min", streaming=True)

# Define the State structure
class State(TypedDict):
    messages: Annotated[list, "add_messages"]

# Initialize and configure the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", lambda state: {"messages": [llm.invoke(state["messages"])]})
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# Define a function to execute the chatbot based on user input
def stream_graph_updates(user_input: str):
    """Stream the chatbot's response to a user input"""
    print(f"User: {user_input}")
    print("Agent: ", end="", flush=True)
    
    # Start streaming events from the graph with the user's input
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        # Process all events
        for node_name, node_updates in event.items():
            # Check if the event contains a message update
            if "messages" in node_updates:
                # For streaming output, we typically get content chunks
                for message in node_updates["messages"]:
                    if hasattr(message, "content"):
                        print(message.content, end="", flush=True)
    print()  # Add a new line after the response completes

# Example usage
user_query = "Who is Ada Lovelace?"
stream_graph_updates(user_query)
```

## Visualizing Graph Structure

LangGraph allows you to visualize your application's architecture, which is helpful for debugging and explaining complex workflows.

```python
# Import necessary libraries
from IPython.display import Image
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI

# Define state and graph as in previous examples
class State(TypedDict):
    messages: Annotated[list, "add_messages"]

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", lambda state: {"messages": [llm.invoke(state["messages"])]})
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# Generate and display the diagram of the graph
try:
    # This creates a Mermaid diagram as a PNG image
    graph_visualization = graph.get_graph().draw_mermaid_png()
    display(Image(graph_visualization))
except Exception as e:
    print(f"Diagram generation failed: {e}")
    print("Note: Diagram generation requires additional dependencies like graphviz")
```

The visualization typically shows a simple flow for a basic chatbot:

![Example graph structure](./assets/graph-image.svg)

## Integrating External Tools

A powerful feature of LangGraph is the ability to incorporate external tools, allowing your chatbot to access and process information beyond its training data.

```python
# Import necessary libraries
import os
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from IPython.display import Image

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"  # Replace with your actual API key

# Define the language model
llm = ChatOpenAI(model="gpt-4o-min")

# Initialize Wikipedia API wrapper to fetch top one result
api_wrapper = WikipediaAPIWrapper(top_k_results=1)

# Create a Wikipedia query tool using the API wrapper
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
tools = [wikipedia_tool]

# Bind the Wikipedia tool to the language model
llm_with_tools = llm.bind_tools(tools)

# Define the State structure
class State(TypedDict):
    messages: Annotated[list, "add_messages"]

# Initialize StateGraph
graph_builder = StateGraph(State)

# Define chatbot function to respond with Wikipedia-enhanced LLM
def chatbot(state: State):
    """Process user input with tool-augmented LLM"""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Add chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)

# Create a ToolNode to handle tool calls and add it to the graph
# This node will process any tool invocations from the LLM
tool_node = ToolNode(tools=[wikipedia_tool])
graph_builder.add_node("tools", tool_node)

# Set up conditional routing:
# - If the LLM outputs a tool call, route to the tools node
# - Otherwise, route to END
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,  # This function examines the output to detect tool calls
    {
        "tool_call": "tools",  # If tool call detected, go to tools node
        "no_tool_call": END    # If no tool call, end the conversation
    }
)

# Connect tools back to chatbot for further processing
graph_builder.add_edge("tools", "chatbot")

# Connect START to chatbot to begin the flow
graph_builder.add_edge(START, "chatbot")

# Compile the graph
graph = graph_builder.compile()

# Visualize the graph with tools
display(Image(graph.get_graph().draw_mermaid_png()))
```

The resulting graph shows a more complex flow:
1. User input goes to the chatbot
2. If the chatbot needs external information, it calls the tools node
3. Results from tools flow back to the chatbot for incorporation into the response
4. Final answer is delivered to the user

## Implementing Conversation Memory

For a chatbot to maintain context across multiple interactions, we need to implement memory. LangGraph provides built-in support for this through its checkpointing system.

```python
# Import necessary libraries
import os
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langgraph.checkpoint.memory import MemorySaver

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"  # Replace with your actual API key

# Define the language model
llm = ChatOpenAI(model="gpt-4o-min")

# Setup tools as in the previous example
api_wrapper = WikipediaAPIWrapper(top_k_results=1)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
llm_with_tools = llm.bind_tools([wikipedia_tool])

# Define State and build graph as before
class State(TypedDict):
    messages: Annotated[list, "add_messages"]

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", lambda state: {"messages": [llm_with_tools.invoke(state["messages"])]})
graph_builder.add_node("tools", ToolNode(tools=[wikipedia_tool]))
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Initialize memory for persistent conversations
memory = MemorySaver()

# Compile the graph with memory checkpointing
graph = graph_builder.compile(checkpointer=memory)

# Set up a streaming function for a single user session
def stream_memory_responses(user_input: str):
    """Process user input with memory of previous exchanges"""
    print(f"User: {user_input}")
    print("Agent: ", end="", flush=True)
    
    # Configure this request to use a specific thread ID
    # This ensures all interactions are part of the same conversation
    config = {"configurable": {"thread_id": "single_session_memory"}}
    
    # Stream the events in the graph
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]}, 
        config=config
    ):
        # Process all events
        for node_name, node_updates in event.items():
            # Check if the event contains a message update
            if "messages" in node_updates:
                # Print each message or chunk
                for message in node_updates["messages"]:
                    if hasattr(message, "content"):
                        print(message.content, end="", flush=True)
    print()  # Add a new line after response completes

# Example of multi-turn conversation with memory
print("Example conversation with memory:")
stream_memory_responses("Tell me about the Eiffel Tower.")
stream_memory_responses("Who built it?")  # The model remembers we're discussing the Eiffel Tower
```

## Advanced Concepts

### Multi-User Support

For applications serving multiple users, you can create separate conversation threads:

```python
def get_user_response(user_id: str, user_input: str):
    """Handle input for a specific user with their own conversation memory"""
    config = {"configurable": {"thread_id": f"user_{user_id}"}}
    return graph.invoke(
        {"messages": [{"role": "user", "content": user_input}]}, 
        config=config
    )
```

### Handling Complex Workflows

For more sophisticated applications, you can create branching paths based on user intent:

```python
def route_by_intent(state: State):
    """Determine how to process the user's input based on detected intent"""
    intent_detector = llm.bind(
        prompt_template="Analyze this message and categorize it as one of: question, booking, complaint, feedback\nMessage: {input}\nCategory:"
    )
    user_message = state["messages"][-1].content
    intent = intent_detector.invoke({"input": user_message}).strip().lower()
    
    routing = {
        "question": "knowledge_base",
        "booking": "reservation_system",
        "complaint": "support_agent",
        "feedback": "feedback_collection"
    }
    
    return routing.get(intent, "general_response")
```
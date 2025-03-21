# multiple tools


```python 
# Import modules required for defining tool nodes
from langgraph.prebuilt import ToolNode

# List of tools
tools = [wikipedia_tool, palindrome_checker, historical_events]

# Pass the tools to the ToolNode()
tool_node = ToolNode(tools)

# Bind tools to the LLM
model_with_tools = llm.bind_tools(tools)
```

# Tools decision

```python 
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

workflow = StateGraph(MessagesState)
# Use MessagesState to define the state of the function
def should_continue(state: MessagesState):
    
    # Get the last message from the state
    last_message = state["messages"][-1]
    
    # Check if the last message includes tool calls
    if last_message.tool_calls:
        return "tools"
    
    # End the conversation if no tool calls are present
    return END

# Extract the last message from the history
def call_model(state: MessagesState):
    last_message = state["messages"][-1]

    # If the last message has tool calls, return the tool's response
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        
        # Return only the messages from the tool call
        return {"messages": [AIMessage(content=last_message.tool_calls[0]["response"])]}
    
    # Otherwise, proceed with a regular LLM response
    return {"messages": [model_with_tools.invoke(state["messages"])]}

# Add nodes for chatbot and tools
workflow.add_node("chatbot", call_model)
workflow.add_node("tools", tool_node)

# Define an edge connecting START to the chatbot
workflow.add_edge(START, "chatbot")

# Define conditional edges and route "tools" back to "chatbot"
workflow.add_conditional_edges("chatbot", should_continue, ["tools", END])
workflow.add_edge("tools", "chatbot")

# Set up memory and compile the workflow
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

display(Image(app.get_graph().draw_mermaid_png()))
```

# Usiug multi tools models

```python 
# Create input message with the user's query
def multi_tool_output(query):
    inputs = {"messages": [HumanMessage(content=query)]}
    
    # Stream messages and metadata from the chatbot application
    for msg, metadata in app.stream(inputs, config, stream_mode="messages"):
        
        # Check if the message has content and is not from a human
        if msg.content and not isinstance(msg, HumanMessage):
            print(msg.content, end="", flush=True)    
    print("\n")

# Call the chatbot with different tools
multi_tool_output("Is `may a moody baby doom a yam` a palindrome?")
multi_tool_output("What happened on 20th July, 1969?")
```

# Multiple queries

```python 
# Print the user query first for every interaction 
def user_agent_multiturn(queries):  
    for query in queries:
        print(f"User: {query}")
        
        # Stream through messages corresponding to queries, excluding metadata 
        print("Agent: " + "".join(msg.content for msg, metadata in app.stream(
                {"messages": [HumanMessage(content=query)]}, config, stream_mode="messages") 
            
            # Filter out the human messages to print agent messages
            if msg.content and not isinstance(query, HumanMessage)) + "/n")       

queries = ["Is `stressed desserts?` a palindrome?", "What about the word `kayak`?",
    "What happened on the May 8th, 1945?", "What about 9 November 1989?"]
user_agent_multiturn(queries)
```
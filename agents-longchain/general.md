# LangChain Agents: Core Concepts

LangChain is a framework for developing applications powered by language models. This document covers the fundamental concepts of LangChain agents and how to implement them effectively.

## Key Concepts

### Tools and Agents

- **Tool**: A function that an agent can use to perform a specific action or retrieve information from the external world.
- **Agent**: An LLM-powered system that takes user queries, determines which tools to use, and constructs a response.
- **ReAct Framework**: A methodology that combines **Rea**soning and **Act**ion. The agent follows a cycle of:
  1. **Reasoning**: Analyzing the query and deciding which tool to use
  2. **Acting**: Executing the chosen tool
  3. **Observing**: Interpreting the tool's output
  4. **Reasoning Again**: Planning the next step based on observations

## Basic Agent Implementation

Creating a basic ReAct agent involves defining tools and connecting them to a language model:

```python
# Import necessary libraries
from langchain.agents import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

# Define a simple tool
from langchain.tools import tool

@tool
def count_r_in_word(word: str) -> int:
    """Count how many times the letter 'r' appears in a word."""
    return word.lower().count('r')

# Initialize the language model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Create the agent
app = create_react_agent(model=model, tools=[count_r_in_word])

# Create a query
query = "How many r's are in the word 'Terrarium'?"

# Invoke the agent and store the response
response = app.invoke({"messages": [("human", query)]})

# Print the agent's response
print(response['messages'][-1].content)
```

## Creating Custom Tools

Custom tools allow agents to perform specific tasks tailored to your application. Each tool should:
- Have a clear, specific purpose
- Return the appropriate data type
- Include a descriptive docstring to guide the agent

```python
# Import necessary libraries
from langchain.tools import tool
import math

# Define this math function as a tool
@tool
def hypotenuse_length(input: str) -> float:
    """Calculates the length of the hypotenuse of a right-angled triangle given the lengths of the other two sides.
    
    Args:
        input (str): A comma-separated string containing two numbers representing the lengths of the two sides
              Example: "3, 4" or "5.5, 7.8"
    
    Returns:
        float: The length of the hypotenuse
    """
    
    # Split the input string to get the lengths of the triangle
    sides = input.split(',')
    
    # Convert the input values to floats, removing extra spaces
    a = float(sides[0].strip())
    b = float(sides[1].strip())
    
    # Calculate the hypotenuse using the Pythagorean theorem
    return math.sqrt(a**2 + b**2)
```

## Managing Conversational Context

Agents can maintain context across multiple interactions, enabling natural follow-up questions and references to previous exchanges:

```python
# Import necessary libraries
from langchain.agents import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import math

# Define the tool
@tool
def hypotenuse_length(input: str) -> float:
    """Calculates the length of the hypotenuse of a right-angled triangle given the lengths of the other two sides."""
    sides = input.split(',')
    a = float(sides[0].strip())
    b = float(sides[1].strip())
    return math.sqrt(a**2 + b**2)

# Initialize model and tools
model = ChatOpenAI(model="gpt-3.5-turbo")
tools = [hypotenuse_length]

# Create the initial query
query = "What is the value of the hypotenuse for a triangle with sides 3 and 5?"

# Create the ReAct agent
app = create_react_agent(model, tools=tools)

# Invoke the agent with a query and store the messages
response = app.invoke({"messages": [("human", query)]})
message_history = response["messages"]

# Print the initial response
print(f"User: {query}")
print(f"Agent: {message_history[-1].content}\n")

# Create a follow-up query
new_query = "What about one with sides 12 and 14?"

# Invoke the app with the full message history
response = app.invoke({"messages": message_history + [("human", new_query)]})

# Extract and print the follow-up response
print(f"User: {new_query}")
print(f"Agent: {response['messages'][-1].content}")
```

## Best Practices for LangChain Agents

1. **Tool Design**
   - Keep tools focused on a single responsibility
   - Provide clear documentation with examples
   - Handle edge cases and errors gracefully

2. **Agent Prompting**
   - Use clear and specific instructions
   - Break complex tasks into smaller steps
   - Provide examples of expected inputs and outputs

3. **Performance Optimization**
   - Cache results for expensive operations
   - Use streaming for better user experience
   - Consider batching requests when appropriate

4. **Testing and Evaluation**
   - Test with a diverse set of inputs
   - Evaluate against a set of benchmark tasks
   - Gather user feedback to identify improvement areas

## Advanced Agent Features

- **Tool Retrieval**: Dynamically selecting tools based on the query
- **Multi-step Planning**: Breaking down complex tasks into simpler subtasks
- **Self-reflection**: Evaluating and correcting previous steps
- **Memory Management**: Efficiently handling long conversations

These concepts form the foundation for building more sophisticated agent-based applications that can perform complex tasks involving reasoning and interaction with external tools.

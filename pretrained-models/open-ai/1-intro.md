# 1 - Introduction to OpenAI API

## Key Components of the OpenAI API

* **Models**: Different AI models with varying capabilities (GPT-4o, GPT-4o-mini, etc.)
* **Data**: The information you provide to the model (prompts, messages, context)
* **Parameters**: Settings that control model behavior (temperature, max_tokens, etc.)

## Working Programmatically with OpenAI

When using the OpenAI API in your applications, be aware of:

* **Usage Cost**: Determined by the model requested and the size of input and output tokens
* **Rate Limits**: Limits on how many requests you can make in a given time period
* **Token Limits**: Maximum combined input and output tokens per request

### Model Comparison Table

| Model | Strengths | Use Cases | Context Window | Relative Cost |
|-------|-----------|-----------|---------------|---------------|
| GPT-4o | Most capable, multimodal | Complex reasoning, vision tasks | 128K tokens | Higher |
| GPT-4o-mini | Good balance of capability/cost | General purpose tasks | 128K tokens | Medium |
| GPT-3.5 Turbo | Fast, cost-effective | Simple tasks, high volume | 16K tokens | Lower |

## Basic API Usage Example

```python 
# Import the OpenAI package
from openai import OpenAI

# Create the OpenAI client with your API key
client = OpenAI(api_key="<OPENAI_API_TOKEN>")

# Create a request to the Chat Completions endpoint
response = client.chat.completions.create(
  model="gpt-4o-mini",  # Specify which model to use
  messages=[
    {"role": "user", 
     "content": "Write a polite reply accepting an AI Engineer job offer."}
  ]
)

# Print the response from the model
print(response.choices[0].message.content)
```

## Text Generation and Transformation

The OpenAI API allows you to generate and transform text in various ways. Key parameters include:

### Temperature

Temperature controls the randomness of the model's output:
* **Low temperature (0.0-0.3)**: More deterministic, focused, and predictable responses
* **Medium temperature (0.4-0.7)**: Balance of creativity and coherence
* **High temperature (0.8-1.0)**: More random, creative, and diverse responses

### Max Tokens

The `max_tokens` parameter limits the length of the model's response:
* Setting an appropriate limit helps control costs and response size
* If set too low, responses may be cut off mid-sentence

### Example: Text Summarization

```python 
from openai import OpenAI

# Initialize the client
client = OpenAI(api_key="<OPENAI_API_TOKEN>")

# Define the text to be summarized
prompt="""Summarize the following text into two concise bullet points:
Investment refers to the act of committing money or capital to an enterprise with the expectation of obtaining an added income or profit in return. There are a variety of investment options available, including stocks, bonds, mutual funds, real estate, precious metals, and currencies. Making an investment decision requires careful analysis, assessment of risk, and evaluation of potential rewards. Good investments have the ability to produce high returns over the long term while minimizing risk. Diversification of investment portfolios reduces risk exposure. Investment can be a valuable tool for building wealth, generating income, and achieving financial security. It is important to be diligent and informed when investing to avoid losses."""

# Create a request to the Chat Completions endpoint
response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[{"role": "user", "content": prompt}],
  max_tokens=400,    # Limit the response length
  temperature=0.5    # Balanced creativity and determinism
)

# Display the model's response
print(response.choices[0].message.content)
```

Example output:
```shell
- Investment involves committing capital to various options (like stocks, bonds, and real estate) with the expectation of generating income or profit, requiring careful analysis of risks and rewards.  
- Diversifying investment portfolios helps minimize risk, making informed investment decisions crucial for building wealth and achieving financial security.
```

## Classification Tasks

The OpenAI API excels at classification tasks such as:
* Label assignment
* Categorization
* Sentiment analysis

### Prompting Techniques

* **Zero-shot**: No examples provided, just instructions
* **One-shot**: A single example provided to guide the model
* **Few-shot**: Multiple examples to establish a pattern

### Example: Sentiment Classification

```python  
from openai import OpenAI

# Initialize the client
client = OpenAI(api_key="<OPENAI_API_TOKEN>")

# Define a multi-line prompt to classify sentiment
prompt = """Classify sentiment in the following statements as positive, negative, or neutral:
1. Unbelievably good!
2. Shoes fell apart on the second use.
3. The shoes look nice, but they aren't very comfortable.
4. Can't wait to show them off!"""

# Create a request to the Chat Completions endpoint
response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[{"role": "user", "content": prompt}],
  max_tokens=100,
  temperature=0.1  # Low temperature for more deterministic classification
)

print(response.choices[0].message.content)
```

## Conversational AI (Chat)

The OpenAI Chat API enables multi-turn conversations by maintaining a message history.

### Key Components:
* **System Message**: Sets the behavior/personality of the assistant
* **User Messages**: What the end-user says to the assistant
* **Assistant Messages**: The model's previous responses

### Example: Math Tutoring Conversation

```python 
from openai import OpenAI

# Initialize the client
client = OpenAI(api_key="<OPENAI_API_TOKEN>")

# Initialize conversation with a system message
messages = [{"role": "system", "content": "You are a helpful math tutor."}]
user_msgs = ["Explain what pi is.", "Summarize this in two bullet points."]

# Simulate a conversation
for q in user_msgs:
    print("User: ", q)
    
    # Add the user's message to the conversation history
    user_dict = {"role": "user", "content": q}
    messages.append(user_dict)
    
    # Send the entire conversation history to the API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,  # The full conversation history
        max_tokens=100
    )
    
    # Get the assistant's response and add it to the conversation history
    assistant_message = response.choices[0].message.content
    assistant_dict = {"role": "assistant", "content": assistant_message}
    messages.append(assistant_dict)
    
    print("Assistant: ", assistant_message, "\n")
```


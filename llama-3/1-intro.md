# Introduction to Llama 3

This guide covers the basics of working with Llama 3 models using the `llama-cpp-python` library. Llama 3 is an open-source large language model (LLM) developed by Meta that can be run locally on your machine.

## Setting Up

Before you begin, ensure you have the library installed:

```bash
pip install llama-cpp-python
```

You'll also need to download a Llama 3 model file (GGUF format) from [Hugging Face](https://huggingface.co/models?search=llama+3+gguf).

## Basic Interaction

The following example shows how to load a model and generate a simple text completion:

```python
# Import the necessary libraries
import os
from llama_cpp import Llama

# Set the path to your downloaded model
llama_path = "path/to/your/llama-3-model.gguf"  # Replace with your actual model path

# Initialize the Llama model
llm = Llama(model_path=llama_path)

# Ask a simple question
question = "What is the most used database for data storage?"
response = llm(question)
print(response)
```

The output will be a dictionary containing various information:

```shell
{'id': 'cmpl-7e0ac270-da59-433e-8a0f-4bd85ebd488a', 'object': 'text_completion', 'created': 1742598763, 'model': '/home/ubuntu/models/Llama-3.2-1B-Instruct-Q3_K_L.gguf', 'choices': [{'text': ' In the world of SQL Server, the most used database is SQL Server Management Studio', 'index': 0, 'logprobs': None, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 11, 'completion_tokens': 16, 'total_tokens': 27}}
```

To get just the generated text:

```python
print(response['choices'][0]['text'])
```

## Tuning Response Parameters

You can control various aspects of the model's output:

* `temperature`: Controls randomness (higher = more creative, lower = more deterministic)
* `top_k`: Limits vocabulary to top K most likely tokens
* `top_p`: Nucleus sampling (only considers tokens with cumulative probability < p)
* `max_tokens`: Maximum number of tokens to generate

Example with tuning parameters:

```python 
output = llm(
    "What are the symptoms of strep throat?", 
    # Set the model parameters 
    temperature=1.0,  # Note: Using = instead of : for parameters
    max_tokens=10,    # Limit response length
    top_k=2,          # Restrict word choices
    top_p=0.6         # Note: Fixed typo from op_p to top_p
)

print(output['choices'][0]['text'])
```

## Assigning System Roles

You can use system prompts to guide the model's behavior:

```python 
# Import the necessary libraries
from llama_cpp import Llama

# Initialize the model (if not already done)
# llm = Llama(model_path=llama_path)

# Add a system message to the conversation list
conv = [
    {
        "role": "system",
        "content": "You are a helpful and professional customer support assistant for an internet service provider. If the question or instruction doesn't relate to internet service, quote the response: 'Sorry, I can't answer that.'"
    },
    {
        "role": "user",
        "content": "Help me decide which stocks to invest in."
    }
]

# Generate a response
result = llm.create_chat_completion(messages=conv, max_tokens=15)

# Extract the model response from the result object
assistant_content = result["choices"][0]["message"]["content"]
print(assistant_content)  # Should output: "Sorry, I can't answer that."
```

## Prompt Engineering Techniques

### One-Shot Learning

Provide a single example to guide the model's response format:

```python 
# Import the necessary libraries (if not already imported)
# from llama_cpp import Llama

# Add formatting to the prompt
prompt = """
Instruction: Explain the concept of gravity in simple terms.
Question: What is gravity?
Answer:
"""

# Send the prompt to the model
output = llm(prompt, max_tokens=15, stop=["Question:"]) 
print(output['choices'][0]['text'])
```

### Few-Shot Learning

Provide multiple examples to establish a pattern:

```python 
# Multiple examples to establish a pattern
prompt = """Review 1: I ordered from this place last night, and I'm impressed! 
Sentiment 1: Positive,
Review 2: My order was delayed by over an hour without any updates. Disappointing!  
Sentiment 2: Negative,
Review 3: The food quality is top-notch. Highly recommend! 
Sentiment 3: Positive,
Review 4: Delicious food, and excellent customer service! 
Sentiment 4:"""

# Send the prompt to the model with a stop word
output = llm(prompt, max_tokens=2, stop=["Review"]) 
print(output['choices'][0]['text'])  # Should output: " Positive"
```

## Structured Output

You can request JSON-formatted responses:

```python 
# First, create a conversation with a user query
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning and how is it used?"}
]

# Request a JSON-formatted response
output = llm.create_chat_completion(
    messages=messages,
    response_format={
        "type": "json_object",
        "schema": {
            "type": "object",
            # Set the properties of the JSON fields and their data types
            "properties": {"Question": {"type": "string"}, "Answer": {"type": "string"}}
        }
    }
)

print(output['choices'][0]['message']['content'])
```

## Maintaining Conversation Context

This class helps maintain conversation history:

```python 
class Conversation:
    def __init__(self, llm: Llama, system_prompt='', history=None):
        self.llm = llm
        self.system_prompt = system_prompt
        # Fix potential mutable default argument issue
        self.history = [{"role": "system", "content": self.system_prompt}]
        if history:
            self.history.extend(history)

    def create_completion(self, user_prompt=''):
        # Add the user prompt to the history
        self.history.append({"role": "user", "content": user_prompt})
        # Send the history messages to the LLM
        output = self.llm.create_chat_completion(messages=self.history)
        conversation_result = output['choices'][0]['message']
        # Append the conversation_result to the history
        self.history.append(conversation_result)
        return conversation_result['content']

# Example usage
system_instruction = "You are a travel expert that recommends a travel destination based on a prompt. Return the location name only as 'City, Country'."
chatbot = Conversation(llm, system_prompt=system_instruction)

# Ask for the initial travel recommendation
first_recommendation = chatbot.create_completion("Recommend a Spanish-speaking city.")
print(first_recommendation)

# Add an additional request to update the recommendation
second_recommendation = chatbot.create_completion("A different city in the same country")
print(second_recommendation)
```

Output: 
```shell
Malaga, Spain
Barcelona, Spain
```

## Best Practices

1. **Start with a lower temperature** (0.1-0.3) for factual responses and higher (0.7-1.0) for creative tasks
2. **Use system prompts** to guide the model's behavior and set constraints
3. **Experiment with few-shot examples** to improve response formatting
4. **Maintain conversation context** for coherent multi-turn interactions
5. **Consider model size and resource constraints** when running locally
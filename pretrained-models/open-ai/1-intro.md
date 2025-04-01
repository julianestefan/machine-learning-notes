# Introduction to OpenAI API

This guide provides an overview of the OpenAI API, its capabilities, and how to use it in various applications.

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
import os

# Create the OpenAI client with your API key
# Better to use environment variables for API keys
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "<YOUR_API_KEY>")

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
import os

# Initialize the client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "<YOUR_API_KEY>")

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
import os

# Initialize the client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "<YOUR_API_KEY>")

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

Expected output:
```
1. Positive
2. Negative
3. Mixed/Neutral
4. Positive
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
import os

# Initialize the client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "<YOUR_API_KEY>")

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

## Content Moderation

OpenAI provides a dedicated API for content moderation that helps identify potentially harmful or unsafe content.

```python 
from openai import OpenAI
import os

# Initialize the client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "<YOUR_API_KEY>")

# Create a request to the Moderation endpoint
response = client.moderations.create(
    model="text-moderation-latest",
    input="My favorite book is To Kill a Mockingbird."
)

# Print the category scores
print(response.results[0].category_scores)
```

Output:
```shell
CategoryScores(
    harassment=5.243551186140394e-06,
    harassment_threatening=1.1516095810293336e-06,
    hate=4.767837526742369e-05,
    hate_threatening=3.2021056028952444e-08,
    self_harm=9.466615438213921e-07,
    self_harm_instructions=5.426785065765216e-08,
    self_harm_intent=1.5536235764557205e-07,
    sexual=3.545879735611379e-06,
    sexual_minors=1.1304399549771915e-06,
    violence=0.0001064608441083692,
    violence_graphic=1.086988686438417e-05
    # Additional scores may be returned depending on the API version
)
```

### How to Use Moderation Results

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "<YOUR_API_KEY>")

def check_content_safety(text):
    """
    Check if text contains content that violates content policy
    
    Args:
        text (str): The text to check
        
    Returns:
        bool: True if content is safe, False otherwise
    """
    response = client.moderations.create(
        model="text-moderation-latest",
        input=text
    )
    
    # Check if any category is flagged
    return not response.results[0].flagged

# Example usage
user_input = "Let's discuss the history of space exploration."
if check_content_safety(user_input):
    print("Content is safe to process")
else:
    print("Content may violate content policy")
```

## Audio Processing

### Speech to Text Transcription

The Whisper model can transcribe spoken language in audio files to text.

```python 
from openai import OpenAI
import os

# Initialize the client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "<YOUR_API_KEY>")

# Open the audio file
with open("audio-sample.mp3", "rb") as audio_file:
    # Create a transcript from the audio file
    response = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file
    )

    # Extract and print the transcript text
    print(response.text)
```

### Audio Translation

For translating speech in one language to text in English:

```python 
from openai import OpenAI
import os

# Initialize the client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "<YOUR_API_KEY>")

# Open the audio file
with open("foreign-language-audio.wav", "rb") as audio_file:
    # Optional: Provide a prompt to guide translation
    prompt = "This is a lecture about climate change"
    
    # Create a translation from the audio file
    response = client.audio.translations.create(
        model="whisper-1",
        file=audio_file,
        prompt=prompt  # Optional contextual hint
    )
    
    print(response.text)
```

## Building Multimodal Applications

### Combining Models for Advanced Workflows

You can combine different API endpoints to create powerful workflows:

```python 
from openai import OpenAI
import os

# Initialize the client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "<YOUR_API_KEY>")

def process_audio_and_analyze(audio_file_path):
    """
    Transcribe audio and then analyze the content
    
    Args:
        audio_file_path (str): Path to the audio file
        
    Returns:
        dict: Analysis results
    """
    # Step 1: Transcribe the audio file
    with open(audio_file_path, "rb") as audio_file:
        transcription_response = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
    
    transcript = transcription_response.text
    print(f"Transcription: {transcript}")
    
    # Step 2: Analyze the transcribed text
    analysis_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes transcripts."},
            {"role": "user", "content": f"Analyze this transcript and identify the main topics, sentiment, and any action items: \n\n{transcript}"}
        ],
        temperature=0.3
    )
    
    analysis = analysis_response.choices[0].message.content
    print(f"\nAnalysis: {analysis}")
    
    return {
        "transcript": transcript,
        "analysis": analysis
    }

# Example usage
results = process_audio_and_analyze("meeting-recording.mp3")
```

## Error Handling and Best Practices

When working with the OpenAI API, proper error handling is essential:

```python
from openai import OpenAI
from openai.types.error import APIError, RateLimitError
import time
import os

# Initialize the client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "<YOUR_API_KEY>")

def get_completion_with_retry(prompt, max_retries=3, backoff_factor=2):
    """
    Get completion with exponential backoff retry logic
    
    Args:
        prompt (str): The prompt to send to the API
        max_retries (int): Maximum number of retries
        backoff_factor (int): Factor to increase wait time between retries
        
    Returns:
        str: The completion text
    """
    retries = 0
    while retries <= max_retries:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
        
        except RateLimitError:
            if retries == max_retries:
                raise
            
            # Calculate wait time with exponential backoff
            wait_time = backoff_factor ** retries
            print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
            
        except APIError as e:
            if retries == max_retries:
                raise
            
            # Only retry on 5xx errors, which indicate server issues
            if e.status_code and 500 <= e.status_code < 600:
                wait_time = backoff_factor ** retries
                print(f"Server error {e.status_code}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
            else:
                # Don't retry on 4xx errors, which indicate client issues
                raise

# Example usage
try:
    result = get_completion_with_retry("Explain quantum computing in simple terms.")
    print(result)
except Exception as e:
    print(f"Error: {e}")
```

## Cost Optimization Strategies

Here are strategies to optimize API usage costs:

| Strategy | Description | Implementation |
|----------|-------------|----------------|
| Token Counting | Monitor and limit tokens used | Use tiktoken library to count tokens before sending |
| Caching | Cache responses for identical requests | Implement a caching layer using Redis or another caching system |
| Model Selection | Use the smallest model that meets requirements | Start with GPT-3.5 Turbo before trying GPT-4o |
| Batching | Combine multiple requests into one | Group similar tasks when possible |
| Chunking | Break large documents into smaller pieces | Process long texts in manageable segments |

### Example: Token Counting

```python
import tiktoken
from openai import OpenAI
import os

# Initialize the client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "<YOUR_API_KEY>")

def num_tokens_from_string(string, model="gpt-4o-mini"):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def process_with_token_awareness(prompt, max_tokens=1000):
    """Process text with awareness of token limits"""
    # Count tokens in the prompt
    prompt_tokens = num_tokens_from_string(prompt)
    
    if prompt_tokens > max_tokens:
        print(f"Warning: Prompt has {prompt_tokens} tokens, exceeding the {max_tokens} limit.")
        # Truncate or summarize the prompt here if needed
        return None
    
    print(f"Prompt contains {prompt_tokens} tokens")
    
    # Calculate a safe limit for response tokens
    response_token_limit = min(4000, 8192 - prompt_tokens)
    
    # Make the API call with appropriate limits
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=response_token_limit
    )
    
    return response.choices[0].message.content

# Example usage
long_text = "..." # Your long text here
result = process_with_token_awareness(long_text, max_tokens=2000)
```

## Conclusion

The OpenAI API provides powerful capabilities for a wide range of natural language processing tasks. By understanding its components, parameters, and best practices, you can effectively integrate AI capabilities into your applications while optimizing for cost and performance.

As you become more familiar with the API, you can combine different endpoints and techniques to create increasingly sophisticated solutions for text generation, analysis, classification, and multimodal applications.
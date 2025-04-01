# Prompt Engineering: Best Practices and Techniques

Effective prompt engineering is critical for getting high-quality outputs from Large Language Models (LLMs). This guide explores various prompt engineering techniques with practical examples using OpenAI's API.

## Setup

First, let's set up our environment with the necessary imports and a helper function that we'll use throughout this guide:

```python
# Import the OpenAI library
from openai import OpenAI
import os

# Initialize the OpenAI client
# You can set your API key as an environment variable or provide it directly
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "<YOUR_API_KEY>")

def get_response(prompt, system_message=None, temperature=0):
    """
    Get a response from OpenAI's API.
    
    Args:
        prompt (str): The user prompt to send to the API
        system_message (str, optional): Optional system message to guide the model
        temperature (float, optional): Controls randomness. Defaults to 0 for deterministic outputs.
        
    Returns:
        str: The model's response text
    """
    messages = []
    
    # Add system message if provided
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    # Add user prompt
    messages.append({"role": "user", "content": prompt})
    
    # Create a request to the chat completions endpoint
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # You can change this to other models as needed
        messages=messages,
        temperature=temperature
    )
    
    return response.choices[0].message.content
```

## Principles of Effective Prompts

Good prompts typically follow these key principles:

| Principle | Description | Example |
|-----------|-------------|---------|
| Clarity | Use precise language and avoid ambiguity | "List 5 benefits of exercise" instead of "Tell me about exercise" |
| Specificity | Include detailed instructions | "Write a 100-word summary of quantum computing in simple terms" |
| Structure | Organize the prompt logically | "First explain X, then compare it with Y" |
| Action Verbs | Use clear directives | "Analyze" instead of "Look at" |
| Output Format | Specify the desired format | "Format your answer as a bulleted list" |
| Context | Provide relevant background information | "As a financial advisor helping a recent college graduate..." |

### Example: Structured Prompting with Clear Instructions

```python
# Story completion example with proper structure
story_beginning = "The old lighthouse stood at the edge of the cliff, its light flickering in the storm..."

# Create a prompt that completes the story with specific instructions
prompt = f"""
Complete the story inside the triple backticks with approximately 150 words. 
Include atmospheric details and a surprising ending.
```{story_beginning}```
"""

# Get the generated response 
response = get_response(prompt)

print("\nOriginal story beginning:\n", story_beginning)
print("\nCompleted story:\n", response)
```

## Structured Outputs and Conditional Prompts

### Tables

When requesting tabular data, specify the exact columns and formatting you want:

```python
# Create a prompt that generates a formatted table
prompt = """
Generate a table of 10 must-read science fiction books, with these exact columns:
- Title
- Author
- Year Published
- Key Themes
- Why It's Essential

Format the table with markdown syntax.
"""

response = get_response(prompt)
print(response)
```

### Structured Text Analysis

```python
# Sample text to analyze
text = "The artificial intelligence revolution has transformed industries from healthcare to finance, creating both opportunities and challenges for society."

# Create the instructions with specified output format
prompt = f"""
Analyze the text provided within triple backticks and output the following:

- Language: Identify the language of the text
- Main Topic: Extract the primary subject of discussion
- Key Entities: List any organizations, technologies, or concepts mentioned
- Tone: Assess whether the tone is neutral, positive, or negative

Format your response using the structure above.

```{text}```
"""

response = get_response(prompt)
print(response)
```

### Conditional Outputs

Instruct the model to provide different outputs based on specific conditions:

```python
# Sample text for conditional analysis
text = "Machine learning systems are increasingly being deployed in critical infrastructure."

# Create conditional instructions
prompt = f"""
You will be provided with a text delimited by triple backticks. 
1. Determine the language of the text
2. Count the number of words it contains
3. If the text has more than 10 words, generate a summary in 5 words or less
4. If the text has 10 words or fewer, respond with 'No summary needed'

Format your response as:
- Language: <language>
- Word Count: <number>
- Summary: <brief summary or 'No summary needed'>

```{text}```
"""

response = get_response(prompt)
print(response)
```

## Few-Shot Prompting

Few-shot prompting provides examples to help the model understand the desired pattern of response:

```python
# Sentiment analysis with few-shot examples
prompt = """
Classify the sentiment of each product review as positive (1), negative (-1), or neutral (0).

Examples:
Review: "The product quality exceeded my expectations."
Sentiment: 1

Review: "I had a terrible experience with this product's customer service."
Sentiment: -1

Review: "The product arrived on time and works as described."
Sentiment: 1

Now classify this review:
Review: "The design is sleek but the battery life is disappointing."
Sentiment:
"""

response = get_response(prompt)
print(f"Sentiment classification: {response}")
```

Few-shot prompting is particularly effective for:
- Classification tasks
- Text formatting tasks
- Style imitation
- Complex reasoning patterns

## Multi-Step Prompting

Breaking complex tasks into smaller, sequential steps:

```python
# Create a prompt detailing steps for vacation planning
prompt = """
Create a comprehensive plan for a beach vacation by following these steps:

Step 1: Suggest four potential beach destinations with different characteristics (family-friendly, romantic, adventure-focused, budget).

Step 2: For each location, list two accommodation options (one luxury, one budget-friendly) with approximate price ranges.

Step 3: Propose three activities unique to each destination.

Step 4: Generate a pros and cons evaluation for each destination considering factors like cost, accessibility, and attractions.
"""

response = get_response(prompt)
print(response)
```

## Chain of Thought Prompting

Chain of Thought (CoT) prompting encourages the model to work through a problem step by step:

```python
# Create a chain-of-thought prompt for a math problem
prompt = """
Solve this problem step by step:

If my friend is 20 years old, and her father is currently twice her age, how old will her father be in 10 years?

Walk through your reasoning process before giving the final answer.
"""

response = get_response(prompt)
print(response)
```

### One-Shot Chain of Thought

Providing an example of the reasoning process can significantly improve results:

```python
# Define the example with reasoning steps
prompt = """
Example:
Q: If a shirt originally costs $25 and is on sale for 20% off, what is the sale price?
A: To find the sale price, I need to:
   1. Calculate the discount amount: $25 × 20% = $25 × 0.2 = $5
   2. Subtract the discount from the original price: $25 - $5 = $20
   So the sale price is $20.

Now solve this problem with similar step-by-step reasoning:
Q: A car travels 150 miles in 2.5 hours. What is its average speed in miles per hour?
A:
"""

response = get_response(prompt)
print(response)
```

## Self-Consistency Prompting

Self-consistency prompting generates multiple reasoning paths and selects the most consistent answer:

```python
# Create a prompt that encourages multiple solution paths
prompt = """
Imagine three different experts are solving this math problem independently. 
Each expert should show their complete reasoning process.
The final answer should be determined by comparing all three approaches.

Problem: A store has 50 devices, of which 60% are mobile phones and the rest are laptops. 
During the day, 3 customers each buy one mobile phone, and one of those customers also buys a laptop.
The store then receives a delivery of 10 new laptops and 5 new mobile phones.
How many laptops and mobile phones does the store have at the end of the day?
"""

# Use a higher temperature to encourage diversity in reasoning paths
response = get_response(prompt, temperature=0.7)
print(response)
```

## Practical Applications

### Text Summarization

```python
long_article = """
[Your long text content would go here. This could be a news article, research paper, etc.]
"""

prompt = f"""
Summarize the text below in three different ways:
1. A single sentence summary (15-20 words)
2. A paragraph summary (50-75 words)
3. A bullet-point summary with 3-5 key points

Text to summarize:
```{long_article}```
"""

response = get_response(prompt)
print(response)
```

### Text Transformation

```python
text_to_transform = "The company announced a 15% increase in quarterly revenue, exceeding market expectations."

prompt = f"""
Transform the following text in these three different ways:
1. Translate to Spanish
2. Rewrite in a more formal tone
3. Expand with additional relevant details (2-3 sentences)

Original text: "{text_to_transform}"
"""

response = get_response(prompt)
print(response)
```

### Code Generation and Explanation

```python
prompt = """
1. Write a Python function that calculates the Fibonacci sequence up to n terms.
2. Include docstrings and comments explaining the code.
3. Provide an example of how to call this function.
4. Explain the time and space complexity of your implementation.
"""

response = get_response(prompt)
print(response)
```

## Chatbot Development

System prompts are powerful tools for defining a chatbot's behavior and personality:

```python
# Define a specific persona for a customer service chatbot
system_message = """
You are a customer service representative for TechGadgets, an electronics retailer.
Your name is Alex. You should be helpful, knowledgeable about electronics, and polite.
You should not make up information about products. If you don't know something, 
acknowledge that and offer to connect the customer with a human representative.
"""

user_question = "Can you help me figure out which smartphone has the best camera quality under $500?"

response = get_response(user_question, system_message=system_message)
print(f"Chatbot: {response}")
```

### Incorporating Context

Providing relevant context helps chatbots give more accurate and helpful responses:

```python
# Service description context
service_description = """
MyPersonalDelivery is a premium same-day delivery service for restaurants and retailers.
We offer:
- Delivery within 2 hours guaranteed
- Real-time tracking of your orders
- Insulated packaging to keep food at optimal temperature
- No minimum order requirement
- 24/7 customer support
- Special handling for fragile items
- Eco-friendly packaging options
"""

system_message = f"""
You are a customer service representative for a delivery service.
The following text describes the services offered:
```{service_description}```
Base your responses only on the information provided above.
"""

user_question = "What benefits does MyPersonalDelivery offer compared to standard delivery services?"

response = get_response(user_question, system_message=system_message)
print(f"Response: {response}")
```

## Advanced Techniques

### Combining Methods

For complex tasks, combining multiple prompt engineering techniques often yields the best results:

```python
prompt = """
I need help analyzing a company's quarterly financial results.

First, use this example as a guide:
Example Analysis:
Revenue: $5.2M (+15% YoY)
Analysis: Strong growth driven by new product lines and expanded market share.

Now, step by step:
1. Extract the key financial metrics from the provided data
2. Compare with industry benchmarks
3. Identify strengths and areas of concern
4. Provide 3 strategic recommendations based on the analysis

Data to analyze:
Company: TechInnovate Inc.
Q2 2023 Revenue: $3.8M (previous year: $3.2M)
Profit margin: 18% (industry average: 22%)
Customer acquisition cost: $120 (down from $150)
New product revenue contribution: 35%
"""

response = get_response(prompt)
print(response)
```

### Iterative Refinement

Sometimes the best approach is to build prompts iteratively, refining based on the responses:

```python
# Initial prompt
initial_prompt = "Write a brief description of quantum computing"

initial_response = get_response(initial_prompt)
print("Initial response:\n", initial_response)

# Refined prompt based on initial response
refined_prompt = f"""
Based on this description of quantum computing:
"{initial_response}"

Please enhance it by:
1. Adding an analogy that would help a high school student understand
2. Including 2-3 practical applications being developed currently
3. Explaining how it differs from classical computing in simple terms
"""

refined_response = get_response(refined_prompt)
print("\nRefined response:\n", refined_response)
```

## Conclusion

Effective prompt engineering is both an art and a science. The techniques covered in this guide—from structured outputs to chain-of-thought reasoning—provide a foundation for crafting prompts that elicit precise, useful responses from language models. As these models continue to evolve, so too will the strategies for prompting them effectively.

Remember that experimentation is key: what works for one task may not work for another, and finding the optimal prompt often requires iteration and refinement.

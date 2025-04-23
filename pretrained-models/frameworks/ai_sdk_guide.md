# Vercel AI SDK Guide

This guide provides an overview of the Vercel AI SDK, focusing on its core concepts, model integration, and building interactive UI components based on the provided documentation.

## 1. Introduction to Vercel AI SDK

The Vercel AI SDK is a TypeScript toolkit designed to help developers build AI-powered applications, particularly integrating Large Language Models (LLMs) into frameworks like React, Next.js, Vue, Svelte, and Node.js.

**Core Goals:**

*   **Abstract Away Provider Differences:** Provides a unified API to interact with various LLM providers, simplifying the process of switching between models.
*   **Simplify UI Development:** Eliminates boilerplate code for building interactive AI experiences like chatbots.
*   **Enable Rich Interactions:** Allows developers to go beyond simple text output and generate interactive components. 

**Key Components:**

*   **AI SDK Core:** Provides the central, unified API for calling LLMs.
*   **AI SDK UI:** Offers abstractions (primarily hooks) for building chat, completion, and assistant interfaces in frontend frameworks.

## 2. Core Concepts: Unified API

AI SDK Core allows you to interact with different LLMs using consistent functions like `generateText`.

**Example: Generating Text with OpenAI GPT-4.5**

```typescript
import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';

const { text } = await generateText({
  model: openai('gpt-4.5-preview'), // Specify the OpenAI model
  prompt: 'Explain the concept of quantum entanglement.',
});

console.log(text);
```

**Example: Generating Text with DeepSeek R1**

```typescript
import { deepseek } from '@ai-sdk/deepseek';
import { generateText } from 'ai';

const { reasoning, text } = await generateText({
  model: deepseek('deepseek-reasoner'), // Specify the DeepSeek model
  prompt: 'Explain quantum entanglement.',
});

console.log('Reasoning:', reasoning);
console.log('Text:', text);
```

## 3. LLM Provider Integration

The SDK supports various providers. Switching between them involves changing the provider import and the model instance.

**Example: Switching to DeepSeek via Fireworks**

```typescript
import { fireworks } from '@ai-sdk/fireworks';
import {
  generateText,
  wrapLanguageModel,
  extractReasoningMiddleware,
} from 'ai';

// Middleware to extract reasoning tokens
const enhancedModel = wrapLanguageModel({
  model: fireworks('accounts/fireworks/models/deepseek-r1'), // Use Fireworks provider
  middleware: extractReasoningMiddleware({ tagName: 'think' }),
});

const { reasoning, text } = await generateText({
  model: enhancedModel,
  prompt: 'Explain quantum entanglement.',
});
```

**Example: Switching to DeepSeek via Groq**

```typescript
import { groq } from '@ai-sdk/groq';
import {
  generateText,
  wrapLanguageModel,
  extractReasoningMiddleware,
} from 'ai';

// Middleware to extract reasoning tokens
const enhancedModel = wrapLanguageModel({
  model: groq('deepseek-r1-distill-llama-70b'), // Use Groq provider
  middleware: extractReasoningMiddleware({ tagName: 'think' }),
});

const { reasoning, text } = await generateText({
  model: enhancedModel,
  prompt: 'Explain quantum entanglement.',
});
```

## 4. Embeddings

Embeddings represent text or other data as numerical vectors, allowing for semantic understanding and similarity calculations. They are fundamental for tasks like RAG and semantic search
### 4.1. Generating Embeddings

AI SDK Core provides functions to generate embeddings for single values or batches of values.

**Single Value (`embed`):**

```typescript
import { embed } from 'ai';
import { openai } from '@ai-sdk/openai';

// Select an embedding model from a provider
const embeddingModel = openai.embedding('text-embedding-3-small');

// Generate embedding for a single string
const { embedding, usage } = await embed({
  model: embeddingModel,
  value: 'sunny day at the beach',
});

console.log('Embedding Vector (first 5 values):', embedding.slice(0, 5));
console.log('Token Usage:', usage); // e.g., { tokens: 6 }
```

**Batch Embedding (`embedMany`):**

Efficiently embed multiple values at once, often used when indexing documents for RAG.

```typescript
import { openai } from '@ai-sdk/openai';
import { embedMany } from 'ai';

const embeddingModel = openai.embedding('text-embedding-3-small');

const valuesToEmbed = [
  'sunny day at the beach',
  'rainy afternoon in the city',
  'snowy night in the mountains',
];

// 'embeddings' is an array of number[] vectors, in the same order as input
const { embeddings, usage } = await embedMany({
  model: embeddingModel,
  values: valuesToEmbed,
});

console.log(`Generated ${embeddings.length} embeddings.`);
console.log('Total Token Usage:', usage);
```

### 4.2. Calculating Similarity

Use the `cosineSimilarity` helper to measure the semantic similarity between two embedding vectors (closer to 1 means more similar).

```typescript
import { openai } from '@ai-sdk/openai';
import { cosineSimilarity, embedMany } from 'ai';

const embeddingModel = openai.embedding('text-embedding-3-small');

const { embeddings } = await embedMany({
  model: embeddingModel,
  values: ['hot dog', 'sunny day at the beach', 'frankfurter'],
});

const similarityHotDogBeach = cosineSimilarity(embeddings[0], embeddings[1]);
const similarityHotDogFrank = cosineSimilarity(embeddings[0], embeddings[2]);

console.log(`Similarity (Hot Dog vs Beach): ${similarityHotDogBeach.toFixed(4)}`);
console.log(`Similarity (Hot Dog vs Frankfurter): ${similarityHotDogFrank.toFixed(4)}`);
```

### 4.3. Settings and Configuration

Both `embed` and `embedMany` support settings like:
*   `maxRetries`: Control retry attempts on failure (default: 2).
*   `abortSignal`: Abort the request or set a timeout.
*   `headers`: Add custom HTTP headers to the request.

```typescript
// Example: Disabling retries
const { embedding } = await embed({
  model: embeddingModel,
  value: 'some text',
  maxRetries: 0,
});
```

## 5. Generating Structured Data: Getting Reliable JSON Output

**Learning Objectives:** By the end of this section, you should be able to:
*   Explain why getting structured data (like JSON) from LLMs can be challenging.
*   Use `generateObject` to reliably generate a complete, validated JSON object based on a schema.
*   Use `streamObject` to stream partial JSON objects for interactive UIs.
*   Understand different output strategies (`object`, `array`, `enum`, `no-schema`).
*   Recognize how to handle errors specific to object generation.

**The "Why": Moving Beyond Freeform Text**

Sometimes, you don't just want the LLM to *talk* about something; you need it to provide information in a specific, machine-readable format, like JSON. Imagine asking for recipe details, product specifications, or classified user feedback. Just getting back a paragraph of text makes it hard for your application to *use* that information reliably.

While you can prompt an LLM to "respond in JSON format," the output isn't always guaranteed to be valid JSON, nor will it necessarily follow the *exact* structure (schema) you need. This leads to brittle code that has to parse and validate potentially incorrect LLM output.

**The SDK Solution: `generateObject` and `streamObject`**

The Vercel AI SDK tackles this with `generateObject` and `streamObject`. These functions work with schema definitions (like Zod, which is highly recommended for its type safety and ease of use) to:

1.  Clearly communicate the desired data structure to the LLM.
2.  Automatically validate the LLM's output against your schema.
3.  Provide standardized ways to handle errors if the output doesn't match.

Think of it like giving the LLM a strict template to fill out, ensuring you get predictable, usable data.

### 5.1. `generateObject`: Generating Complete Objects

Use `generateObject` when you need the entire structured object at once, after the LLM has finished generating it. This is suitable for backend processing or when the full object is required before proceeding.

**Example: Generating a Recipe Object**

```typescript
import { generateObject } from 'ai';
import { openai } from '@ai-sdk/openai'; // Or your preferred provider
import { z } from 'zod'; // We'll use Zod for schema definition

// 1. Define the desired output structure using Zod
const recipeSchema = z.object({
  recipe: z.object({
    name: z.string().describe("The name of the recipe"), // Descriptions help the LLM
    ingredients: z.array(
      z.object({
        name: z.string(),
        amount: z.string().describe("e.g., '1 cup', '2 tbsp'")
      })
    ).describe("List of ingredients"),
    steps: z.array(z.string()).describe("Ordered list of cooking steps"),
  }),
});

async function getRecipe(prompt: string) {
  // 2. Choose a model capable of following instructions (tool/function calling helps)
  const model = openai('gpt-4o-mini');

  console.log("Requesting recipe from LLM...");
  try {
    // 3. Call generateObject with the model, schema, and prompt
    const { object, usage, finishReason, response } = await generateObject({
      model: model,
      schema: recipeSchema, // Pass the Zod schema directly
      prompt: prompt,
      // Optional: Add schemaName/schemaDescription for more context
      // schemaName: 'RecipeOutput',
      // schemaDescription: 'Structured representation of a cooking recipe.',
    });

    // 4. 'object' is now a fully validated JavaScript object matching recipeSchema
    console.log('Generated Recipe (Validated):');
    console.log(JSON.stringify(object, null, 2)); // Pretty print the object
    console.log('Token Usage:', usage);
    console.log('Finish Reason:', finishReason); // e.g., 'stop', 'tool-calls'

  } catch (error) {
    // 5. Handle potential errors (more on this later)
    console.error("Error generating object:", error);
  }
}

// Example usage:
getRecipe('Generate a simple recipe for chocolate chip cookies.');
```

**Explanation:**

1.  **Schema Definition:** We use Zod to define the precise shape of the `recipe` object we expect. Adding `.describe()` helps the LLM understand the purpose of each field.
2.  **Model Selection:** Choose a model known for good instruction following or tool/function calling capabilities.
3.  **`generateObject` Call:** The core function takes the model instance, the Zod schema, and the user's prompt.
4.  **Validated Output:** If successful, the `object` property contains the parsed and validated data, typed according to your Zod schema. You also get metadata like token `usage` and the `finishReason`.
5.  **Error Handling:** If the LLM fails to produce valid JSON matching the schema, `generateObject` will throw an error (see Section 5.6).

### 5.2. `streamObject`: Streaming Partial Objects

Waiting for a large, complex object to be fully generated can hurt user experience, especially in interactive UIs. `streamObject` solves this by providing the object piece by piece as it's generated by the LLM.

**Analogy:** Instead of waiting for the entire newspaper to be printed (`generateObject`), you get articles as they come off the press (`streamObject`).

This is particularly powerful for "Generative UI," where parts of the interface can appear incrementally.

**Example: Streaming a Recipe**

```typescript
import { streamObject } from 'ai';
import { openai } from '@ai-sdk/openai';
import { z } from 'zod';

// Assume recipeSchema is defined as in the generateObject example
const model = openai('gpt-4o-mini');

async function streamRecipe(prompt: string) {
  console.log("Streaming recipe from LLM...");

  // 1. Call streamObject - returns an object with streams and promises
  const { partialObjectStream, usagePromise, finishReasonPromise, validationErrorPromise } = streamObject({
    model: model,
    schema: recipeSchema,
    prompt: prompt,
    onError: (error) => {
      // Handle errors that occur *during* streaming (e.g., network issues)
      console.error("Streaming error caught by onError:", error);
    },
    onFinish: (event) => {
      // Called when streaming is complete, *before* final validation
      console.log("Streaming finished. Final validation starting...");
      console.log("  Finish Reason:", event.finishReason);
      console.log("  Usage:", event.usage);
      // Note: event.error contains any validation error at this point
      if (event.error) {
         console.warn("  Validation Error reported in onFinish:", event.error);
      }
      // event.partialObject contains the final (but potentially invalid) raw object
    }
  });

  console.log('Streaming Recipe Parts (Partial Objects):');
  try {
    // 2. Iterate through the stream of partial objects
    for await (const partialObject of partialObjectStream) {
      // Process the partial object (e.g., update UI)
      // Note: Fields might be missing or incomplete until the stream ends.
      console.log(JSON.stringify(partialObject, null, 2));
    }
  } catch(error) {
      // This catches errors during stream iteration itself
      console.error("Error during stream iteration:", error);
  }


  // 3. Await promises for final metadata and potential validation errors
  const usage = await usagePromise;
  const finishReason = await finishReasonPromise;
  const validationError = await validationErrorPromise; // Check this for schema errors

  console.log('--- Stream Finalization ---');
  console.log('Final Token Usage:', usage);
  console.log('Final Finish Reason:', finishReason);
  if (validationError) {
    console.error('Final Schema Validation Error:', validationError);
  } else {
    console.log('Schema validation successful.');
  }
  console.log('-------------------------');
}

// Example usage:
streamRecipe('Generate a detailed recipe for sourdough bread.');
```

**Explanation:**

1.  **`streamObject` Call:** Similar to `generateObject`, but returns an object containing `partialObjectStream` (an async iterable) and promises for `usage`, `finishReason`, and importantly, `validationErrorPromise`. Callbacks like `onError` and `onFinish` provide hooks into the streaming lifecycle.
2.  **Iterating the Stream:** The `for await...of` loop processes each chunk of the object as it arrives. The `partialObject` in each iteration might be incomplete. You'd typically use this to update a UI state reactively.
3.  **Final Validation:** After the stream ends, you *must* check the `validationErrorPromise`. If it resolves with an error, the final object generated by the LLM did not conform to the schema.

### 5.3. Output Strategies (`output` parameter)

You can tell `generateObject` or `streamObject` what kind of high-level structure you expect using the `output` parameter:

*   **`object` (Default):** Generates a single JSON object matching the provided `schema`. This is the most common case.
*   **`array`:** Generates a JSON array where *each element* matches the `schema`. Useful for lists of items.
    *   When used with `streamObject`, you get an `elementStream` to process each array item individually as it's generated.
    *   *Use Case:* Generating a list of product descriptions, character profiles, or task items.
    ```typescript
    // streamObject({ output: 'array', schema: productSchema, ... })
    // for await (const product of elementStream) { /* process product */ }
    ```
*   **`enum`:** Generates *one value* from a predefined list (an array of strings). Only works with `generateObject`.
    *   *Use Case:* Simple classification tasks, like determining sentiment ('positive', 'negative', 'neutral') or categorizing an email ('spam', 'important', 'promo').
    ```typescript
    // const { object: sentiment } = await generateObject({ output: 'enum', enum: ['good', 'bad'], ... })
    ```
*   **`no-schema`:** Generates a JSON object *without* validating it against a specific schema. Use this if you need JSON structure but don't have a strict schema or want to handle validation manually.
    *   *Use Case:* When you need arbitrary JSON output based on the prompt, and strict validation isn't required upfront.

### 5.4. Generation Modes (`mode` parameter)

This parameter hints to the SDK *how* it should try to get the structured output from the LLM:

*   **`auto` (Default & Recommended):** The SDK intelligently chooses the best method available for the specific model (e.g., using the model's native JSON mode if available, or falling back to tool/function calling). Let the SDK handle the details.
*   **`tool`:** Forces the SDK to use the "tool calling" mechanism. The schema you provide is presented to the LLM as the definition of a tool it should call.
*   **`json`:** Explicitly requests the model's JSON output mode, if supported. May involve prompt engineering if the model doesn't have a native JSON mode. Check provider/model documentation for compatibility.

In most cases, `auto` is the best choice.

### 5.5. Schema Name and Description

Optionally provide `schemaName` and `schemaDescription` strings to `generateObject` or `streamObject`. These can give the LLM extra context about the *purpose* of the structured data you're asking for, which can sometimes improve results, especially when using `mode: 'tool'`.

```typescript
await generateObject({
  model: model,
  schema: recipeSchema,
  schemaName: 'RecipeOutput', // Optional name for the 'tool'
  schemaDescription: 'A structured representation of a cooking recipe, including ingredients and steps.', // Optional description
  prompt: 'Generate a pizza recipe.',
});
```

### 5.6. Error Handling: When Generation Fails

If `generateObject` cannot produce an object that successfully validates against your schema after retries, it throws a `NoObjectGeneratedError`. This custom error type is helpful because it contains:

*   `cause`: The underlying error (e.g., JSON parsing error, validation error).
*   `text`: The raw (invalid) text the LLM generated.
*   `response`: The raw HTTP response metadata.
*   `usage`: Token usage information.

You should wrap `generateObject` calls in a `try...catch` block and specifically check for this error type.

```typescript
import { generateObject, NoObjectGeneratedError } from 'ai';
// Assume model, schema, prompt defined

try {
  const { object } = await generateObject({ model, schema, prompt });
  // Use the validated object
} catch (error) {
  if (NoObjectGeneratedError.isInstance(error)) {
    // Handle the specific failure case
    console.error('Failed to generate a valid object conforming to the schema!');
    console.error('Cause:', error.cause); // What went wrong (e.g., Zod validation issue)
    console.error('Problematic Text:', error.text); // Log the LLM's invalid output
    console.error('Token Usage:', error.usage);
    // Maybe fall back to a simpler request or inform the user
  } else {
    // Handle other unexpected errors (network, API keys, etc.)
    console.error('An unexpected error occurred:', error);
  }
}
```

For `streamObject`, remember that validation happens *after* the stream completes. Check the `validationErrorPromise` or the `error` property in the `onFinish` callback's event object.

### 5.7. Experimental Features

The SDK sometimes includes experimental features related to object generation:

*   **`experimental_repairText` (for `generateObject`):** Provide an async function that attempts to fix slightly malformed JSON text before the final parsing attempt. Useful for minor syntax errors (like missing closing braces).
*   **`experimental_output` (with `generateText`/`streamText`):** An experimental way to request structured output *alongside* regular text generation within a single `generateText` or `streamText` call. This is useful if a model supports mixed outputs (e.g., explaining something and providing data via a tool call in the same response).

Refer to the official documentation for the latest status and usage of experimental features.

## 6. Building Interactive UIs: Simplifying Frontend Development

**Learning Objectives:** By the end of this section, you should be able to:
*   Explain the challenges of building interactive AI frontends (state management, streaming, API calls).
*   Describe how AI SDK UI hooks (`useChat`, `useCompletion`, `useObject`, `useAssistant`) solve these challenges.
*   Implement a basic chatbot using `useChat` in a Next.js application.
*   Implement a text completion interface using `useCompletion`.
*   Understand the purpose and basic usage of `useObject` for streaming structured data to the UI.
*   Understand the purpose and basic usage of `useAssistant` for interacting with the OpenAI Assistants API.
*   Identify the key features and customization options for each hook.

**The "Why": Taming Frontend Complexity for AI**

Integrating AI, especially streaming LLM responses, into a user interface presents unique challenges. You need to:

*   **Manage State:** Keep track of messages, user input, loading indicators, and error states.
*   **Handle Streaming:** Efficiently receive and display response chunks as they arrive without blocking the UI.
*   **API Interaction:** Make requests to your backend API route, handle responses, and manage potential network errors.
*   **Boilerplate:** Write repetitive code for form handling, state updates, and stream processing.

Doing this manually for every AI feature can be tedious and error-prone.

**The SDK Solution: UI Hooks for Common Patterns**

The Vercel AI SDK UI packages (`@ai-sdk/react`, `@ai-sdk/vue`, etc.) provide pre-built abstractions, primarily in the form of **hooks**, designed for popular frontend frameworks. These hooks encapsulate the common logic for building specific AI interactions, letting you focus on your UI presentation and application logic.

Think of these hooks as specialized assistants for your frontend:

*   `useChat`: Your go-to assistant for building conversational, multi-turn chat interfaces.
*   `useCompletion`: An assistant for simpler "prompt-in, text-out" scenarios, like autocompletion or generating short text snippets.
*   `useObject`: An experimental assistant for displaying structured JSON data that streams in piece by piece.
*   `useAssistant`: A highly specialized assistant designed specifically for the complexities of the OpenAI Assistants API.

Let's explore how they work, primarily using React/Next.js examples.

### 6.1. `useChat`: Building Conversational Experiences

**The Problem:** Building a chatbot requires managing a list of messages (from user and AI), handling user input, submitting the conversation history to the backend, receiving a streaming response, and updating the message list dynamically.

**The Solution:** The `useChat` hook manages all this state and interaction logic for you.

**How it Works:**
1.  You provide user input controls and display messages using data from the hook.
2.  When the user submits input (e.g., via a form), `useChat` automatically:
    *   Appends the user message to its internal `messages` state.
    *   Sends the entire `messages` history to your backend API endpoint (defaults to `/api/chat`).
    *   Receives the streaming response from your API.
    *   Appends the streaming AI response to the `messages` state chunk by chunk.
    *   Manages loading and error states.

**Example: Basic Chatbot (Next.js)**

This requires two parts: the backend API route that calls the LLM and the frontend component using the hook.

**1. Backend API Route (`app/api/chat/route.ts`)**

This route takes the message history, sends it to the LLM via `streamText`, and streams the result back.

```typescript:app/api/chat/route.ts
// File: app/api/chat/route.ts
import { openai } from '@ai-sdk/openai';
import { streamText, CoreMessage } from 'ai';

// Allow streaming responses for up to 30 seconds
export const maxDuration = 30;

export async function POST(req: Request) {
  try {
    // 1. Extract the messages from the request body
    const { messages }: { messages: CoreMessage[] } = await req.json();

    // 2. Call the LLM with the message history using streamText
    const result = await streamText({
      model: openai('gpt-4o-mini'), // Or your preferred model
      messages: messages, // Pass the conversation history
      // Optional: Add system prompts, other settings
    });

    // 3. Respond with the AI's streamed response
    // '.toDataStreamResponse()' sets up the correct headers and format
    // for the useChat hook to consume.
    return result.toDataStreamResponse();

  } catch (error) {
    console.error("Error in chat API route:", error);
    // Handle errors appropriately, maybe return a JSON error response
    return new Response("Internal Server Error", { status: 500 });
  }
}
```

*   **Key Point:** The backend uses `streamText` and crucially returns `result.toDataStreamResponse()`. This specific response format is expected by `useChat`.

**2. Frontend Component (`app/page.tsx`)**

This component renders the chat interface using the state and handlers provided by `useChat`.

```typescript:app/page.tsx
// File: app/page.tsx
'use client'; // Required for hooks in Next.js App Router

import { useChat } from '@ai-sdk/react'; // Import the hook for React

export default function Chat() {
  // 1. Initialize the hook. It provides everything needed for a chat UI.
  const {
    messages,       // Array of message objects (role: 'user' | 'assistant' | ...)
    input,          // Current value of the input field
    handleInputChange, // Default handler for input changes
    handleSubmit,   // Default handler for form submission
    isLoading,      // Boolean indicating if a response is streaming
    error,          // Error object if something went wrong
    // ... other helpers like reload, stop, setMessages, etc.
  } = useChat({
    // Optional config:
    api: '/api/chat', // Defaults to this, customize if needed
    // initialMessages: [], // Start with predefined messages
    // onResponse: (res) => console.log('Got response:', res),
    // onFinish: (msg) => console.log('Stream finished:', msg),
    // onError: (err) => console.error('Chat error:', err),
    // sendReasoning: true, // If backend sends reasoning tokens
  });

  return (
    <div className="flex flex-col w-full max-w-md py-24 mx-auto stretch">
      {/* 2. Display the messages */}
      <div className="space-y-4 mb-4 h-96 overflow-y-auto p-4 border rounded">
        {messages.length > 0
          ? messages.map(m => (
              <div key={m.id} className={`whitespace-pre-wrap ${m.role === 'user' ? 'text-right' : ''}`}>
                <strong className="capitalize">{m.role === 'user' ? 'You' : 'AI'}: </strong>
                {/* Optionally render reasoning if backend sends it */}
                {m.reasoning && <pre className="text-xs text-gray-500 bg-gray-100 p-1 my-1 rounded">{m.reasoning}</pre>}
                {m.content}
              </div>
            ))
          : <div className="text-center text-gray-500">No messages yet. Ask something!</div>
        }
      </div>

      {/* Display loading indicator */}
      {isLoading && <div className="text-center text-gray-500 my-2">AI thinking...</div>}

      {/* Display error message */}
      {error && <div className="text-red-500 text-center my-2">Error: {error.message}</div>}

      {/* 3. Chat input form */}
      <form onSubmit={handleSubmit}>
        <input
          className="fixed bottom-0 w-full max-w-md p-2 mb-8 border border-gray-300 rounded shadow-xl disabled:opacity-50"
          value={input}
          placeholder="Say something..."
          onChange={handleInputChange} // Connects input value to the hook's state
          disabled={isLoading} // Disable input while waiting for response
        />
         {/* The submit button is implicitly handled by the form's onSubmit */}
         {/* You could add an explicit button too: <button type="submit" disabled={isLoading}>Send</button> */}
      </form>
    </div>
  );
}
```

*   **Key Points:**
    *   `useChat` manages the complex state (`messages`, `input`, `isLoading`, `error`).
    *   `handleInputChange` and `handleSubmit` are default handlers that connect the form to the hook's logic.
    *   The component just needs to *render* the UI based on the hook's state.

**Use Cases:** Customer support bots, coding assistants, general-purpose chatbots, interactive tutorials.

### 6.2. `useCompletion`: Simple Prompt-to-Text Generation

**The Problem:** You need a quick way to generate text based on a single user prompt, like suggesting email subject lines, completing a sentence, or generating a short paragraph, and you want the result streamed to the UI.

**The Solution:** The `useCompletion` hook simplifies this one-shot prompt-and-response pattern.

**How it Works:**
1.  The user provides a prompt in an input field.
2.  When submitted, `useCompletion` sends *only the current prompt* to your backend API (defaults to `/api/completion`).
3.  It receives the streaming text response and makes it available in the `completion` state variable.
4.  It also manages `isLoading` and `error` states.

**Example: Simple Text Completion (Next.js)**

**1. Backend API Route (`app/api/completion/route.ts`)**

This route takes just a prompt string and streams back the LLM's completion.

```typescript:app/api/completion/route.ts
// File: app/api/completion/route.ts
import { streamText } from 'ai';
import { openai } from '@ai-sdk/openai';

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

export async function POST(req: Request) {
  try {
    // 1. Extract the prompt from the request
    const { prompt }: { prompt: string } = await req.json();

    // 2. Call the LLM with the prompt
    const result = await streamText({
      model: openai('gpt-3.5-turbo'), // Often a smaller/faster model is suitable
      prompt: prompt,
    });

    // 3. Stream the completion back
    return result.toDataStreamResponse();

  } catch (error) {
    console.error("Error in completion API route:", error);
    return new Response("Internal Server Error", { status: 500 });
  }
}
```

**2. Frontend Component (`app/page.tsx`)**

This component uses `useCompletion` to get the prompt and display the streamed result.

```typescript:app/page.tsx
// File: app/page.tsx
'use client';

import { useCompletion } from '@ai-sdk/react';

export default function CompletionPage() {
  // 1. Initialize the hook
  const {
    completion,      // The streamed completion text state
    input,           // The current input prompt state
    handleInputChange, // Default handler for input changes
    handleSubmit,    // Default handler for form submission
    isLoading,       // Loading state
    error,           // Error state
    stop,            // Function to stop the stream manually
  } = useCompletion({
    api: '/api/completion', // Point to the backend route
    // Optional config:
    // onFinish: (prompt, completion) => console.log('Completed:', completion),
    // onError: (err) => console.error('Completion error:', err),
  });

  return (
    <div className="flex flex-col w-full max-w-md py-12 mx-auto stretch">
      {/* 2. Input form */}
      <form onSubmit={handleSubmit}>
        <label className="block mb-2 font-medium">
          Enter prompt:
          <textarea
            className="w-full p-2 mb-2 border border-gray-300 rounded shadow-xl"
            rows={3}
            name="prompt" // Optional: name can be useful
            value={input}
            onChange={handleInputChange}
            placeholder="e.g., Write a tagline for a coffee shop"
            disabled={isLoading}
          />
        </label>
        <button
          type="submit"
          disabled={isLoading}
          className="px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50"
        >
          {isLoading ? 'Generating...' : 'Generate Completion'}
        </button>
        {isLoading && (
           <button
             type="button"
             onClick={stop}
             className="ml-2 px-4 py-2 bg-red-500 text-white rounded"
           >
             Stop
           </button>
        )}
      </form>

      {/* 3. Display error */}
      {error && (
        <div className="mt-4 text-red-500">Error: {error.message}</div>
      )}

      {/* 4. Display completion */}
      {completion && (
        <div className="mt-4 p-4 border border-gray-200 rounded bg-gray-50 whitespace-pre-wrap">
          <h3 className="font-semibold">Completion Result:</h3>
          {completion}
        </div>
      )}
    </div>
  );
}
```

*   **Key Differences from `useChat`:** Manages a single `completion` string, not a list of messages. Sends only the latest prompt, not history. Simpler state management.

**Use Cases:** Autocomplete features, email subject generation, summarizing text, generating short descriptions, simple creative writing prompts.

### 6.3. `useObject`: Streaming Structured Data (Experimental)

**The Problem:** Sometimes, you need the AI to generate structured data (JSON) and display it incrementally as it streams in, rather than waiting for the whole object (covered by `streamObject` in the backend - Section 5.2).

**The Solution:** The `useObject` hook (experimental, React only) connects a frontend component to a backend API route that uses `streamObject`.

**How it Works (Conceptual):**
1.  You define a schema (e.g., with Zod) shared between the frontend and backend.
2.  The backend API route uses `streamObject` with that schema and streams the partial object back (often using `result.toTextStreamResponse()`).
3.  `useObject` receives the stream, parses the partial JSON chunks, and updates its `object` state variable.
4.  Your component re-renders as the `object` state gets progressively filled.
5.  It handles loading, error, and final validation against the schema.

*(See the example in the original Section 6.3 of the attached file for a code illustration, as the core concept remains the same).*

**Use Cases:** Dynamically generating and displaying complex UI elements based on LLM output (Generative UI), showing structured data like lists or tables as they load, real-time data dashboards populated by AI.

### 6.4. `useAssistant`: Interacting with OpenAI Assistants API

**The Problem:** The OpenAI Assistants API is powerful but stateful and complex. It involves managing Threads, Messages, Runs, Steps, and Tool Calls, requiring intricate backend logic to handle the conversation lifecycle and stream updates correctly.

**The Solution:** The `useAssistant` hook is specifically designed to manage the frontend state and interactions required for the OpenAI Assistants API (v1/v2).

**How it Works:**
1.  It maintains the essential state: `status` (the current state of the Assistant's run), `messages` (including user, assistant, and potentially data messages), `input`, `error`, and the current `threadId`.
2.  It communicates with a dedicated backend API route.
3.  This backend route uses the official `openai` Node SDK and the `AssistantResponse` utility from the Vercel AI SDK (`ai` package) to handle the complex event streaming from the OpenAI API (message deltas, tool calls, run status changes).
4.  The backend uses `sendMessage` and `sendDataMessage` (provided by `AssistantResponse`) to forward updates to the frontend hook.
5.  `useAssistant` updates its state based on these streamed updates, allowing your UI to react to the Assistant's progress.

*(See the example in the original Section 6.4 of the attached file for a code illustration. Note the complexity in the backend route required to handle the Assistant's lifecycle).*

**Key Points:**
*   Tightly coupled with the OpenAI Assistants API structure.
*   Requires a specific backend setup using `AssistantResponse`.
*   Abstracts the complex state management (Threads, Runs) needed for Assistants.
*   Supports `sendDataMessage` for sending custom structured data related to Assistant actions (like tool calls) to the frontend.

**Use Cases:** Building applications directly on top of the OpenAI Assistants API, leveraging its persistent threads and built-in tool execution capabilities.

**Choosing the Right Hook:**

*   **Multi-turn Chat:** Use `useChat`.
*   **Simple Prompt -> Streaming Text:** Use `useCompletion`.
*   **Streaming Structured JSON to UI:** Use `useObject` (Experimental).
*   **OpenAI Assistants API:** Use `useAssistant`.

These hooks provide powerful building blocks for creating engaging and interactive AI-powered user interfaces with significantly less boilerplate code.

## 7. Agentic Systems: Enabling AI Actions and Complex Tasks

**Learning Objectives:** By the end of this section, you should be able to:
*   Explain what constitutes an "agentic" AI system and why it's needed.
*   Identify the core building blocks of agents: single-step generation, tool usage, and multi-step execution.
*   Describe common agent patterns like sequential processing, routing, and evaluation loops.
*   Define tools using the `tool` helper, including descriptions and Zod parameter schemas.
*   Use tools with `generateText` and `streamText`.
*   Implement multi-step tool usage with `maxSteps` to solve complex problems.
*   Structure final agent answers using a dedicated "answer tool" and `toolChoice: 'required'`.
*   Inspect agent execution using the `steps` property and `onStepFinish` callback for debugging and observability.
*   Manage conversation history correctly in multi-step scenarios using `responseMessages`.
*   Control when and how an agent uses tools via the `toolChoice` parameter.

**The "Why": Beyond Conversation and Generation**

So far, we've seen how to generate text, structured data, and build interactive UIs. But what if you need the AI to do more than just *respond*? What if you need it to **take actions**, interact with external systems, or solve problems that require multiple steps and access to real-time information or specific capabilities?

This is where **agentic systems** come in. An "agent" in this context is an AI system designed to:

1.  **Understand Intent:** Figure out the user's goal.
2.  **Plan:** Break down the goal into smaller steps (potentially involving different tools or information sources).
3.  **Act:** Execute those steps, often by calling tools (functions, APIs, database queries).
4.  **Observe:** Analyze the results of its actions.
5.  **Reason:** Decide the next step based on observations, potentially iterating until the goal is achieved.

**Analogy:** Think of a simple chatbot like a knowledgeable librarian who can answer questions based on the books they have (`generateText`). An agent is more like a research assistant who can not only consult books but also use a calculator (`calculatorTool`), search online databases (`searchTool`), summarize findings, and compile a structured report (`answerTool`), potentially performing these actions in sequence until the research task is complete.

The Vercel AI SDK provides the core functionalities (`generateText`, `streamText`, `tool`, `maxSteps`) to build these more sophisticated agentic systems.

### 7.1. Agent Building Blocks: From Simple to Complex

Agents are constructed by combining fundamental capabilities:

1.  **Single-Step LLM Generation:** The simplest form. A single call to `generateText` or `generateObject`. Suitable for direct tasks like classification, summarization, or simple Q&A where no external action or multi-step reasoning is needed.
    *   *Use Case:* Classifying customer feedback sentiment.
2.  **Tool Usage (Single Step):** Empowering the LLM with tools. A call to `generateText` or `streamText` where the model *might* choose to call one of the provided `tools` to fulfill the request in a single turn.
    *   *Use Case:* Answering "What's the weather in London?" by calling a weather API tool. Calculating "5! * 3" using a calculator tool.
3.  **Multi-Step Tool Usage:** The core of more complex agents. Allowing the LLM to make a *sequence* of tool calls (and reasoning steps) across multiple turns automatically, managed by the SDK using the `maxSteps` parameter. This enables solving problems where the path isn't known beforehand.
    *   *Use Case:* Planning a trip, which might involve calling tools to find flights, check hotel availability, and then summarize the plan. Debugging code by analyzing errors, searching for solutions, and proposing fixes iteratively.
4.  **Multi-Agent Systems (Advanced):** For highly complex tasks, you might orchestrate multiple specialized agents. One agent could act as a router, delegating sub-tasks to other agents with specific expertise or tools. While possible to build using the SDK's core functions, this pattern requires more custom application logic.
    *   *Use Case:* A complex financial analysis system where one agent retrieves market data, another performs technical analysis, and a third generates a report.

### 7.2. Common Agent Patterns (Application Logic)

While the SDK provides the building blocks, you often structure your agent's workflow using application-level patterns:

*   **Sequential Processing (Chains):** Steps executed in a fixed order. Output A -> Process B -> Output C. Simple and predictable.
    *   *SDK Implementation:* Multiple separate calls to `generateText`/`generateObject`, passing outputs as inputs.
    *   *Use Case:* Generate draft email -> Evaluate tone using `generateObject` -> Refine draft based on evaluation.
*   **Routing:** Use an initial LLM call (often `generateObject`) to decide which path, model, prompt, or toolset to use next.
    *   *SDK Implementation:* An initial `generateObject` call determines parameters for subsequent `generateText`/`streamText` calls.
    *   *Use Case:* Classify user intent (e.g., 'Sales Question', 'Support Issue') and route to different specialized prompts or tool-equipped agents.
*   **ReAct (Reasoning and Acting):** A common pattern where the LLM explicitly reasons about the next step, decides on an action (often a tool call), executes it, observes the result, and repeats. The SDK's multi-step tool usage (`maxSteps`) automates a form of this loop.
*   **Evaluator/Refinement Loops:** Use one LLM call to generate an output and another (potentially `generateObject` with criteria) to evaluate it. Feed the evaluation back to the generator for refinement until a quality threshold is met.
    *   *SDK Implementation:* A loop in your application code calling `generateText` and `generateObject` iteratively.
    *   *Use Case:* Generating code, then using another LLM call to check for errors or adherence to style guides, then refining the code.

### 7.3. Choosing Your Agent Design

Consider these trade-offs:

*   **Flexibility vs. Control:** More steps and tool autonomy (`maxSteps > 1`) offer flexibility but less predictability. Single steps or fixed chains offer more control.
*   **Complexity & Cost:** Multi-step agents are powerful but involve more LLM calls (higher latency and cost) and complexity.
*   **Error Handling:** Autonomous agents require robust error handling within tools and potentially logic to recover from failed steps.
*   **Maintainability:** Start simple. Introduce tools and multi-step execution only when necessary.

**Recommendation:** Begin with the simplest approach that meets the need (single step, maybe one tool). Incrementally add multi-step execution (`maxSteps`) or more complex patterns if required.

### 7.4. Defining Tools: Giving the LLM Capabilities

Tools are the core mechanism for agents to interact with the world beyond text generation.

*   **Definition:** Use the `tool` helper function from the `ai` package.
*   **`description` (Crucial!):** Explain clearly and concisely *what the tool does* and *when the LLM should use it*. Provide examples if helpful. This is the LLM's primary guide.
*   **`parameters` (Schema):** Define the expected inputs using a Zod schema (`zod`) or JSON schema (`jsonSchema`). Include `.describe()` for each parameter to clarify its purpose for the LLM. Use `z.object({})` for tools with no arguments.
*   **`execute` (Optional but common):** An `async` function containing the tool's logic. It receives the parameters (validated against the schema) and should return the result the LLM needs. If `execute` is omitted, the SDK will return the tool call details, and your application logic must handle the execution.

```typescript
import { tool } from 'ai';
import { z } from 'zod'; // Make sure to install Zod: npm install zod
import * as mathjs from 'mathjs'; // Example dependency: npm install mathjs
// Assume fetchWeatherData is an external function you've defined elsewhere
// import { fetchWeatherData } from './api/weather';

// Example: Calculator Tool
export const calculatorTool = tool({
  // Why: Clearly state the tool's purpose and usage context.
  description:
    'Calculates the result of a mathematical expression. ' +
    'Use this for arithmetic, trigonometry, etc. ' +
    "Example expressions: '1.2 * (2 + 4.5)', '12.7 cm to inch', 'sin(45 deg) ^ 2'.",
  parameters: z.object({
    // Why: Define expected input with type and description.
    expression: z.string().describe('The mathematical expression to evaluate. Ensure it\'s a valid expression for mathjs.'),
  }),
  // Why: The function that performs the actual work when the tool is called.
  execute: async ({ expression }) => {
    console.log(`Executing calculator tool with expression: ${expression}`);
    try {
      // Safely evaluate the expression
      const result = mathjs.evaluate(expression);
      // Return the result to the LLM
      return result;
    } catch (error) {
      // Handle potential errors during evaluation and inform the LLM
      console.error("Calculator tool error:", error);
      return `Error evaluating expression "${expression}": ${error instanceof Error ? error.message : 'Unknown error'}`;
    }
  },
});

// Example: Weather Tool (execute handled by application or omitted for now)
export const weatherTool = tool({
  description: 'Gets the current weather for a specific location.',
  parameters: z.object({
    location: z.string().describe('The city and state, e.g., San Francisco, CA'),
    unit: z.enum(['celsius', 'fahrenheit']).default('celsius').describe("Temperature unit")
  }),
  // execute: async ({ location, unit }) => {
  //   console.log(`Fetching weather for ${location} in ${unit}`);
  //   return await fetchWeatherData(location, unit); // Call your API logic
  // }
  // If execute is omitted, the LLM call will return a toolCall object
  // with toolName 'weather' and args { location: '...', unit: '...' }
});
```

### 7.5. Using Tools with `generateText` and `streamText`

Pass your defined tools to the LLM call via the `tools` parameter. This is an object where keys are the names the LLM will use to refer to the tool, and values are the tool definitions created with the `tool` helper.

```typescript
import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';
import { calculatorTool, weatherTool } from './tools'; // Assume tools are imported

const model = openai('gpt-4o-mini'); // Choose a model good at tool use

async function askWithTools(prompt: string) {
  console.log(`Asking: ${prompt}`);
  const { text, toolCalls, toolResults, finishReason } = await generateText({
    model: model,
    // Provide the tools the LLM can use
    tools: {
      calculate: calculatorTool, // LLM refers to this tool as 'calculate'
      getCurrentWeather: weatherTool // LLM refers to this as 'getCurrentWeather'
    },
    prompt: prompt,
  });

  // Analyze the result:
  console.log('LLM Finish Reason:', finishReason); // 'stop', 'tool-calls'
  console.log('LLM Text Response:', text); // Text generated by the LLM (might be empty if only tools were called)

  if (toolCalls?.length) {
    console.log('LLM Tool Calls:', toolCalls);
    // Example: [{ toolCallId: '...', toolName: 'calculate', args: { expression: '2+2' } }]
    // If tools had no 'execute', your application logic would run them here.
  }

  if (toolResults?.length) {
    console.log('Tool Execution Results:', toolResults);
    // Example: [{ toolCallId: '...', toolName: 'calculate', result: 4 }]
    // Populated if the tool definition included an 'execute' function that ran.
  }
}

// Example where the LLM likely uses the calculator
askWithTools("What is the result of log10(100) * 5?");

// Example where the LLM likely calls the weather tool (if execute were defined)
// askWithTools("What's the weather like in Paris in fahrenheit?");
```

### 7.6. Multi-Step Tool Usage: Solving Complex Problems (`maxSteps`)

**The "Why":** Many problems can't be solved in one shot. They require a sequence of actions, where the result of one step informs the next. Planning a trip, debugging code, or complex calculations often fall into this category. The exact sequence might not be known beforehand.

**The Solution:** The `maxSteps` parameter in `generateText` and `streamText` enables the LLM to perform multiple reasoning steps and tool calls autonomously within a single SDK call.

**Analogy:** It's like giving your research assistant (the LLM) permission to not just use the calculator once, but to perform a series of calculations, lookups, or other tool actions as needed to arrive at the final answer, up to a certain limit (`maxSteps`).

**The Automatic Workflow:**
1.  Your initial prompt is sent to the LLM.
2.  LLM Response:
    *   If it's just text: The process stops, and the text is returned.
    *   If it includes a `toolCall` for a tool *with* an `execute` function:
        a.  The SDK pauses LLM generation.
        b.  The SDK calls your tool's `execute` function with the arguments provided by the LLM.
        c.  The SDK sends the `toolResult` back to the LLM as context for the *next* step.
        d.  The LLM continues, now aware of the tool's result.
    *   If it includes a `toolCall` for a tool *without* an `execute` function: The process stops, returning the `toolCall` details for your application to handle.
3.  Looping: If a tool with `execute` was called, the process loops back to Step 2 (LLM generates based on new context).
4.  Termination: The loop stops when:
    *   The LLM generates only text.
    *   The `maxSteps` limit (counting LLM generation turns + tool execution turns) is reached.
    *   A tool *without* `execute` is called.
    *   An error occurs.

```typescript
import { z } from 'zod';
import { generateText, tool } from 'ai';
import { openai } from '@ai-sdk/openai';
import * as mathjs from 'mathjs'; // Requires: npm install mathjs

const model = openai('gpt-4o-mini');
const calculator = tool({
  description: 'Evaluate mathematical expressions.',
  parameters: z.object({ expression: z.string() }),
  execute: async ({ expression }) => {
     console.log(`   [Calculator executing: ${expression}]`);
     return mathjs.evaluate(expression);
  }
});

async function solveMultiStepProblem(prompt: string) {
  console.log(`Solving multi-step: ${prompt}`);
  const { text, toolCalls, toolResults, finishReason, usage, steps } = await generateText({
    model: model,
    tools: {
      calculator: calculator // Provide the tool with its execute function
    },
    maxSteps: 5, // Allow up to 5 steps (LLM calls + tool executions)
    system: 'You solve math problems step-by-step. Use the calculator when needed. Explain your final answer.',
    prompt: prompt,
  });

  console.log('\n--- Multi-Step Result ---');
  console.log('Final Text:', text); // The final text output from the LLM
  console.log('Finish Reason:', finishReason); // e.g., 'stop', 'max-steps', 'tool-calls' (if final step was a call)
  console.log('Total Usage:', usage);
  console.log('-----------------------');

  // You can also inspect the intermediate steps (see 7.6.2)
  // console.log('\nIntermediate Steps:', steps);
}

// Example requiring multiple calculation steps
solveMultiStepProblem("A company's revenue is $10,000. Cost of Goods Sold is $4,500. Operating Expenses are $2,000. What is the net profit margin percentage?");
// Expected flow:
// 1. LLM asks calculator for 10000 - 4500 (Gross Profit) -> Result: 5500
// 2. LLM asks calculator for 5500 - 2000 (Net Profit) -> Result: 3500
// 3. LLM asks calculator for (3500 / 10000) * 100 (Margin %) -> Result: 35
// 4. LLM generates final text explaining the result is 35%.
```

#### 7.6.1. Structured Final Answers (`toolChoice: 'required'` + Answer Tool)

**The "Why":** Sometimes, especially after a multi-step process, you don't want just a text explanation; you need the agent's final answer in a reliable, structured format (JSON), similar to `generateObject` but at the *end* of an agentic workflow.

**The Solution:** Define a special "answer tool" whose `parameters` schema matches your desired final JSON structure. Crucially, **do not provide an `execute` function** for this tool. Then, in your `generateText` call:
1.  Include the answer tool in the `tools` list alongside operational tools (like `calculator`).
2.  Set `toolChoice: 'required'`.

This forces the LLM's *very last step* to be a call to one of the provided tools. By carefully prompting the LLM to use the specific "answer tool" when it has finished its reasoning and calculations, you guide it to structure its final output according to your schema. The result will be in the `toolCalls` array.

```typescript
import { z } from 'zod';
import { generateText, tool } from 'ai';
import { openai } from '@ai-sdk/openai';
// Assume calculator tool is defined as above

// 1. Define the schema for the final structured answer
const calculationAnswerSchema = z.object({
  reasoningSteps: z.array(
    z.object({
      stepDescription: z.string(),
      calculation: z.string().optional(), // e.g., "10000 - 4500"
      result: z.union([z.string(), z.number()]).optional(), // e.g., 5500
    }),
  ).describe("Step-by-step reasoning and calculations performed."),
  finalAnswer: z.number().describe("The final numerical answer."),
  explanation: z.string().describe("Brief textual explanation of the result and process."),
});

// 2. Define the 'answer' tool with the schema but NO execute function
const finalAnswerTool = tool({
  description: 'Use this tool ONLY to provide the final answer, reasoning steps, calculations, and explanation once the problem is fully solved.',
  parameters: calculationAnswerSchema,
  // NO 'execute' - calling this tool terminates the agent loop successfully.
});

async function solveAndStructure(prompt: string) {
  const model = openai('gpt-4o-mini'); // Model good at tool use

  console.log(`Solving and structuring: ${prompt}`);
  const { text, toolCalls, finishReason, steps } = await generateText({
    model: model,
    tools: {
      // Provide operational tools...
      calculator: calculator,
      // ...and the final answer structuring tool
      provideFinalAnswer: finalAnswerTool // LLM refers to it by this name
    },
    // 3. Force the final step to be a tool call
    toolChoice: 'required',
    maxSteps: 10, // Allow enough steps for calculation and final structuring
    // 4. Prompt instructs use of calculator AND the final answer tool
    system: `Solve the math problem step-by-step. Use the 'calculator' tool for all calculations.
             When you have the final answer and all steps, you MUST call the 'provideFinalAnswer' tool with the detailed steps, calculations, final numeric answer, and a textual explanation. Do not provide the final answer in plain text.`,
    prompt: prompt,
  });

  console.log('\n--- Structured Answer Result ---');
  console.log('Finish Reason:', finishReason); // Should ideally be 'tool-calls'

  // 5. Extract the structured answer from the toolCalls array
  const answerCall = toolCalls?.find(call => call.toolName === 'provideFinalAnswer');

  if (answerCall) {
    // The validated arguments of the 'provideFinalAnswer' call ARE the structured result
    const structuredOutput = answerCall.args;
    console.log('Structured Answer Extracted:');
    console.log('  Final Numeric Answer:', structuredOutput.finalAnswer);
    console.log('  Explanation:', structuredOutput.explanation);
    console.log('  Reasoning Steps:', JSON.stringify(structuredOutput.reasoningSteps, null, 2));
  } else {
    console.log('Error: Final answer tool was not called.');
    console.log('Final Text:', text); // Log text for debugging
    console.log('All Tool Calls:', toolCalls); // Log all calls for debugging
  }
  console.log('------------------------------');
}

solveAndStructure("Revenue $10k, COGS $4.5k, OpEx $2k. Net profit margin %?");
```
*Note:* Success depends heavily on the model's ability to follow the prompt instructing it to use the specific answer tool *last*. `toolChoice: 'required'` ensures *a* tool is called last, but the prompt guides *which* tool.

#### 7.6.2. Accessing Intermediate Steps (`steps` Property)

**The "Why":** When an agent takes multiple steps, especially autonomously, understanding *how* it arrived at the result is crucial for debugging, evaluation, and building trust. You need to see the intermediate thoughts, actions, and observations.

**The Solution:** The result object returned by `generateText` (when `maxSteps > 0`) includes a `steps` property. This is an array containing the detailed history of the execution flow, step-by-step.

Each element in the `steps` array represents one turn and includes:
*   `type`: 'text', 'tool-calls', or 'tool-results'.
*   The corresponding data (`text`, `toolCalls`, or `toolResults`).
*   `usage`: Token usage for that specific step.

```typescript
// Building on the multi-step example:
async function solveAndShowSteps(prompt: string) {
  const { text, finishReason, steps } = await generateText({
    model: model,
    tools: { calculator: calculator },
    maxSteps: 5,
    system: 'Solve step-by-step, use calculator.',
    prompt: prompt,
  });

  console.log('Final Text:', text);
  console.log('Finish Reason:', finishReason);

  console.log('\n--- Execution Steps Trace ---');
  steps.forEach((step, index) => {
    console.log(`Step ${index + 1}:`);
    console.log(`  Type: ${step.type}`);
    if (step.type === 'text') {
      console.log(`  LLM Text Output: "${step.text}"`);
    } else if (step.type === 'tool-calls') {
      // Log the arguments the LLM decided to use for the tool
      console.log(`  LLM Tool Calls: ${JSON.stringify(step.toolCalls)}`);
    } else if (step.type === 'tool-results') {
      // Log the actual result returned by the tool's execute function
      console.log(`  Tool Results Received: ${JSON.stringify(step.toolResults)}`);
    }
    console.log(`  Usage for Step: ${JSON.stringify(step.usage)}`);
    console.log('  --------------------');
  });
  console.log('--------------------------');
}

solveAndShowSteps("Calculate (15 * 4) + (100 / 5)");
```

#### 7.6.3. Callback on Step Completion (`onStepFinish`)

**The "Why":** Sometimes you need to react *during* the agent's execution, not just inspect the `steps` array afterwards. You might want to log progress to a database, update a UI with intermediate status, save state, or even potentially interrupt the flow based on an intermediate result.

**The Solution:** Provide an `async` callback function to the `onStepFinish` option in `generateText` or `streamText`. This function receives the `TextStep` object (which contains the details of the step that just finished) allowing you to perform side effects.

```typescript
import { generateText, TextStep } from 'ai';
// Assume model and calculator tool defined

async function solveWithStepCallback(prompt: string) {
  let stepCounter = 0;

  console.log(`Solving with step callback: ${prompt}`);
  const result = await generateText({
    model: model,
    tools: { calculator: calculator },
    maxSteps: 5,
    system: 'Solve step-by-step, use calculator.',
    prompt: prompt,
    // This callback runs after each LLM response or tool execution completes
    onStepFinish: async (step: TextStep) => {
      stepCounter++;
      console.log(`\n---> Finishing Step ${stepCounter} (Type: ${step.type}) <---`);
      // You can inspect step.text, step.toolCalls, step.toolResults, step.usage etc.
      // Example: Log tool calls as they happen
      if (step.type === 'tool-calls') {
         console.log(`  [Callback] LLM calling tools: ${JSON.stringify(step.toolCalls)}`);
      }
      // Example: Log tool results as they are received
      if (step.type === 'tool-results') {
          console.log(`  [Callback] Received tool results: ${JSON.stringify(step.toolResults)}`);
      }
      // Add custom logic: save to DB, update UI status, etc.
      // await logStepToDatabase(step);
      console.log(`----------------------------------------`);
    },
  });

  console.log('\nFinal Agent Result:', result.text);
}

solveWithStepCallback("What is 25% of 80?");
```

### 7.7. Handling State and History (`responseMessages`)

**The "Why":** In multi-step tool usage (`maxSteps > 1`), the LLM needs the full context of the conversation, including the tool calls it made and the results it received, to reason effectively in subsequent steps. Simply appending the final `text` output isn't enough.

**The Solution:** The result object from `generateText` (and the `onFinish` event for `streamText`) contains `responseMessages`. This is an array of `CoreMessage` objects representing *all* the messages generated during *that specific multi-step call* by the assistant, including:
*   The assistant message(s) containing `toolCalls`.
*   The corresponding `tool` role messages containing the `toolResults`.
*   The final assistant message containing the textual response (`text`).

**Crucially, you should append this entire `responseMessages` array to your existing conversation history** before making the *next* call to the agent, ensuring it has the complete context.

```typescript
import { generateText, CoreMessage } from 'ai';
import { openai } from '@ai-sdk/openai';
// Assume model and calculator tool are defined

async function runMultiStepChatTurn(messages: CoreMessage[]): Promise<CoreMessage[]> {
  console.log("--- Running Agent Turn ---");
  console.log("Input Messages:", messages);

  const result = await generateText({
    model: model,
    tools: { calculator: calculator },
    maxSteps: 5,
    messages: messages, // Pass the current complete history
  });

  console.log("\nAgent's Final Text Output:", result.text);
  console.log("Messages generated this turn (responseMessages):", result.responseMessages);

  // **Correct History Update:** Append ALL messages from this turn
  const updatedMessages = [...messages, ...result.responseMessages];

  console.log("\nUpdated Full Message History:", updatedMessages);
  console.log("--------------------------\n");
  return updatedMessages; // Return the new history for the next turn
}

// Example Usage
async function simulateChat() {
  let conversationHistory: CoreMessage[] = [
    { role: 'user', content: 'What is 5 + 7?' }
  ];
  conversationHistory = await runMultiStepChatTurn(conversationHistory);
  // Output: History includes user msg, assistant tool call, tool result, assistant answer.

  // Add another user message
  conversationHistory.push({ role: 'user', content: 'Now multiply that result by 3.' });
  conversationHistory = await runMultiStepChatTurn(conversationHistory);
  // Output: History includes previous turn + new user msg, assistant tool call, tool result, assistant answer.
}

// simulateChat();
```

### 7.8. Controlling Tool Usage (`toolChoice`)

**The "Why":** You might not always want the LLM to freely decide whether or how to use tools. You might need to force a specific tool, prevent tool use entirely, or guarantee that *some* tool is used.

**The Solution:** The `toolChoice` parameter guides the LLM's tool usage:

*   `'auto'` (Default): The LLM decides whether to call a tool from the provided `tools` list or respond with text. Most flexible.
*   `'required'`: Forces the LLM to call *one* of the available tools. Useful with the "Structured Final Answer" pattern (7.6.1) or when the primary task *must* involve a tool.
*   `'none'`: Prevents the LLM from calling *any* tools, even if provided in the `tools` list. It must respond with text only.
*   `{ type: 'tool', toolName: 'yourToolName' }`: Forces the LLM to call the *specific* tool named `yourToolName`.
*   `{ type: 'function', functionName: 'yourFuncName' }`: Deprecated alias for the above.

```typescript
// Force the model to call the 'calculator' tool
const { toolCalls } = await generateText({
  model: model,
  tools: { calculator: calculator },
  prompt: "Calculate 100 / 4 please.",
  // Why: Ensure the calculator is used for this specific request.
  toolChoice: { type: 'tool', toolName: 'calculator' }
});
console.log('Forced Tool Call:', toolCalls);

// Prevent any tool usage, even if calculator is available
const { text, toolCalls: noToolCalls } = await generateText({
    model: model,
    tools: { calculator: calculator },
    prompt: "What is 2+2? Just give me the text answer.",
    // Why: Ensure a direct text response without involving tools.
    toolChoice: 'none'
});
console.log('No Tool Call Response:', text); // Should be '4' or similar text
console.log('Tool Calls Made:', noToolCalls); // Should be undefined or empty
```

### 7.9. Toolkits

Combine your custom tools with pre-built sets of tools (toolkits) from various providers that integrate with the AI SDK standard (e.g., agentic.ai, Composio, Browserbase). This can quickly add capabilities like web browsing, file system access, or interacting with specific APIs (Gmail, GitHub, etc.). Check the Vercel AI SDK documentation or toolkit providers for integration details.

### 7.10. Generative UI (`streamUI`)

A specialized use of the tool mechanism where tools don't `execute` backend logic but instead `generate` UI components (e.g., React Server Components). The LLM decides which UI component "tool" to call based on the conversation, allowing for dynamic, AI-driven interface generation. See `streamUI` documentation for details.

### 7.11. Agent Evaluation & Debugging

Evaluating agentic systems is challenging but essential.

*   **Key Metrics:**
    *   *Task Success:* Did the agent achieve the overall goal?
    *   *Tool Use Quality:* Correct tool selection? Correct arguments? Avoided hallucinating calls?
    *   *Efficiency:* Steps/tokens used? Latency? Cost?
    *   *Robustness:* How does it handle errors or unexpected tool results?
*   **Debugging with the SDK:**
    *   **Inspect `steps`:** The primary way to trace the agent's reasoning and actions (See 7.6.2). Look at the sequence of text, tool calls, and tool results.
    *   **Use `onStepFinish`:** Log detailed information *during* execution for real-time monitoring (See 7.6.3).
    *   **Tool Logging:** Add extensive logging *inside* your tool's `execute` functions to see inputs and outputs.
    *   **Examine `responseMessages`:** Ensure the correct history, including tool interactions, is maintained between turns (See 7.7).
*   **External Tools:** Consider platforms like LangSmith for more advanced tracing and evaluation capabilities.

## 8. Streaming Fundamentals

These hooks provide powerful building blocks for creating engaging and interactive AI-powered user interfaces with significantly less boilerplate code.

## 7. Agentic Systems: Enabling AI Actions and Complex Tasks

**Learning Objectives:** By the end of this section, you should be able to:
*   Explain what constitutes an "agentic" AI system and why it's needed.
*   Identify the core building blocks of agents: single-step generation, tool usage, and multi-step execution.
*   Describe common agent patterns like sequential processing, routing, and evaluation loops.
*   Define tools using the `tool` helper, including descriptions and Zod parameter schemas.
*   Use tools with `generateText` and `streamText`.
*   Implement multi-step tool usage with `maxSteps` to solve complex problems.
*   Structure final agent answers using a dedicated "answer tool" and `toolChoice: 'required'`.
*   Inspect agent execution using the `steps` property and `onStepFinish` callback for debugging and observability.
*   Manage conversation history correctly in multi-step scenarios using `responseMessages`.
*   Control when and how an agent uses tools via the `toolChoice` parameter.

**The "Why":** Beyond Conversation and Generation

So far, we've seen how to generate text, structured data, and build interactive UIs. But what if you need the AI to do more than just *respond*? What if you need it to **take actions**, interact with external systems, or solve problems that require multiple steps and access to real-time information or specific capabilities?

This is where **agentic systems** come in. An "agent" in this context is an AI system designed to:

1.  **Understand Intent:** Figure out the user's goal.
2.  **Plan:** Break down the goal into smaller steps (potentially involving different tools or information sources).
3.  **Act:** Execute those steps, often by calling tools (functions, APIs, database queries).
4.  **Observe:** Analyze the results of its actions.
5.  **Reason:** Decide the next step based on observations, potentially iterating until the goal is achieved.

**Analogy:** Think of a simple chatbot like a knowledgeable librarian who can answer questions based on the books they have (`generateText`). An agent is more like a research assistant who can not only consult books but also use a calculator (`calculatorTool`), search online databases (`searchTool`), summarize findings, and compile a structured report (`answerTool`), potentially performing these actions in sequence until the research task is complete.

The Vercel AI SDK provides the core functionalities (`generateText`, `streamText`, `tool`, `maxSteps`) to build these more sophisticated agentic systems.

### 7.1. Agent Building Blocks: From Simple to Complex

Agents are constructed by combining fundamental capabilities:

1.  **Single-Step LLM Generation:** The simplest form. A single call to `generateText` or `generateObject`. Suitable for direct tasks like classification, summarization, or simple Q&A where no external action or multi-step reasoning is needed.
    *   *Use Case:* Classifying customer feedback sentiment.
2.  **Tool Usage (Single Step):** Empowering the LLM with tools. A call to `generateText` or `streamText` where the model *might* choose to call one of the provided `tools` to fulfill the request in a single turn.
    *   *Use Case:* Answering "What's the weather in London?" by calling a weather API tool. Calculating "5! * 3" using a calculator tool.
3.  **Multi-Step Tool Usage:** The core of more complex agents. Allowing the LLM to make a *sequence* of tool calls (and reasoning steps) across multiple turns automatically, managed by the SDK using the `maxSteps` parameter. This enables solving problems where the path isn't known beforehand.
    *   *Use Case:* Planning a trip, which might involve calling tools to find flights, check hotel availability, and then summarize the plan. Debugging code by analyzing errors, searching for solutions, and proposing fixes iteratively.
4.  **Multi-Agent Systems (Advanced):** For highly complex tasks, you might orchestrate multiple specialized agents. One agent could act as a router, delegating sub-tasks to other agents with specific expertise or tools. While possible to build using the SDK's core functions, this pattern requires more custom application logic.
    *   *Use Case:* A complex financial analysis system where one agent retrieves market data, another performs technical analysis, and a third generates a report.

### 7.2. Common Agent Patterns (Application Logic)

While the SDK provides the building blocks, you often structure your agent's workflow using application-level patterns:

*   **Sequential Processing (Chains):** Steps executed in a fixed order. Output A -> Process B -> Output C. Simple and predictable.
    *   *SDK Implementation:* Multiple separate calls to `generateText`/`generateObject`, passing outputs as inputs.
    *   *Use Case:* Generate draft email -> Evaluate tone using `generateObject` -> Refine draft based on evaluation.
*   **Routing:** Use an initial LLM call (often `generateObject`) to decide which path, model, prompt, or toolset to use next.
    *   *SDK Implementation:* An initial `generateObject` call determines parameters for subsequent `generateText`/`streamText` calls.
    *   *Use Case:* Classify user intent (e.g., 'Sales Question', 'Support Issue') and route to different specialized prompts or tool-equipped agents.
*   **ReAct (Reasoning and Acting):** A common pattern where the LLM explicitly reasons about the next step, decides on an action (often a tool call), executes it, observes the result, and repeats. The SDK's multi-step tool usage (`maxSteps`) automates a form of this loop.
*   **Evaluator/Refinement Loops:** Use one LLM call to generate an output and another (potentially `generateObject` with criteria) to evaluate it. Feed the evaluation back to the generator for refinement until a quality threshold is met.
    *   *SDK Implementation:* A loop in your application code calling `generateText` and `generateObject` iteratively.
    *   *Use Case:* Generating code, then using another LLM call to check for errors or adherence to style guides, then refining the code.

### 7.3. Choosing Your Agent Design

Consider these trade-offs:

*   **Flexibility vs. Control:** More steps and tool autonomy (`maxSteps > 1`) offer flexibility but less predictability. Single steps or fixed chains offer more control.
*   **Complexity & Cost:** Multi-step agents are powerful but involve more LLM calls (higher latency and cost) and complexity.
*   **Error Handling:** Autonomous agents require robust error handling within tools and potentially logic to recover from failed steps.
*   **Maintainability:** Start simple. Introduce tools and multi-step execution only when necessary.

**Recommendation:** Begin with the simplest approach that meets the need (single step, maybe one tool). Incrementally add multi-step execution (`maxSteps`) or more complex patterns if required.

### 7.4. Defining Tools: Giving the LLM Capabilities

Tools are the core mechanism for agents to interact with the world beyond text generation.

*   **Definition:** Use the `tool` helper function from the `ai` package.
*   **`description` (Crucial!):** Explain clearly and concisely *what the tool does* and *when the LLM should use it*. Provide examples if helpful. This is the LLM's primary guide.
*   **`parameters` (Schema):** Define the expected inputs using a Zod schema (`zod`) or JSON schema (`jsonSchema`). Include `.describe()` for each parameter to clarify its purpose for the LLM. Use `z.object({})` for tools with no arguments.
*   **`execute` (Optional but common):** An `async` function containing the tool's logic. It receives the parameters (validated against the schema) and should return the result the LLM needs. If `execute` is omitted, the SDK will return the tool call details, and your application logic must handle the execution.

```typescript
import { tool } from 'ai';
import { z } from 'zod'; // Make sure to install Zod: npm install zod
import * as mathjs from 'mathjs'; // Example dependency: npm install mathjs
// Assume fetchWeatherData is an external function you've defined elsewhere
// import { fetchWeatherData } from './api/weather';

// Example: Calculator Tool
export const calculatorTool = tool({
  // Why: Clearly state the tool's purpose and usage context.
  description:
    'Calculates the result of a mathematical expression. ' +
    'Use this for arithmetic, trigonometry, etc. ' +
    "Example expressions: '1.2 * (2 + 4.5)', '12.7 cm to inch', 'sin(45 deg) ^ 2'.",
  parameters: z.object({
    // Why: Define expected input with type and description.
    expression: z.string().describe('The mathematical expression to evaluate. Ensure it\'s a valid expression for mathjs.'),
  }),
  // Why: The function that performs the actual work when the tool is called.
  execute: async ({ expression }) => {
    console.log(`Executing calculator tool with expression: ${expression}`);
    try {
      // Safely evaluate the expression
      const result = mathjs.evaluate(expression);
      // Return the result to the LLM
      return result;
    } catch (error) {
      // Handle potential errors during evaluation and inform the LLM
      console.error("Calculator tool error:", error);
      return `Error evaluating expression "${expression}": ${error instanceof Error ? error.message : 'Unknown error'}`;
    }
  },
});

// Example: Weather Tool (execute handled by application or omitted for now)
export const weatherTool = tool({
  description: 'Gets the current weather for a specific location.',
  parameters: z.object({
    location: z.string().describe('The city and state, e.g., San Francisco, CA'),
    unit: z.enum(['celsius', 'fahrenheit']).default('celsius').describe("Temperature unit")
  }),
  // execute: async ({ location, unit }) => {
  //   console.log(`Fetching weather for ${location} in ${unit}`);
  //   return await fetchWeatherData(location, unit); // Call your API logic
  // }
  // If execute is omitted, the LLM call will return a toolCall object
  // with toolName 'weather' and args { location: '...', unit: '...' }
});
```

### 7.5. Using Tools with `generateText` and `streamText`

Pass your defined tools to the LLM call via the `tools` parameter. This is an object where keys are the names the LLM will use to refer to the tool, and values are the tool definitions created with the `tool` helper.

```typescript
import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';
import { calculatorTool, weatherTool } from './tools'; // Assume tools are imported

const model = openai('gpt-4o-mini'); // Choose a model good at tool use

async function askWithTools(prompt: string) {
  console.log(`Asking: ${prompt}`);
  const { text, toolCalls, toolResults, finishReason } = await generateText({
    model: model,
    // Provide the tools the LLM can use
    tools: {
      calculate: calculatorTool, // LLM refers to this tool as 'calculate'
      getCurrentWeather: weatherTool // LLM refers to this as 'getCurrentWeather'
    },
    prompt: prompt,
  });

  // Analyze the result:
  console.log('LLM Finish Reason:', finishReason); // 'stop', 'tool-calls'
  console.log('LLM Text Response:', text); // Text generated by the LLM (might be empty if only tools were called)

  if (toolCalls?.length) {
    console.log('LLM Tool Calls:', toolCalls);
    // Example: [{ toolCallId: '...', toolName: 'calculate', args: { expression: '2+2' } }]
    // If tools had no 'execute', your application logic would run them here.
  }

  if (toolResults?.length) {
    console.log('Tool Execution Results:', toolResults);
    // Example: [{ toolCallId: '...', toolName: 'calculate', result: 4 }]
    // Populated if the tool definition included an 'execute' function that ran.
  }
}

// Example where the LLM likely uses the calculator
askWithTools("What is the result of log10(100) * 5?");

// Example where the LLM likely calls the weather tool (if execute were defined)
// askWithTools("What's the weather like in Paris in fahrenheit?");
```

### 7.6. Multi-Step Tool Usage: Solving Complex Problems (`maxSteps`)

**The "Why":** Many problems can't be solved in one shot. They require a sequence of actions, where the result of one step informs the next. Planning a trip, debugging code, or complex calculations often fall into this category. The exact sequence might not be known beforehand.

**The Solution:** The `maxSteps` parameter in `generateText` and `streamText` enables the LLM to perform multiple reasoning steps and tool calls autonomously within a single SDK call.

**Analogy:** It's like giving your research assistant (the LLM) permission to not just use the calculator once, but to perform a series of calculations, lookups, or other tool actions as needed to arrive at the final answer, up to a certain limit (`maxSteps`).

**The Automatic Workflow:**
1.  Your initial prompt is sent to the LLM.
2.  LLM Response:
    *   If it's just text: The process stops, and the text is returned.
    *   If it includes a `toolCall` for a tool *with* an `execute` function:
        a.  The SDK pauses LLM generation.
        b.  The SDK calls your tool's `execute` function with the arguments provided by the LLM.
        c.  The SDK sends the `toolResult` back to the LLM as context for the *next* step.
        d.  The LLM continues, now aware of the tool's result.
    *   If it includes a `toolCall` for a tool *without* an `execute` function: The process stops, returning the `toolCall` details for your application to handle.
3.  Looping: If a tool with `execute` was called, the process loops back to Step 2 (LLM generates based on new context).
4.  Termination: The loop stops when:
    *   The LLM generates only text.
    *   The `maxSteps` limit (counting LLM generation turns + tool execution turns) is reached.
    *   A tool *without* `execute` is called.
    *   An error occurs.

```typescript
import { z } from 'zod';
import { generateText, tool } from 'ai';
import { openai } from '@ai-sdk/openai';
import * as mathjs from 'mathjs'; // Requires: npm install mathjs

const model = openai('gpt-4o-mini');
const calculator = tool({
  description: 'Evaluate mathematical expressions.',
  parameters: z.object({ expression: z.string() }),
  execute: async ({ expression }) => {
     console.log(`   [Calculator executing: ${expression}]`);
     return mathjs.evaluate(expression);
  }
});

async function solveMultiStepProblem(prompt: string) {
  console.log(`Solving multi-step: ${prompt}`);
  const { text, toolCalls, toolResults, finishReason, usage, steps } = await generateText({
    model: model,
    tools: {
      calculator: calculator // Provide the tool with its execute function
    },
    maxSteps: 5, // Allow up to 5 steps (LLM calls + tool executions)
    system: 'You solve math problems step-by-step. Use the calculator when needed. Explain your final answer.',
    prompt: prompt,
  });

  console.log('\n--- Multi-Step Result ---');
  console.log('Final Text:', text); // The final text output from the LLM
  console.log('Finish Reason:', finishReason); // e.g., 'stop', 'max-steps', 'tool-calls' (if final step was a call)
  console.log('Total Usage:', usage);
  console.log('-----------------------');

  // You can also inspect the intermediate steps (see 7.6.2)
  // console.log('\nIntermediate Steps:', steps);
}

// Example requiring multiple calculation steps
solveMultiStepProblem("A company's revenue is $10,000. Cost of Goods Sold is $4,500. Operating Expenses are $2,000. What is the net profit margin percentage?");
// Expected flow:
// 1. LLM asks calculator for 10000 - 4500 (Gross Profit) -> Result: 5500
// 2. LLM asks calculator for 5500 - 2000 (Net Profit) -> Result: 3500
// 3. LLM asks calculator for (3500 / 10000) * 100 (Margin %) -> Result: 35
// 4. LLM generates final text explaining the result is 35%.
```

#### 7.6.1. Structured Final Answers (`toolChoice: 'required'` + Answer Tool)

**The "Why":** Sometimes, especially after a multi-step process, you don't want just a text explanation; you need the agent's final answer in a reliable, structured format (JSON), similar to `generateObject` but at the *end* of an agentic workflow.

**The Solution:** Define a special "answer tool" whose `parameters` schema matches your desired final JSON structure. Crucially, **do not provide an `execute` function** for this tool. Then, in your `generateText` call:
1.  Include the answer tool in the `tools` list alongside operational tools (like `calculator`).
2.  Set `toolChoice: 'required'`.

This forces the LLM's *very last step* to be a call to one of the provided tools. By carefully prompting the LLM to use the specific "answer tool" when it has finished its reasoning and calculations, you guide it to structure its final output according to your schema. The result will be in the `toolCalls` array.

```typescript
import { z } from 'zod';
import { generateText, tool } from 'ai';
import { openai } from '@ai-sdk/openai';
// Assume calculator tool is defined as above

// 1. Define the schema for the final structured answer
const calculationAnswerSchema = z.object({
  reasoningSteps: z.array(
    z.object({
      stepDescription: z.string(),
      calculation: z.string().optional(), // e.g., "10000 - 4500"
      result: z.union([z.string(), z.number()]).optional(), // e.g., 5500
    }),
  ).describe("Step-by-step reasoning and calculations performed."),
  finalAnswer: z.number().describe("The final numerical answer."),
  explanation: z.string().describe("Brief textual explanation of the result and process."),
});

// 2. Define the 'answer' tool with the schema but NO execute function
const finalAnswerTool = tool({
  description: 'Use this tool ONLY to provide the final answer, reasoning steps, calculations, and explanation once the problem is fully solved.',
  parameters: calculationAnswerSchema,
  // NO 'execute' - calling this tool terminates the agent loop successfully.
});

async function solveAndStructure(prompt: string) {
  const model = openai('gpt-4o-mini'); // Model good at tool use

  console.log(`Solving and structuring: ${prompt}`);
  const { text, toolCalls, finishReason, steps } = await generateText({
    model: model,
    tools: {
      // Provide operational tools...
      calculator: calculator,
      // ...and the final answer structuring tool
      provideFinalAnswer: finalAnswerTool // LLM refers to it by this name
    },
    // 3. Force the final step to be a tool call
    toolChoice: 'required',
    maxSteps: 10, // Allow enough steps for calculation and final structuring
    // 4. Prompt instructs use of calculator AND the final answer tool
    system: `Solve the math problem step-by-step. Use the 'calculator' tool for all calculations.
             When you have the final answer and all steps, you MUST call the 'provideFinalAnswer' tool with the detailed steps, calculations, final numeric answer, and a textual explanation. Do not provide the final answer in plain text.`,
    prompt: prompt,
  });

  console.log('\n--- Structured Answer Result ---');
  console.log('Finish Reason:', finishReason); // Should ideally be 'tool-calls'

  // 5. Extract the structured answer from the toolCalls array
  const answerCall = toolCalls?.find(call => call.toolName === 'provideFinalAnswer');

  if (answerCall) {
    // The validated arguments of the 'provideFinalAnswer' call ARE the structured result
    const structuredOutput = answerCall.args;
    console.log('Structured Answer Extracted:');
    console.log('  Final Numeric Answer:', structuredOutput.finalAnswer);
    console.log('  Explanation:', structuredOutput.explanation);
    console.log('  Reasoning Steps:', JSON.stringify(structuredOutput.reasoningSteps, null, 2));
  } else {
    console.log('Error: Final answer tool was not called.');
    console.log('Final Text:', text); // Log text for debugging
    console.log('All Tool Calls:', toolCalls); // Log all calls for debugging
  }
  console.log('------------------------------');
}

solveAndStructure("Revenue $10k, COGS $4.5k, OpEx $2k. Net profit margin %?");
```
*Note:* Success depends heavily on the model's ability to follow the prompt instructing it to use the specific answer tool *last*. `toolChoice: 'required'` ensures *a* tool is called last, but the prompt guides *which* tool.

#### 7.6.2. Accessing Intermediate Steps (`steps` Property)

**The "Why":** When an agent takes multiple steps, especially autonomously, understanding *how* it arrived at the result is crucial for debugging, evaluation, and building trust. You need to see the intermediate thoughts, actions, and observations.

**The Solution:** The result object returned by `generateText` (when `maxSteps > 0`) includes a `steps` property. This is an array containing the detailed history of the execution flow, step-by-step.

Each element in the `steps` array represents one turn and includes:
*   `type`: 'text', 'tool-calls', or 'tool-results'.
*   The corresponding data (`text`, `toolCalls`, or `toolResults`).
*   `usage`: Token usage for that specific step.

```typescript
// Building on the multi-step example:
async function solveAndShowSteps(prompt: string) {
  const { text, finishReason, steps } = await generateText({
    model: model,
    tools: { calculator: calculator },
    maxSteps: 5,
    system: 'Solve step-by-step, use calculator.',
    prompt: prompt,
  });

  console.log('Final Text:', text);
  console.log('Finish Reason:', finishReason);

  console.log('\n--- Execution Steps Trace ---');
  steps.forEach((step, index) => {
    console.log(`Step ${index + 1}:`);
    console.log(`  Type: ${step.type}`);
    if (step.type === 'text') {
      console.log(`  LLM Text Output: "${step.text}"`);
    } else if (step.type === 'tool-calls') {
      // Log the arguments the LLM decided to use for the tool
      console.log(`  LLM Tool Calls: ${JSON.stringify(step.toolCalls)}`);
    } else if (step.type === 'tool-results') {
      // Log the actual result returned by the tool's execute function
      console.log(`  Tool Results Received: ${JSON.stringify(step.toolResults)}`);
    }
    console.log(`  Usage for Step: ${JSON.stringify(step.usage)}`);
    console.log('  --------------------');
  });
  console.log('--------------------------');
}

solveAndShowSteps("Calculate (15 * 4) + (100 / 5)");
```

#### 7.6.3. Callback on Step Completion (`onStepFinish`)

**The "Why":** Sometimes you need to react *during* the agent's execution, not just inspect the `steps` array afterwards. You might want to log progress to a database, update a UI with intermediate status, save state, or even potentially interrupt the flow based on an intermediate result.

**The Solution:** Provide an `async` callback function to the `onStepFinish` option in `generateText` or `streamText`. This function receives the `TextStep` object (which contains the details of the step that just finished) allowing you to perform side effects.

```typescript
import { generateText, TextStep } from 'ai';
// Assume model and calculator tool defined

async function solveWithStepCallback(prompt: string) {
  let stepCounter = 0;

  console.log(`Solving with step callback: ${prompt}`);
  const result = await generateText({
    model: model,
    tools: { calculator: calculator },
    maxSteps: 5,
    system: 'Solve step-by-step, use calculator.',
    prompt: prompt,
    // This callback runs after each LLM response or tool execution completes
    onStepFinish: async (step: TextStep) => {
      stepCounter++;
      console.log(`\n---> Finishing Step ${stepCounter} (Type: ${step.type}) <---`);
      // You can inspect step.text, step.toolCalls, step.toolResults, step.usage etc.
      // Example: Log tool calls as they happen
      if (step.type === 'tool-calls') {
         console.log(`  [Callback] LLM calling tools: ${JSON.stringify(step.toolCalls)}`);
      }
      // Example: Log tool results as they are received
      if (step.type === 'tool-results') {
          console.log(`  [Callback] Received tool results: ${JSON.stringify(step.toolResults)}`);
      }
      // Add custom logic: save to DB, update UI status, etc.
      // await logStepToDatabase(step);
      console.log(`----------------------------------------`);
    },
  });

  console.log('\nFinal Agent Result:', result.text);
}

solveWithStepCallback("What is 25% of 80?");
```

### 7.7. Handling State and History (`responseMessages`)

**The "Why":** In multi-step tool usage (`maxSteps > 1`), the LLM needs the full context of the conversation, including the tool calls it made and the results it received, to reason effectively in subsequent steps. Simply appending the final `text` output isn't enough.

**The Solution:** The result object from `generateText` (and the `onFinish` event for `streamText`) contains `responseMessages`. This is an array of `CoreMessage` objects representing *all* the messages generated during *that specific multi-step call* by the assistant, including:
*   The assistant message(s) containing `toolCalls`.
*   The corresponding `tool` role messages containing the `toolResults`.
*   The final assistant message containing the textual response (`text`).

**Crucially, you should append this entire `responseMessages` array to your existing conversation history** before making the *next* call to the agent, ensuring it has the complete context.

```typescript
import { generateText, CoreMessage } from 'ai';
import { openai } from '@ai-sdk/openai';
// Assume model and calculator tool are defined

async function runMultiStepChatTurn(messages: CoreMessage[]): Promise<CoreMessage[]> {
  console.log("--- Running Agent Turn ---");
  console.log("Input Messages:", messages);

  const result = await generateText({
    model: model,
    tools: { calculator: calculator },
    maxSteps: 5,
    messages: messages, // Pass the current complete history
  });

  console.log("\nAgent's Final Text Output:", result.text);
  console.log("Messages generated this turn (responseMessages):", result.responseMessages);

  // **Correct History Update:** Append ALL messages from this turn
  const updatedMessages = [...messages, ...result.responseMessages];

  console.log("\nUpdated Full Message History:", updatedMessages);
  console.log("--------------------------\n");
  return updatedMessages; // Return the new history for the next turn
}

// Example Usage
async function simulateChat() {
  let conversationHistory: CoreMessage[] = [
    { role: 'user', content: 'What is 5 + 7?' }
  ];
  conversationHistory = await runMultiStepChatTurn(conversationHistory);
  // Output: History includes user msg, assistant tool call, tool result, assistant answer.

  // Add another user message
  conversationHistory.push({ role: 'user', content: 'Now multiply that result by 3.' });
  conversationHistory = await runMultiStepChatTurn(conversationHistory);
  // Output: History includes previous turn + new user msg, assistant tool call, tool result, assistant answer.
}

// simulateChat();
```

### 7.8. Controlling Tool Usage (`toolChoice`)

**The "Why":** You might not always want the LLM to freely decide whether or how to use tools. You might need to force a specific tool, prevent tool use entirely, or guarantee that *some* tool is used.

**The Solution:** The `toolChoice` parameter guides the LLM's tool usage:

*   `'auto'` (Default): The LLM decides whether to call a tool from the provided `tools` list or respond with text. Most flexible.
*   `'required'`: Forces the LLM to call *one* of the available tools. Useful with the "Structured Final Answer" pattern (7.6.1) or when the primary task *must* involve a tool.
*   `'none'`: Prevents the LLM from calling *any* tools, even if provided in the `tools` list. It must respond with text only.
*   `{ type: 'tool', toolName: 'yourToolName' }`: Forces the LLM to call the *specific* tool named `yourToolName`.
*   `{ type: 'function', functionName: 'yourFuncName' }`: Deprecated alias for the above.

```typescript
// Force the model to call the 'calculator' tool
const { toolCalls } = await generateText({
  model: model,
  tools: { calculator: calculator },
  prompt: "Calculate 100 / 4 please.",
  // Why: Ensure the calculator is used for this specific request.
  toolChoice: { type: 'tool', toolName: 'calculator' }
});
console.log('Forced Tool Call:', toolCalls);

// Prevent any tool usage, even if calculator is available
const { text, toolCalls: noToolCalls } = await generateText({
    model: model,
    tools: { calculator: calculator },
    prompt: "What is 2+2? Just give me the text answer.",
    // Why: Ensure a direct text response without involving tools.
    toolChoice: 'none'
});
console.log('No Tool Call Response:', text); // Should be '4' or similar text
console.log('Tool Calls Made:', noToolCalls); // Should be undefined or empty
```

### 7.9. Toolkits

Combine your custom tools with pre-built sets of tools (toolkits) from various providers that integrate with the AI SDK standard (e.g., agentic.ai, Composio, Browserbase). This can quickly add capabilities like web browsing, file system access, or interacting with specific APIs (Gmail, GitHub, etc.). Check the Vercel AI SDK documentation or toolkit providers for integration details.

### 7.10. Generative UI (`streamUI`)

A specialized use of the tool mechanism where tools don't `execute` backend logic but instead `generate` UI components (e.g., React Server Components). The LLM decides which UI component "tool" to call based on the conversation, allowing for dynamic, AI-driven interface generation. See `streamUI` documentation for details.

### 7.11. Agent Evaluation & Debugging

Evaluating agentic systems is challenging but essential.

*   **Key Metrics:**
    *   *Task Success:* Did the agent achieve the overall goal?
    *   *Tool Use Quality:* Correct tool selection? Correct arguments? Avoided hallucinating calls?
    *   *Efficiency:* Steps/tokens used? Latency? Cost?
    *   *Robustness:* How does it handle errors or unexpected tool results?
*   **Debugging with the SDK:**
    *   **Inspect `steps`:** The primary way to trace the agent's reasoning and actions (See 7.6.2). Look at the sequence of text, tool calls, and tool results.
    *   **Use `onStepFinish`:** Log detailed information *during* execution for real-time monitoring (See 7.6.3).
    *   **Tool Logging:** Add extensive logging *inside* your tool's `execute` functions to see inputs and outputs.
    *   **Examine `responseMessages`:** Ensure the correct history, including tool interactions, is maintained between turns (See 7.7).
*   **External Tools:** Consider platforms like LangSmith for more advanced tracing and evaluation capabilities.
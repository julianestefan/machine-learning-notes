# Vector Databases with Pinecone

## Introduction

Vector databases store and efficiently query high-dimensional vector representations of data. They are essential components of modern AI applications, particularly those involving semantic search, recommendation systems, and retrieval-augmented generation (RAG).

Pinecone is a managed vector database designed specifically for machine learning applications. It provides fast similarity search for vectors with low latency at scale.

## Index Types

Pinecone offers two main index types, each with different characteristics:

| Feature | Pod-based | Serverless |
| --- | --- | --- |
| Resource management | Manual | Automatic |
| Scaling | Manual | Automatic |
| Hardware selection | Customizable | Managed |
| Cost model | Fixed with predictable costs | Pay-per-use |
| Best for | Production workloads with predictable usage | Development, variable workloads |

### Pod-based Indexes

Pod-based indexes allow you to choose specific hardware configurations that determine:

* Storage capacity
* Query latency
* Query throughput
* Vector dimensions supported

Pod types include:

| Pod Type | Description | Use Case |
| --- | --- | --- |
| s1 | Cost-optimized for storage | Large datasets with moderate performance needs |
| p1 | Balanced performance | General-purpose vector search |
| p2 | High performance | Low-latency, high-throughput applications |

Each pod type comes in different sizes (x1, x2, x4, x8) that determine the resources allocated.

### Serverless Indexes

Serverless indexes automatically handle resource management:

* Scale to zero when not in use
* Auto-scale with query volume
* Pay only for what you use
* No pod configuration required

```python
# Import the Pinecone library
from pinecone import Pinecone, ServerlessSpec, PodSpec

# Initialize the Pinecone client
pc = Pinecone(api_key="your_api_key")

# Create a serverless index
pc.create_index(
    name="serverless-index",
    dimension=1536,  # OpenAI embedding dimension
    metric="cosine",  # Distance metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# Create a pod-based index
pc.create_index(
    name="pod-index",
    dimension=1536,
    metric="cosine",
    spec=PodSpec(
        environment="gcp-starter",
        pod_type="p1.x1"  # Pod type and size
    )
)
```

## Index Management

### Connecting to Indexes

```python
# Import the Pinecone library
from pinecone import Pinecone

# Initialize the Pinecone client
pc = Pinecone(api_key="your_api_key")

# Connect to your existing index
index = pc.Index("my-index")

# Print the index statistics
stats = index.describe_index_stats()
print(f"Total vector count: {stats['total_vector_count']}")
print(f"Namespaces: {stats['namespaces']}")
```

### Listing and Deleting Indexes

```python
# Import the Pinecone library
from pinecone import Pinecone

# Initialize the Pinecone client
pc = Pinecone(api_key="your_api_key")

# List all available indexes
indexes = pc.list_indexes()
print(f"Available indexes: {indexes}")

# Delete a specific index
pc.delete_index("index-to-delete")
```

### Namespaces

Namespaces allow partitioning vectors within a single index:

* **Use cases**: Multitenancy, data versioning, logical separation
* **Isolation**: Queries only search within the specified namespace
* **Performance**: No overhead compared to a non-namespaced approach

### Organization Structure

Pinecone organizes resources hierarchically:

* **Organization**: Top-level container for all resources
* **Projects**: Groups of related indexes within an organization
* **Indexes**: Individual vector databases within a project
* **Namespaces**: Partitions within an index

### Access Control

Pinecone provides role-based access control:

* **Organization Owner**: Full access to all resources
* **Organization User**: Access to specific projects they're invited to
* **Project Owner**: Full access to specific project resources

## Vector Operations

### Vector Ingestion

Vectors are inserted or updated using the `upsert` operation:

```python
# Import necessary libraries
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# Initialize clients
pc = Pinecone(api_key="your_pinecone_api_key")
openai_client = OpenAI(api_key="your_openai_api_key")

# Create or connect to an index
index_name = "document-index"
dimension = 1536  # OpenAI embedding dimension

# Check if index exists, if not create it
if index_name not in pc.list_indexes():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Sample data
documents = [
    "Artificial intelligence is transforming how we interact with technology.",
    "Vector databases store and query embeddings for semantic search.",
    "Machine learning models require large datasets for training."
]

# Create unique IDs
ids = [f"doc_{i}" for i in range(len(documents))]

# Generate embeddings
response = openai_client.embeddings.create(
    input=documents,
    model="text-embedding-3-small"
)
embeddings = [item.embedding for item in response.data]

# Prepare vector records with metadata
vector_records = [
    {
        "id": ids[i],
        "values": embeddings[i],
        "metadata": {"text": documents[i], "source": "example", "length": len(documents[i])}
    }
    for i in range(len(documents))
]

# Upsert vectors into the index
index.upsert(vectors=vector_records)

# Verify the insertion
stats = index.describe_index_stats()
print(f"Total vectors: {stats['total_vector_count']}")
```

### Fetching Vectors by ID

Retrieve specific vectors using their IDs:

```python
# Import the Pinecone library
from pinecone import Pinecone

# Initialize the Pinecone client and connect to index
pc = Pinecone(api_key="your_api_key")
index = pc.Index("document-index")

# Fetch specific vectors by ID
ids = ["doc_0", "doc_1"]
fetched_vectors = index.fetch(ids=ids)

# Process the results
for vec_id, vector_data in fetched_vectors["vectors"].items():
    print(f"ID: {vec_id}")
    print(f"Metadata: {vector_data['metadata']}")
    print(f"Vector dimension: {len(vector_data['values'])}")
    print("---")
```

### Querying for Similar Vectors

Find vectors most similar to a query vector:

```python
# Import necessary libraries
from pinecone import Pinecone
from openai import OpenAI

# Initialize clients
pc = Pinecone(api_key="your_pinecone_api_key")
openai_client = OpenAI(api_key="your_openai_api_key")
index = pc.Index("document-index")

# Create a query embedding
query_text = "How do vector databases work?"
query_response = openai_client.embeddings.create(
    input=query_text,
    model="text-embedding-3-small"
)
query_embedding = query_response.data[0].embedding

# Query the index
query_result = index.query(
    vector=query_embedding,
    top_k=3,               # Return top 3 matches
    include_values=True,   # Include vector values
    include_metadata=True  # Include metadata
)

# Process and display results
print(f"Query: {query_text}\n")
print("Top matches:")
for match in query_result["matches"]:
    print(f"ID: {match['id']}")
    print(f"Score: {match['score']:.4f}")
    print(f"Text: {match['metadata']['text']}")
    print("---")
```

### Distance Metrics

Pinecone supports multiple distance metrics for similarity calculations:

| Metric | Description | Best For |
| --- | --- | --- |
| Cosine | Measures angle between vectors | Text embeddings, normalized vectors |
| Euclidean | Measures straight-line distance | Spatial data, non-normalized vectors |
| Dot Product | Measures directional similarity | When vector magnitude matters |

```python
# Specify distance metric during index creation
pc.create_index(
    name="euclidean-index",
    dimension=1536,
    metric="euclidean",  # Options: "cosine", "euclidean", "dotproduct"
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)
```

### Metadata Filtering

Filter query results based on metadata values:

```python
# Import necessary libraries
from pinecone import Pinecone
from openai import OpenAI

# Initialize clients
pc = Pinecone(api_key="your_pinecone_api_key")
openai_client = OpenAI(api_key="your_openai_api_key")
index = pc.Index("movie-index")

# Create a query embedding
query_text = "Exciting action movie"
query_response = openai_client.embeddings.create(
    input=query_text,
    model="text-embedding-3-small"
)
query_embedding = query_response.data[0].embedding

# Query with metadata filter
query_result = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True,
    filter={
        "genre": {"$in": ["action", "thriller"]},  # Genre is action OR thriller
        "year": {"$gte": 2020},                   # Year >= 2020
        "rating": {"$gt": 7.5}                    # Rating > 7.5
    }
)

# Process and display results
print(f"Query: {query_text} (with filters)\n")
for match in query_result["matches"]:
    print(f"ID: {match['id']}")
    print(f"Score: {match['score']:.4f}")
    print(f"Title: {match['metadata']['title']}")
    print(f"Genre: {match['metadata']['genre']}")
    print(f"Year: {match['metadata']['year']}")
    print("---")
```

### Updating Vectors

Update existing vectors or their metadata:

```python
# Import the Pinecone library
from pinecone import Pinecone

# Initialize the Pinecone client
pc = Pinecone(api_key="your_api_key")
index = pc.Index("movie-index")

# Update a vector's values and metadata
index.update(
    id="movie_42",
    values=new_embedding,  # New vector values
    set_metadata={
        "genre": "thriller",
        "year": 2024,
        "watched": True
    }
)

# Update only metadata (keeping vector values the same)
index.update(
    id="movie_42",
    set_metadata={"rating": 8.5, "updated_at": "2023-12-15"}
)
```

### Deleting Vectors

Remove vectors using IDs or metadata filtering:

```python
# Import the Pinecone library
from pinecone import Pinecone

# Initialize the Pinecone client
pc = Pinecone(api_key="your_api_key")
index = pc.Index("movie-index")

# Delete specific vectors by ID
index.delete(ids=["movie_13", "movie_27"])

# Delete vectors using metadata filtering
index.delete(
    filter={
        "genre": {"$eq": "horror"},
        "rating": {"$lt": 5.0}
    }
)

# Delete all vectors in a namespace
index.delete(delete_all=True, namespace="old_movies")
```

## Performance Optimization

### Batch Processing

Processing vectors in batches improves throughput:

```python
# Import necessary libraries
import itertools
from pinecone import Pinecone
import numpy as np
from openai import OpenAI

# Initialize clients
pc = Pinecone(api_key="your_pinecone_api_key")
openai_client = OpenAI(api_key="your_openai_api_key")
index = pc.Index("document-index")

def chunks(iterable, batch_size=100):
    """Split an iterable into chunks of specified size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

# Sample large dataset
documents = ["Document " + str(i) for i in range(1000)]

# Process and upsert in batches
batch_size = 100
for i, doc_batch in enumerate(chunks(documents, batch_size)):
    print(f"Processing batch {i+1}/{len(documents)//batch_size}")
    
    # Generate embeddings for the batch
    response = openai_client.embeddings.create(
        input=doc_batch,
        model="text-embedding-3-small"
    )
    embeddings = [item.embedding for item in response.data]
    
    # Create vector records
    vector_batch = [
        {
            "id": f"doc_{i*batch_size+j}",
            "values": embeddings[j],
            "metadata": {"text": doc_batch[j], "batch": i}
        }
        for j in range(len(doc_batch))
    ]
    
    # Upsert the batch
    index.upsert(vectors=vector_batch)

print("Batch processing complete")
```

### Asynchronous Operations

Use asynchronous requests for higher throughput:

```python
# Import necessary libraries
import itertools
from pinecone import Pinecone
import numpy as np
from openai import OpenAI
import time

# Initialize clients
pc = Pinecone(api_key="your_pinecone_api_key")
openai_client = OpenAI(api_key="your_openai_api_key")

# Chunking function
def chunks(iterable, batch_size=100):
    """Split an iterable into chunks of specified size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

# Sample data (pre-generated vectors)
vector_data = [
    {"id": f"vec_{i}", "values": np.random.rand(1536).tolist(), "metadata": {"idx": i}}
    for i in range(10000)
]

# Process asynchronously with thread pooling
start_time = time.time()
with pc.Index("async-index", pool_threads=20) as index:
    # Submit all batch requests asynchronously
    async_results = [
        index.upsert(vectors=chunk, async_req=True) 
        for chunk in chunks(vector_data, batch_size=200)
    ]
    
    # Wait for all requests to complete
    for i, async_result in enumerate(async_results):
        async_result.get()  # This blocks until the request completes
        if (i+1) % 10 == 0:
            print(f"Completed {i+1}/{len(async_results)} batches")

elapsed = time.time() - start_time
print(f"Asynchronous processing complete in {elapsed:.2f} seconds")

# Verify the insertion
index = pc.Index("async-index")
stats = index.describe_index_stats()
print(f"Total vectors: {stats['total_vector_count']}")
```

## Multi-Tenancy Approaches

Pinecone offers multiple approaches for handling multi-tenant applications:

| Approach | Pros | Cons |
| --- | --- | --- |
| Namespaces | Lowest cost, simple management | Limited isolation |
| Metadata filtering | Flexible querying | Potential performance impact |
| Separate indexes | Complete isolation | Higher cost, more complex |

### Using Namespaces

Namespaces provide logical separation within a single index:

```python
# Import the Pinecone library
from pinecone import Pinecone

# Initialize the Pinecone client
pc = Pinecone(api_key="your_api_key")
index = pc.Index("multi-tenant-index")

# Add vectors for tenant A
index.upsert(
    vectors=tenant_a_vectors,
    namespace="tenant_a"  # Specify the namespace
)

# Add vectors for tenant B
index.upsert(
    vectors=tenant_b_vectors,
    namespace="tenant_b"
)

# Query within tenant A's namespace only
query_result = index.query(
    vector=query_embedding,
    namespace="tenant_a",
    top_k=5
)
```

## Practical Applications

### Semantic Search Implementation

```python
# Import necessary libraries
import pandas as pd
import numpy as np
from uuid import uuid4
from pinecone import Pinecone
from openai import OpenAI

# Initialize clients
pc = Pinecone(api_key="your_pinecone_api_key")
openai_client = OpenAI(api_key="your_openai_api_key")

# Connect to index
index = pc.Index("knowledge-base")

# Example dataset (could be loaded from CSV/database)
df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'title': [
        'Introduction to Vector Databases',
        'How to Implement Semantic Search',
        'Understanding Embeddings',
        'Building RAG Applications',
        'Vector Database Performance'
    ],
    'text': [
        'Vector databases are specialized databases designed to store and query vector embeddings efficiently.',
        'Semantic search uses vector embeddings to find results based on meaning rather than keywords.',
        'Embeddings are numerical representations of data that capture semantic meaning in a vector space.',
        'Retrieval-Augmented Generation (RAG) combines retrieval systems with generative AI.',
        'Vector database performance depends on index type, distance metrics, and hardware configuration.'
    ]
})

# Process in batches
batch_limit = 100
for batch_df in np.array_split(df, max(1, len(df) // batch_limit)):
    # Extract texts and metadata
    texts = batch_df['text'].tolist()
    metadatas = [
        {
            "doc_id": int(row['id']), 
            "title": row['title'],
            "text": row['text']
        } 
        for _, row in batch_df.iterrows()
    ]
    
    # Generate unique IDs
    ids = [str(uuid4()) for _ in range(len(texts))]
    
    # Create embeddings
    response = openai_client.embeddings.create(
        input=texts, 
        model="text-embedding-3-small"
    )
    embeddings = [item.embedding for item in response.data]
    
    # Prepare vector records
    vectors = [
        {"id": ids[i], "values": embeddings[i], "metadata": metadatas[i]}
        for i in range(len(texts))
    ]
    
    # Upsert vectors
    index.upsert(vectors=vectors, namespace="knowledge_base")

# Perform a semantic search
def semantic_search(query, top_k=3):
    # Create embedding for query
    query_response = openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = query_response.data[0].embedding
    
    # Query the vector database
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="knowledge_base"
    )
    
    # Format and return results
    formatted_results = []
    for match in results['matches']:
        formatted_results.append({
            "score": match['score'],
            "title": match['metadata']['title'],
            "text": match['metadata']['text']
        })
    
    return formatted_results

# Example search
search_results = semantic_search("How do vector databases work?")
for i, result in enumerate(search_results):
    print(f"{i+1}. {result['title']} (Score: {result['score']:.4f})")
    print(f"   {result['text'][:100]}...")
    print()
```

### RAG Chatbot with Pinecone

```python
# Import necessary libraries
import pandas as pd
import numpy as np
from uuid import uuid4
from pinecone import Pinecone
from openai import OpenAI

# Initialize clients
pc = Pinecone(api_key="your_pinecone_api_key")
openai_client = OpenAI(api_key="your_openai_api_key")

# Connect to index
index = pc.Index("rag-knowledge-base")

# Sample YouTube video transcript dataset
youtube_df = pd.DataFrame({
    'id': [1, 2, 3],
    'title': [
        'Intro to RAG with Pinecone',
        'Building Advanced ChatGPT Applications',
        'Vector Database Performance Tips'
    ],
    'text': [
        'In this video we explore how to implement Retrieval-Augmented Generation with Pinecone and OpenAI.',
        'Learn how to build sophisticated applications using ChatGPT and vector databases for knowledge retrieval.',
        'Discover key optimizations for vector database performance including batching, indexing strategies, and query tuning.'
    ],
    'url': [
        'https://example.com/video1',
        'https://example.com/video2',
        'https://example.com/video3'
    ],
    'published': [
        '2023-10-15',
        '2023-11-20',
        '2023-12-05'
    ]
})

# Index the YouTube transcripts
batch_limit = 100
for batch_df in np.array_split(youtube_df, max(1, len(youtube_df) // batch_limit)):
    # Extract texts and metadata
    texts = batch_df['text'].tolist()
    metadatas = [
        {
            "text_id": int(row['id']),
            "text": row['text'],
            "title": row['title'],
            "url": row['url'],
            "published": row['published']
        } 
        for _, row in batch_df.iterrows()
    ]
    
    # Generate unique IDs
    ids = [str(uuid4()) for _ in range(len(texts))]
    
    # Create embeddings
    response = openai_client.embeddings.create(
        input=texts, 
        model="text-embedding-3-small"
    )
    embeddings = [item.embedding for item in response.data]
    
    # Prepare vector records
    vectors = [
        {"id": ids[i], "values": embeddings[i], "metadata": metadatas[i]}
        for i in range(len(texts))
    ]
    
    # Upsert vectors
    index.upsert(vectors=vectors, namespace="youtube_rag_dataset")

# Define the retrieval function
def retrieve(query, top_k=3, namespace="youtube_rag_dataset", emb_model="text-embedding-3-small"):
    """Retrieve relevant documents from Pinecone."""
    # Create embedding for query
    query_response = openai_client.embeddings.create(
        input=query,
        model=emb_model
    )
    query_embedding = query_response.data[0].embedding
    
    # Query the vector database
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    
    # Extract documents and sources
    retrieved_docs = []
    sources = []
    for match in results['matches']:
        retrieved_docs.append(match['metadata']['text'])
        sources.append((match['metadata']['title'], match['metadata']['url']))
    
    return retrieved_docs, sources

# Build a context-based prompt
def prompt_with_context_builder(query, documents):
    """Build a prompt that includes retrieved context."""
    context = "\n\n".join(documents)
    return f"""Answer the question based on the context below. If you can't answer from the context, say "I don't have enough information."

Context:
{context}

Question: {query}

Answer:"""

# RAG-based question answering
def question_answering(query, chat_model="gpt-4o-mini"):
    """Answer questions using RAG."""
    # Retrieve relevant documents
    documents, sources = retrieve(query, top_k=3)
    
    # Build prompt with context
    prompt = prompt_with_context_builder(query, documents)
    
    # Generate answer with LLM
    response = openai_client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides accurate information based on the context provided."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    # Extract and format answer
    answer = response.choices[0].message.content.strip()
    
    # Add sources
    answer += "\n\nSources:"
    for title, url in sources:
        answer += f"\n- {title}: {url}"
    
    return answer

# Example RAG query
query = "How to build a RAG application with Pinecone?"
answer = question_answering(query)
print(f"Question: {query}\n")
print(answer)
```

## Conclusion

Pinecone offers a powerful vector database solution for AI applications with flexible deployment options, comprehensive APIs, and optimized performance. By following the best practices outlined in this guide, you can effectively build and scale vector search applications for semantic search, RAG, recommendation systems, and other AI use cases.

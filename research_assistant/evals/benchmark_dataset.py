"""
evals/benchmark_dataset.py
20 research Q&A pairs covering AI, tech, science, and current events.
ground_truth is the expected factual answer RAGAS uses for context_recall.
"""

BENCHMARK_QA = [
    {
        "question": "What is LangGraph and how does it differ from LangChain?",
        "ground_truth": "LangGraph is a library built on top of LangChain for building stateful, multi-actor applications with LLMs. It uses a graph structure with nodes and edges, supports cycles (unlike LangChain's DAG-only chains), and provides built-in checkpointing for persistence. LangChain is a broader framework for LLM application development.",
    },
    {
        "question": "What are the main advantages of using Groq for LLM inference?",
        "ground_truth": "Groq uses custom Language Processing Units (LPUs) designed specifically for LLM inference, offering significantly faster token generation speeds (often 10-20x faster than GPU-based systems), lower latency, and competitive pricing with a free tier available for developers.",
    },
    {
        "question": "How does Retrieval-Augmented Generation (RAG) work?",
        "ground_truth": "RAG combines a retrieval system with a generative LLM. Documents are chunked and embedded into a vector store. At query time, the query is embedded and used to retrieve the most semantically similar chunks. These chunks are injected into the LLM's context as additional information, grounding its response in retrieved facts rather than parametric knowledge alone.",
    },
    {
        "question": "What is the difference between short-term and long-term memory in AI agents?",
        "ground_truth": "Short-term memory (in-context) is the information within the current context window — previous messages, tool results, intermediate reasoning. It's lost when the session ends. Long-term memory persists across sessions, typically stored in a database or vector store, and is retrieved as needed to inform future interactions.",
    },
    {
        "question": "What are RAGAS metrics used for evaluating RAG pipelines?",
        "ground_truth": "RAGAS (Retrieval Augmented Generation Assessment) provides metrics including: faithfulness (are answer claims supported by context), answer relevancy (does the answer address the question), context precision (is retrieved context relevant), and context recall (does context cover the ground truth). It uses an LLM as a judge.",
    },
    {
        "question": "What is the CrewAI framework and what are its core concepts?",
        "ground_truth": "CrewAI is a multi-agent orchestration framework where agents are assigned roles, goals, and backstories. A Crew coordinates multiple Agents executing Tasks. It supports sequential and hierarchical process modes, built-in memory (short-term, long-term, entity), and tool integration.",
    },
    {
        "question": "How does tool calling work in LangChain with Groq models?",
        "ground_truth": "Tool calling (function calling) allows LLMs to request execution of defined functions. In LangChain, tools are defined with @tool decorator or as BaseTool subclasses. bind_tools() attaches them to the LLM. When the model's response includes a tool_calls field, the application executes the tool and returns a ToolMessage with the result, continuing the conversation.",
    },
    {
        "question": "What is prompt engineering and what are common techniques?",
        "ground_truth": "Prompt engineering is the practice of designing inputs to LLMs to produce desired outputs. Common techniques include: zero-shot prompting, few-shot examples, chain-of-thought (step-by-step reasoning), role prompting (assigning a persona), structured output instructions, and negative examples (showing what NOT to do).",
    },
    {
        "question": "What is vector similarity search and how does FAISS implement it?",
        "ground_truth": "Vector similarity search finds the most semantically similar vectors to a query vector. FAISS (Facebook AI Similarity Search) is a library for efficient similarity search over dense vectors. It supports exact (flat index) and approximate nearest neighbor (IVF, HNSW) search, balancing speed vs accuracy.",
    },
    {
        "question": "What are LangGraph checkpoints and why are they useful?",
        "ground_truth": "LangGraph checkpoints save the full graph state after each node execution, stored in a backend (SQLite, Postgres, Redis). They enable: resuming interrupted workflows, multi-turn conversation memory by thread_id, debugging by inspecting intermediate states, and time-travel (rewinding to a prior checkpoint).",
    },
    {
        "question": "What is the ReAct prompting pattern in AI agents?",
        "ground_truth": "ReAct (Reason + Act) is a prompting pattern where the agent alternates between reasoning (thinking about what to do) and acting (calling tools). The loop is: Thought → Action → Observation → Thought... until the agent reaches a final answer. It improves reliability by making the reasoning process explicit.",
    },
    {
        "question": "What is the difference between GPT-4o and Claude 3.5 Sonnet?",
        "ground_truth": "GPT-4o is OpenAI's multimodal model supporting text, image, and audio natively. Claude 3.5 Sonnet is Anthropic's model known for strong coding, reasoning, and instruction following. Both are frontier models with large context windows. Key differences include their training data, safety approaches, and API pricing.",
    },
    {
        "question": "How does the Transformer attention mechanism work?",
        "ground_truth": "The Transformer attention mechanism computes compatibility between a query vector and a set of key vectors (via dot product), normalises with softmax to get attention weights, then computes a weighted sum of value vectors. Multi-head attention runs this process in parallel across multiple heads, each learning different relationships.",
    },
    {
        "question": "What is DuckDuckGo and how does it protect privacy compared to Google?",
        "ground_truth": "DuckDuckGo is a search engine that does not track users, store search history, or create user profiles. It does not use personalised search results. Unlike Google, it does not use cookies across sites or share user data with advertisers. It provides anonymous search results to all users.",
    },
    {
        "question": "What are the main components of a multi-agent system?",
        "ground_truth": "A multi-agent system consists of: individual agents (with specific roles, memory, tools), an orchestrator or supervisor (coordinates agent actions), shared state (passed between agents), communication protocols (how agents exchange information), and tools (external capabilities agents can invoke). Frameworks include LangGraph, CrewAI, and AutoGen.",
    },
    {
        "question": "What is fine-tuning an LLM and when should you use it vs RAG?",
        "ground_truth": "Fine-tuning updates a model's weights by training on domain-specific data, teaching the model new behaviors or knowledge. RAG retrieves external knowledge at inference time without changing model weights. Use fine-tuning for style, tone, or capability changes; use RAG for dynamic, large, or frequently updated knowledge bases. They can be combined.",
    },
    {
        "question": "What is agent communication protocols in multi-agent systems?",
        "ground_truth": "Agent communication protocols define how agents exchange information. In LangGraph, agents communicate via shared TypedDict state with typed reducers. In CrewAI, agents communicate through task outputs passed between crew members. Common patterns include blackboard (shared state), message passing, and direct agent-to-agent delegation.",
    },
    {
        "question": "What is Llama 3 and who developed it?",
        "ground_truth": "Llama 3 is an open-source large language model developed by Meta AI. It comes in 8B and 70B parameter variants, with a 405B instruct model. It is available for commercial use under Meta's license. Llama 3.1 added support for 128K context length. It is widely used via Groq, Ollama, and HuggingFace.",
    },
    {
        "question": "How do embeddings represent text semantically?",
        "ground_truth": "Text embeddings are dense vector representations where semantically similar texts are mapped to nearby points in high-dimensional space. They are produced by encoder models (e.g. sentence-transformers) trained on large corpora. Vector distance (cosine similarity, dot product) measures semantic similarity, enabling semantic search and clustering.",
    },
    {
        "question": "What is LLM evaluation and what makes it challenging?",
        "ground_truth": "LLM evaluation measures model quality on tasks like QA, summarization, and reasoning. Challenges include: no single ground truth (multiple valid answers exist), reference-free evaluation requires an LLM judge (costly, prone to bias), benchmark contamination (models trained on test data), and capturing subtle qualities like helpfulness or safety.",
    },
]

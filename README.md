# The core idea of a Multi-Document Agent

The core idea of a Multi-Document Agent is to simulate a knowledgeable assistant that can draw upon information from multiple separate documents to provide informed, accurate answers to user queries. Unlike a traditional, single-document agent that can only access and understand information from one source, a multi-document agent has the capability to consider a broader range of information sources, similar to how a human expert would consult various references to answer complex questions.

## How a multi-document agent works

Here's an outline of how a multi-document agent works:

1. **Document Agents**: Each document (or set of related documents) is paired with a document agent. This agent is responsible for understanding the content within its assigned document(s). It has capabilities such as semantic search, to find relevant snippets within the document, and summarization, to distill the document's content into a concise form.

2. **Top-Level Agent**: A top-level agent oversees the document agents. When a user asks a question, the top-level agent determines which document agents might have relevant information and directs the query appropriately.

3. **Tool Retrieval**: The top-level agent uses a retriever mechanism to identify which tools (i.e., query engines of the individual document agents) are most relevant for the given query.

4. **Reranking**: Retrieved documents or relevant snippets are reranked (possibly by an external system like Cohere) to refine the set of candidate responses, ensuring that the most relevant information is considered.

5. **Query Planning Tool**: This tool is dynamically created based on the retrieved tools to plan out an effective strategy for leveraging the selected documents to answer the user's query.

6. **Answering Queries**: To answer a query, the top-level agent orchestrates the use of the retrieved and reranked tools, conducting a "chain of thought" process over the set of relevant documents to formulate a comprehensive response.

7. **Integration**: The responses from the document agents are integrated into a single coherent answer for the user.

This setup is particularly powerful for complex queries that require cross-referencing information, understanding nuanced details, or comparing statements from multiple sources. The multi-document agent is designed to mimic a human expert's approach to research and analysis, using a rich set of data and tools for a more robust and informed response.

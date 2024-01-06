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

## Imports

```python
# System imports
from pathlib import Path
import os
import pickle
# Third-party imports
from llama_hub.file.unstructured.base import UnstructuredReader
from llama_index.llms import OpenAI
from llama_index import ServiceContext
from llama_index import Document
from llama_index import VectorStoreIndex, SummaryIndex
from llama_index.agent import OpenAIAgent
from llama_index import load_index_from_storage, StorageContext
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.node_parser import SentenceSplitter
from llama_index import VectorStoreIndex
from llama_index.objects import (
ObjectIndex,
SimpleToolNodeMapping,
ObjectRetriever,
)
from llama_index.retrievers import BaseRetriever
from llama_index.postprocessor import CohereRerank
from llama_index.tools import QueryPlanTool
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.llms import OpenAI
from llama_index.agent import FnRetrieverOpenAIAgent, ReActAgent
```

## Setup and Download Data

In this section, we’ll load in the LlamaIndex documentation.

```python
domain = "docs.llamaindex.ai"
docs_url = "https://docs.llamaindex.ai/en/latest/"
!wget -e robots=off --recursive --no-clobber --page-requisites --html-extension --convert-links --restrict-file-names=windows --domains {domain} --no-parent {docs_url}

reader = UnstructuredReader()
all_files_gen = Path("./docs.llamaindex.ai/").rglob("*")
all_files = [f.resolve() for f in all_files_gen]
all_html_files = [f for f in all_files if f.suffix.lower() == ".html"]
len(all_html_files)

docs = []
for idx, f in enumerate(all_html_files):
    print(f"Idx {idx}/{len(all_html_files)}")
    loaded_docs = reader.load_data(file=f, split_documents=True)
    loaded_doc = Document(
        text="\n\n".join([d.get_content() for d in loaded_docs]),
        metadata={"path": str(f)},
    )
    print(loaded_doc.metadata["path"])
    docs.append(loaded_doc)
```

## Define LLM + Service Context + Callback Manager

llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)

## Building Multi-Document Agents

In this section we show you how to construct the multi-document agent. We first build a document agent for each document, and then define the top-level parent agent with an object index.

### Build Document Agent for each Document

In this section we define “document agents” for each document.

We define both a vector index (for semantic search) and summary index (for summarization) for each document. The two query engines are then converted into tools that are passed to an OpenAI function calling agent.

This document agent can dynamically choose to perform semantic search or summarization within a given document.

We create a separate document agent for each document.

```python
async def build_agent_per_doc(nodes, file_base):
    print(file_base)

    vi_out_path = f"./data/llamaindex_docs/{file_base}"
    summary_out_path = f"./data/llamaindex_docs/{file_base}_summary.pkl"
    if not os.path.exists(vi_out_path):
        Path("./data/llamaindex_docs/").mkdir(parents=True, exist_ok=True)
        # build vector index
        vector_index = VectorStoreIndex(nodes, service_context=service_context)
        vector_index.storage_context.persist(persist_dir=vi_out_path)
    else:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=vi_out_path),
            service_context=service_context,
        )

    # build summary index
    summary_index = SummaryIndex(nodes, service_context=service_context)

    # define query engines
    vector_query_engine = vector_index.as_query_engine()
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize"
    )

    # extract a summary
    if not os.path.exists(summary_out_path):
        Path(summary_out_path).parent.mkdir(parents=True, exist_ok=True)
        summary = str(
            await summary_query_engine.aquery(
                "Extract a concise 1-2 line summary of this document"
            )
        )
        pickle.dump(summary, open(summary_out_path, "wb"))
    else:
        summary = pickle.load(open(summary_out_path, "rb"))

    # define tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name=f"vector_tool_{file_base}",
                description=f"Useful for questions related to specific facts",
            ),
        ),
        QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name=f"summary_tool_{file_base}",
                description=f"Useful for summarization questions",
            ),
        ),
    ]

    # build agent
    function_llm = OpenAI(model="gpt-4")
    agent = OpenAIAgent.from_tools(
        query_engine_tools,
        llm=function_llm,
        verbose=True,
        system_prompt=f"""\

You are a specialized agent designed to answer queries about the `{file_base}.html` part of the LlamaIndex docs.
You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
""",
)

    return agent, summary

async def build_agents(docs):
    node_parser = SentenceSplitter()

    # Build agents dictionary
    agents_dict = {}
    extra_info_dict = {}

    for idx, doc in enumerate(tqdm(docs)):
        nodes = node_parser.get_nodes_from_documents([doc])

        # ID will be base + parent
        file_path = Path(doc.metadata["path"])
        file_base = str(file_path.parent.stem) + "_" + str(file_path.stem)
        agent, summary = await build_agent_per_doc(nodes, file_base)

        agents_dict[file_base] = agent
        extra_info_dict[file_base] = {"summary": summary, "nodes": nodes}

    return agents_dict, extra_info_dict

agents_dict, extra_info_dict = await build_agents(docs)
```

## Build Retriever-Enabled OpenAI Agent

We build a top-level agent that can orchestrate across the different document agents to answer any user query.

This RetrieverOpenAIAgent performs tool retrieval before tool use (unlike a default agent that tries to put all tools in the prompt).

Improvements from V0: We make the following improvements compared to the “base” version in V0.

Adding in reranking: we use Cohere reranker to better filter the candidate set of documents.

Adding in a query planning tool: we add an explicit query planning tool that’s dynamically created based on the set of retrieved tools.

```python
# define tool for each document agent

all_tools = []
for file_base, agent in agents_dict.items():
    summary = extra_info_dict[file_base]["summary"]
    doc_tool = QueryEngineTool(
    query_engine=agent,
    metadata=ToolMetadata(
        name=f"tool_{file_base}",
        description=summary,
        ),
    )
    all_tools.append(doc_tool)

print(all_tools[0].metadata)
"""
ToolMetadata(description='LlamaIndex is a data framework that allows LLM applications to ingest, structure, and access private or domain-specific data by providing tools such as data connectors, data indexes, engines, data agents, and application integrations. It is designed for beginners, advanced users, and everyone in between, and offers both high-level and lower-level APIs for customization. LlamaIndex can be installed using pip and has detailed documentation and tutorials available. It is available on GitHub and PyPi, and there is also a Typescript package available. The LlamaIndex community can be joined on Twitter and Discord.', name='tool_latest_index', fn_schema=<class 'llama_index.tools.types.DefaultToolFnSchema'>)
"""

# define an "object" index and retriever over these tools


llm = OpenAI(model_name="gpt-4-0613")

tool_mapping = SimpleToolNodeMapping.from_objects(all_tools)
obj_index = ObjectIndex.from_objects(
    all_tools,
    tool_mapping,
    VectorStoreIndex,
)
vector_node_retriever = obj_index.as_node_retriever(similarity_top_k=10)

# define a custom retriever with reranking

class CustomRetriever(BaseRetriever):
    def __init__(self, vector_retriever, postprocessor=None):
        self._vector_retriever = vector_retriever
        self._postprocessor = postprocessor or CohereRerank(top_n=5)
        super().__init__()

    def _retrieve(self, query_bundle):
        retrieved_nodes = self._vector_retriever.retrieve(query_bundle)
        filtered_nodes = self._postprocessor.postprocess_nodes(
            retrieved_nodes, query_bundle=query_bundle
        )

        return filtered_nodes

# define a custom object retriever that adds in a query planning tool

class CustomObjectRetriever(ObjectRetriever):
    def __init__(self, retriever, object_node_mapping, all_tools, llm=None):
        self._retriever = retriever
        self._object_node_mapping = object_node_mapping
        self._llm = llm or OpenAI("gpt-4-0613")

    def retrieve(self, query_bundle):
        nodes = self._retriever.retrieve(query_bundle)
        tools = [self._object_node_mapping.from_node(n.node) for n in nodes]

        sub_question_sc = ServiceContext.from_defaults(llm=self._llm)
        sub_question_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=tools, service_context=sub_question_sc
        )
        sub_question_description = f"""\

Useful for any queries that involve comparing multiple documents. ALWAYS use this tool for comparison queries - make sure to call this \
tool with the original query. Do NOT use the other tools for any queries involving multiple documents.
"""
        sub_question_tool = QueryEngineTool(
            query_engine=sub_question_engine,
            metadata=ToolMetadata(
            name="compare_tool", description=sub_question_description
            ),
        )

        return tools + [sub_question_tool]

custom_node_retriever = CustomRetriever(vector_node_retriever)

# wrap it with ObjectRetriever to return objects

custom_obj_retriever = CustomObjectRetriever(
    custom_node_retriever, tool_mapping, all_tools, llm=llm,
)
# tmps = custom_obj_retriever.retrieve("hello")
# print(len(tmps))
# 6

top_agent = FnRetrieverOpenAIAgent.from_retriever(
    custom_obj_retriever,
    system_prompt=""" \
    You are an agent designed to answer queries about the documentation.
    Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

    """,
    llm=llm,
    verbose=True,
)
```

"""
# The core idea of a Multi-Document Agent

The core idea of a Multi-Document Agent is to simulate a knowledgeable assistant that can draw upon information from
multiple separate documents to provide informed, accurate answers to user queries. Unlike a traditional, single-document
agent that can only access and understand information from one source, a multi-document agent has the capability to
consider a broader range of information sources, similar to how a human expert would consult various references to
answer complex questions.

## How a multi-document agent works

Here's an outline of how a multi-document agent works:

1. **Document Agents**: Each document (or set of related documents) is paired with a document agent. This agent is
responsible for understanding the content within its assigned document(s). It has capabilities such as semantic search,
to find relevant snippets within the document, and summarization, to distill the document's content into a concise form.

2. **Top-Level Agent**: A top-level agent oversees the document agents. When a user asks a question, the top-level
agent determines which document agents might have relevant information and directs the query appropriately.

3. **Tool Retrieval**: The top-level agent uses a retriever mechanism to identify which tools (i.e., query engines of
the individual document agents) are most relevant for the given query.

4. **Reranking**: Retrieved documents or relevant snippets are reranked (possibly by an external system like Cohere)
to refine the set of candidate responses, ensuring that the most relevant information is considered.

5. **Query Planning Tool**: This tool is dynamically created based on the retrieved tools to plan out an effective
strategy for leveraging the selected documents to answer the user's query.

6. **Answering Queries**: To answer a query, the top-level agent orchestrates the use of the retrieved and reranked
tools, conducting a "chain of thought" process over the set of relevant documents to formulate a comprehensive response.

7. **Integration**: The responses from the document agents are integrated into a single coherent answer for the user.

This setup is particularly powerful for complex queries that require cross-referencing information, understanding
nuanced details, or comparing statements from multiple sources. The multi-document agent is designed to mimic a human
expert's approach to research and analysis, using a rich set of data and tools for a more robust and informed response.
"""
# System imports
from contextlib import asynccontextmanager
import json
from pathlib import Path
import os
import pickle
from urllib.parse import urlparse
from typing import Sequence

# Third-party imports
from dotenv import load_dotenv
from llama_index import download_loader
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
from tqdm import tqdm
from llama_index.schema import BaseNode
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

#

load_dotenv()

#


class CustomRetriever(BaseRetriever):
    def __init__(self, vector_retriever, postprocessor=None) -> None:
        self._vector_retriever = vector_retriever
        self._postprocessor = postprocessor or CohereRerank(
            top_n=5, api_key=os.getenv("COHERE_API_KEY")
        )
        super().__init__()

    def _retrieve(self, query_bundle):
        retrieved_nodes = self._vector_retriever.retrieve(query_bundle)
        filtered_nodes = self._postprocessor.postprocess_nodes(
            retrieved_nodes, query_bundle=query_bundle
        )

        return filtered_nodes


class CustomObjectRetriever(ObjectRetriever):
    def __init__(self, retriever, object_node_mapping, all_tools, llm=None) -> None:
        self._retriever = retriever
        self._object_node_mapping = object_node_mapping
        self._llm = llm or OpenAI(
            "gpt-4-0613",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

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


def download_website(
    url: str,
    corpus_name: str,
    data_base_path: str = "./data",
) -> str:
    domain = urlparse(url).netloc
    corpus_path = os.path.join(data_base_path, corpus_name)
    domain_path = os.path.join(corpus_path, domain)
    if not os.path.exists(domain_path):
        os.system(
            f"wget -e robots=off --recursive --no-clobber --page-requisites --html-extension --convert-links --restrict-file-names=windows --domains {domain} --no-parent -P {corpus_path} {url}"
        )
    return domain_path


def load_documents_from_directory(
    directory_path: str,
    suffix_filter: str | None,
    limit: int | None = None,
) -> list[Document]:
    UnstructuredReader = download_loader("UnstructuredReader")
    reader = UnstructuredReader()
    all_files_gen = Path(directory_path).rglob("*")
    all_files = [f.resolve() for f in all_files_gen]
    if suffix_filter is not None:
        all_files = [f for f in all_files if f.suffix.lower() == suffix_filter]
    if limit is not None:
        all_files = all_files[:limit]

    docs = []
    for idx, f in tqdm(enumerate(all_files), desc="Loading documents"):
        loaded_docs = reader.load_data(file=f, split_documents=True)
        loaded_doc = Document(
            text="\n\n".join([d.get_content() for d in loaded_docs]),
            metadata={"path": str(f)},
        )
        docs.append(loaded_doc)

    return docs


async def create_document_agent(
    nodes: Sequence[BaseNode],
    file_base: str,
    data_base_path: str,
    corpus_name: str,
    service_context: ServiceContext,
) -> tuple[OpenAIAgent, str]:
    base_path = os.path.join(data_base_path, corpus_name, "agents")
    vi_out_path = os.path.join(base_path, file_base)
    summary_out_path = os.path.join(base_path, f"{file_base}_summary.pkl")

    # Create or load the vector index
    if not os.path.exists(vi_out_path):
        Path(base_path).mkdir(parents=True, exist_ok=True)
        vector_index = VectorStoreIndex(nodes, service_context=service_context)
        vector_index.storage_context.persist(persist_dir=vi_out_path)
    else:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=vi_out_path),
            service_context=service_context,
        )

    # Create or load the summary index
    summary_index = SummaryIndex(nodes, service_context=service_context)

    # Create query engines for the vector and summary indices
    vector_query_engine = vector_index.as_query_engine()
    summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize")

    # Create or load the summary
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

    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name=f"vector_tool_{file_base}",
                description="Useful for questions related to specific facts",
            ),
        ),
        QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name=f"summary_tool_{file_base}",
                description="Useful for summarization questions",
            ),
        ),
    ]

    function_llm = OpenAI(
        model="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    agent = OpenAIAgent.from_tools(
        query_engine_tools,
        llm=function_llm,
        verbose=True,
        system_prompt=f"""\

You are a specialized agent designed to answer queries about the `{file_base}` part of the {corpus_name} docs.
You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
""",
    )
    return agent, summary


async def create_document_agents(
    docs: list[Document],
    data_base_path: str,
    corpus_name: str,
    service_context: ServiceContext,
) -> tuple[dict, dict]:
    node_parser = SentenceSplitter()

    document_agents = {}
    extra_info = {}

    for doc in docs:
        nodes = node_parser.get_nodes_from_documents([doc])

        # ID will be base + parent
        file_path = Path(doc.metadata["path"])
        file_base = str(file_path.parent.stem) + "_" + str(file_path.stem)
        file_base = file_base.replace(".", "_")

        agent, summary = await create_document_agent(
            nodes=nodes,
            file_base=file_base,
            data_base_path=data_base_path,
            corpus_name=corpus_name,
            service_context=service_context,
        )

        document_agents[file_base] = agent
        extra_info[file_base] = {"summary": summary, "nodes": nodes}

    return document_agents, extra_info


async def create_multidoc_agent(
    url: str,
    corpus_name: str,
) -> None:
    data_base_path = os.getenv("DATA_BASE_PATH", "./data")

    llm = OpenAI(
        model_name="gpt-4-0613",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    service_context = ServiceContext.from_defaults(llm=llm)

    #

    corpus_path = download_website(
        url=url,
        corpus_name=corpus_name,
        data_base_path=data_base_path,
    )

    docs = load_documents_from_directory(
        directory_path=corpus_path,
        suffix_filter=".html",
        # limit=10,
    )

    document_agents, extra_info = await create_document_agents(
        docs=docs,
        data_base_path=data_base_path,
        corpus_name=corpus_name,
        service_context=service_context,
    )

    all_tools = []
    for file_base, agent in document_agents.items():
        summary = extra_info[file_base]["summary"]
        doc_tool = QueryEngineTool(
            query_engine=agent,
            metadata=ToolMetadata(
                name=f"tool_{file_base}",
                description=summary,
            ),
        )
        all_tools.append(doc_tool)

    tool_mapping = SimpleToolNodeMapping.from_objects(all_tools)
    obj_index = ObjectIndex.from_objects(
        all_tools,
        tool_mapping,
        VectorStoreIndex,
    )

    vector_node_retriever = obj_index.as_node_retriever(similarity_top_k=10)

    custom_node_retriever = CustomRetriever(vector_node_retriever)

    custom_obj_retriever = CustomObjectRetriever(
        custom_node_retriever,
        tool_mapping,
        all_tools,
        llm=llm,
    )

    top_agent = FnRetrieverOpenAIAgent.from_retriever(
        custom_obj_retriever,
        system_prompt=""" \
        You are an agent designed to answer queries about the documentation.
        Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

        """,
        llm=llm,
        verbose=True,
    )
    return top_agent


#


# Store agents in a dictionary
agents = {}


# Load agents from the JSON file when the server starts
async def load_agents():
    try:
        data_base_path = os.getenv("DATA_BASE_PATH", "./data")
        agent_registry_file = os.path.join(data_base_path, "agents.json")
        with open(agent_registry_file, "r") as f:
            agents_registry = json.load(f)
            for corpus_name, url in agents_registry.items():
                agents[corpus_name] = await create_multidoc_agent(url, corpus_name)
    except FileNotFoundError:
        pass  # It's okay if the file doesn't exist


def save_agent(
    corpus_name: str,
    url: str,
):
    data_base_path = os.getenv("DATA_BASE_PATH", "./data")
    agent_registry = os.path.join(data_base_path, "agents.json")
    try:
        with open(agent_registry, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    data.update({corpus_name: url})
    with open(agent_registry, "w") as f:
        json.dump(data, f)


#


@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_agents()  # Call the function when the server starts
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with the appropriate origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#


@app.post("/agent/")
async def create_agent(
    url: str,
    corpus_name: str,
) -> Response:
    agent = await create_multidoc_agent(url, corpus_name)
    agents[corpus_name] = agent
    save_agent(corpus_name=corpus_name, url=url)
    response = Response(status_code=201)
    response.headers["Location"] = f"/agent/{corpus_name}"
    return response


@app.post("/agent/{corpus_name}/query")
async def query_agent(
    corpus_name: str,
    query: str,
) -> str:
    agent = agents.get(corpus_name)
    if agent is None:
        return Response(status_code=404)
    response = await agent.aquery(query)
    return response.response


#

# url = "https://docs.llamaindex.ai/en/latest/"
# corpus_name = "llamaindex_docs"

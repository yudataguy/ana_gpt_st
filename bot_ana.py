"""
Streamlit ANAGPT Chat Bot, v1 (older)
"""


import os

import streamlit as st
from langchain import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from llama_index import (
    GPTListIndex,
    GPTSimpleVectorIndex,
    LLMPredictor,
    ServiceContext,
)
from llama_index.indices.composability import ComposableGraph
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from llama_index.langchain_helpers.agents import (
    GraphToolConfig,
    IndexToolConfig,
    LlamaToolkit,
    create_llama_chat_agent,
)
from streamlit_chat import message

# set years
years = [2022, 2021]

# Load indices from disk
index_set = {}
for year in years:
    cur_index = GPTSimpleVectorIndex.load_from_disk(f"./data/json/index_{year}.json")
    index_set[year] = cur_index

# describe each index to help traversal of composed graph
index_summaries = [f"ANA Financial Report for {year} fiscal year" for year in years]

llm_predictor = LLMPredictor(
    llm=OpenAI(
        temperature=0,
        max_tokens=512,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )
)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# define a list index over the vector indices
# allows us to synthesize information across each index
graph = ComposableGraph.from_indices(
    GPTListIndex,
    [index_set[y] for y in years],
    index_summaries=index_summaries,
    service_context=service_context,
)

# [optional] save to disk
# graph.save_to_disk("./data/json/aggregated.json")

# [optional] load from disk, so you don't need to build graph from scratch
graph = ComposableGraph.load_from_disk(
    "./data/json/aggregated.json",
    service_context=service_context,
)

# define a decompose transform
decompose_transform = DecomposeQueryTransform(llm_predictor, verbose=True)

print("Transform data ...")
# define query configs for graph
query_configs = [
    {
        "index_struct_type": "simple_dict",
        "query_mode": "default",
        "query_kwargs": {
            "similarity_top_k": 1,
            # "include_summary": True
        },
        "query_transform": decompose_transform,
    },
    {
        "index_struct_type": "list",
        "query_mode": "default",
        "query_kwargs": {"response_mode": "tree_summarize", "verbose": True},
    },
]
# graph config
graph_config = GraphToolConfig(
    graph=graph,
    name="Graph Index",
    description="useful for when you want to answer queries that require analyzing multiple financial reports for ANA.",
    query_configs=query_configs,
    tool_kwargs={"return_direct": True},
)

index_configs = []
for y in range(2021, 2023):
    tool_config = IndexToolConfig(
        index=index_set[y],
        name=f"Vector Index {y}",
        description=f"useful for when you want to answer queries about the {y} financial report for ANA",
        index_query_kwargs={"similarity_top_k": 1},
        tool_kwargs={"return_direct": True},
    )
    index_configs.append(tool_config)

toolkit = LlamaToolkit(index_configs=index_configs, graph_configs=[graph_config])

print("Building chatbot...")

memory = ConversationBufferMemory(memory_key="chat_history")
llm = OpenAI(temperature=0)
agent_chain = create_llama_chat_agent(toolkit, llm, memory=memory, verbose=True)


def generate_response(prompt):
    """Generate a response to a prompt."""
    return agent_chain.run(input=prompt)


st.title("✈️ ANAGPT")
st.subheader("2021年と2022年の財務年度のチャットボット")

st.markdown("英語と日本語の両方で質問することができますが、🤖の知識は2021年と2022年の財務年度に限られています。")


if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_input() -> str:
    """Get user input from the text input box."""
    input_text = st.text_input("何を聞きたいですか？", key="input")
    return input_text


user_input = get_input()


if user_input:
    output = generate_response(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)


if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

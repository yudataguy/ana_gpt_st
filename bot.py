import streamlit as st
from langchain.chat_models import ChatOpenAI
from llama_index import (
    GPTSimpleVectorIndex,
    LLMPredictor,
    PromptHelper,
    QuestionAnswerPrompt,
    RefinePrompt,
    ServiceContext,
)
from llama_index.langchain_helpers.text_splitter import SentenceSplitter
from llama_index.logger import LlamaLogger
from llama_index.node_parser import SimpleNodeParser
from streamlit_chat import message

max_input_size = 4096

num_output = 2000

max_chunk_overlap = 5

QA_PROMPT_TMPL = (
    "{context_str}" "\n####\n" "Answer the question in Japanese. {query_str}\n"
)

QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

REFINE_PROMPT_TMPL = (
    "The original question is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the above answer"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer to the question in Japanese,but don't mention words like"
    "'context','given information' etc."
    "If the context isn't useful, output the original answer again."
)
REFINE_PROMPT = RefinePrompt(REFINE_PROMPT_TMPL)

paragraph_separator = "###"

secondary_chunking_regex = (
    "[^｛｝（）［］【】、゠＝…‥。「」『』〝〟⟨⟩〜：！？♪￥]+[｛｝（）［］【】、゠＝…‥。「」『』〝〟⟨⟩〜：！？♪￥]?"
)

my_index = GPTSimpleVectorIndex.load_from_disk("index.json")

llama_logger = LlamaLogger()
chunk_size_limit = 600
temperature = 0
similarity_top_k = 1

sentence_splitter = SentenceSplitter(
    chunk_size=chunk_size_limit,
    chunk_overlap=max_chunk_overlap,
    paragraph_separator=paragraph_separator,
    secondary_chunking_regex=secondary_chunking_regex,
)
node_parser = SimpleNodeParser(text_splitter=sentence_splitter)
prompt_helper = PromptHelper(
    max_input_size=max_input_size,
    num_output=num_output,
    max_chunk_overlap=max_chunk_overlap,
    chunk_size_limit=chunk_size_limit,
)
llm_predictor = LLMPredictor(
    llm=ChatOpenAI(temperature=temperature, model_name="gpt-3.5-turbo")
)
service_context = ServiceContext.from_defaults(
    node_parser=node_parser,
    llm_predictor=llm_predictor,
    prompt_helper=prompt_helper,
    llama_logger=llama_logger,
    chunk_size_limit=chunk_size_limit,
)


def generate_response(prompt):
    response = my_index.query(
        prompt,
        text_qa_template=QA_PROMPT,
        refine_template=REFINE_PROMPT,
        service_context=service_context,
        similarity_top_k=similarity_top_k,
    )
    return response


st.title("✈️ ANAGPT")
st.subheader("2022年の財務年度のチャットボット")

st.markdown("英語と日本語の両方で質問することができますが、🤖の知識は2022年の財務年度に限られています。")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

if "input" not in st.session_state:
    st.session_state["input"] = ""


# def get_input() -> str:
#     """Get user input from the text input box."""
#     input_text = st.text_input("何を聞きたいですか？", key="input")
#     return input_text


def submit():
    st.session_state.input = st.session_state.widget
    if st.session_state.input:
        output = generate_response(st.session_state.input)
        st.session_state.past.append(st.session_state.input)
        st.session_state.generated.append(output.response)
        st.session_state.widget = ""


st.text_input("何を聞きたいですか？", key="widget", on_change=submit)


if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

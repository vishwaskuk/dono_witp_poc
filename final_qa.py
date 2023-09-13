import os


os.system('wget -q https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.4.1/auto_gptq-0.4.1+cu118-cp310-cp310-linux_x86_64.whl')
os.system('pip install -qqq auto_gptq-0.4.1+cu118-cp310-cp310-linux_x86_64.whl --progress-bar off')
os.system('sudo apt-get install poppler-utils')

import uuid
#import replicate
import requests
import streamlit as st
from streamlit.logger import get_logger
import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from pdf2image import convert_from_path
from transformers import AutoTokenizer, TextStreamer, pipeline
from langchain.memory import ConversationBufferMemory
from gtts import gTTS
from io import BytesIO
from langchain.chains import ConversationalRetrievalChain
from streamlit_modal import Modal
import streamlit.components.v1 as components
#from sentence_transformers import SentenceTransformer
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.vectorstores.utils import filter_complex_metadata
import fitz
from PIL import Image

user_session_id = uuid.uuid4()

logger = get_logger(__name__)
st.set_page_config(page_title="Document QA by Dono", page_icon="🤖",  )
st.session_state.disabled = False
st.title("Document QA by Dono")
st.markdown(f"""<style>
            .stApp {{background-image: url("https://media.istockphoto.com/id/450481545/photo/glowing-lightbulb-against-black-background.webp?b=1&s=170667a&w=0&k=20&c=fJ91chWN1UkoKTNUvwgiQwpM80DlRpVC-WlJH_78OvE=");
                     background-attachment: fixed;
                     background-size: cover}}
         </style>
         """, unsafe_allow_html=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

loader = PyPDFDirectoryLoader("/pdfs/")
docs = loader.load()
#len(docs)




@st.cache_resource
def load_model():
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large",model_kwargs={"device":DEVICE})
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={"device":DEVICE})


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
    texts = text_splitter.split_documents(docs)

    db = Chroma.from_documents(texts, embeddings, persist_directory="db")

    model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
    model_basename = "model"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    model = AutoGPTQForCausalLM.from_quantized(
        model_name_or_path,
        revision="gptq-4bit-128g-actorder_True",
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        inject_fused_attention=False,
        device=DEVICE,
        quantize_config=None,
    )

    DEFAULT_SYSTEM_PROMPT = """
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. Always provide the citation for the answer from the text. Try to include any section or subsection present in the text responsible for the answer. Provide reference. Provide page number, section, sub section etc from which answer is taken.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. 
    """.strip()


    def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        return f"""[INST] <<SYS>>{system_prompt}<</SYS>>{prompt} [/INST]""".strip()

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    text_pipeline = pipeline("text-generation",model=model,tokenizer=tokenizer,max_new_tokens=1024,
        temperature=0.2,top_p=0.95,repetition_penalty=1.15,streamer=streamer,)

    llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.2})

    SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."

    template = generate_prompt("""{context}  Question: {question} """,system_prompt=SYSTEM_PROMPT,) #Enter memory here!

    prompt = PromptTemplate(template=template, input_variables=["context",  "question"]) #Add history here

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt,
                           "verbose": False,
                           #"memory": ConversationBufferMemory(
                              #memory_key="history",
                              #input_key="question",
                              #return_messages=True)
                              },)
    return qa_chain


uploaded_file = len(docs)
flag = 0
if uploaded_file is not None:
    flag = 1 

model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
model_basename = "model"

st.session_state["llm_model"] = model_name_or_path


if "messages" not in st.session_state:
    st.session_state.messages = []



for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def on_select():
    st.session_state.disabled = True


def get_message_history():
    for message in st.session_state.messages:
        role, content = message["role"], message["content"]
        yield f"{role.title()}: {content}"


if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        message_history = "\n".join(list(get_message_history())[-3:])
        logger.info(f"{user_session_id} Message History: {message_history}")
        qa_chain = load_model()
        # question = st.text_input("Ask your question", placeholder="Try to include context in your question",
        # disabled=not uploaded_file,)
        result = qa_chain(prompt)
        sound_file = BytesIO()
        tts = gTTS(result['result'], lang='en')
        tts.write_to_fp(sound_file)
        output = [result['result']]

    for item in output:
        full_response += item
        message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)    
    #st.write(repr(result['source_documents'][0].metadata['page']))
    #st.write(repr(result['source_documents'][0]))


    ### READ IN PDF
    page_number = int(result['source_documents'][0].metadata['page'])
    doc = fitz.open(str(result['source_documents'][0].metadata['source']))

    text = str(result['source_documents'][0].page_content)
    if text != '':
        for page in doc:
            ### SEARCH
            text_instances = page.search_for(text)

            ### HIGHLIGHT
            for inst in text_instances:
                highlight = page.add_highlight_annot(inst)
                highlight.update()

    ### OUTPUT
    doc.save("/pdf2image/output.pdf", garbage=4, deflate=True, clean=True)

    # pdf_to_open = repr(result['source_documents'][0].metadata['source'])

    def pdf_page_to_image(pdf_file, page_number, output_image):
        # Open the PDF file
        pdf_document = fitz.open(pdf_file)

        # Get the specific page
        page = pdf_document[page_number]

        # Define the image DPI (dots per inch)
        dpi = 300  # You can adjust this as needed

        # Convert the page to an image
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 100, dpi / 100))

        # Save the image as a PNG file
        pix.save(output_image, "png")

        # Close the PDF file
        pdf_document.close()


    pdf_page_to_image('/pdf2image/output.pdf', page_number, '/pdf2image/output.png')

    image = Image.open('/pdf2image/output.png')
    st.image(image)
    st.audio(sound_file)

    # if 'clickedR' not in st.session_state:
    #     st.session_state.clickedR = False

    # def click_buttonR():
    #     st.session_state.clickedR = True
    #     if st.session_state.clickedR:
    #         message_placeholder.markdown(full_response+repr(result['source_documents'][0]))

    # ref = st.button('References', on_click = click_buttonR)

    
    # if 'clickedA' not in st.session_state:
    #     st.session_state.clickedA = False

    # def click_buttonA():
    #     st.session_state.clickedA = True
    #     if st.session_state.clickedA:
    #         sound_file = BytesIO()
    #         tts = gTTS(result['result'], lang='en')
    #         tts.write_to_fp(sound_file)
    #         st.audio(sound_file)  


    # ref = st.button(':speaker:', on_click = click_buttonA)

  



    #st.session_state.clickedR = False

    # #if ref:
    # message_placeholder.markdown(full_response+repr(result['source_documents'][0]))
    # #if sound:
    # sound_file = BytesIO()
    # tts = gTTS(result['result'], lang='en')
    # tts.write_to_fp(sound_file)
    # html_string = """
    # <audio controls autoplay>
    #   <source src="/content/sound_file" type="audio/wav">
    # </audio>
    # """
    # message_placeholder.markdown(html_string, unsafe_allow_html=True)  # will display a st.audio with the sound you specified in the "src" of the html_string and autoplay it
    # #time.sleep(5)  # wait for 2 seconds to finish the playing of the audio
    response_sentiment = st.radio(
        "How was the Assistant's response?",
        ["😁", "😕", "😢"],
        key="response_sentiment",
        disabled=st.session_state.disabled,
        horizontal=True,
        index=1,
        help="This helps us improve the model.",
        # hide the radio button on click
        on_change=on_select(),
    )
    logger.info(f"{user_session_id} | {full_response} | {response_sentiment}")

    # # Logging to FastAPI Endpoint
    # headers = {"Authorization": f"Bearer {secret_token}"}
    # log_data = {"log": f"{user_session_id} | {full_response} | {response_sentiment}"}
    # response = requests.post(fastapi_endpoint, json=log_data, headers=headers, timeout=10)
    # if response.status_code == 200:
    #     logger.info("Query logged successfully")

    st.session_state.messages.append({"role": "assistant", "content": full_response})





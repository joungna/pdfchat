import sys
import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 제목
st.title("ChatPDF")
st.write("---")

# OpenAI KEY 입력 받기
openai_key = st.text_input('OPEN_AI_API_KEY', type="password")

# 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 올려주세요!", type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    temp_dir.cleanup()  # 임시 디렉토리를 사용 후 삭제
    return pages

# 업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)

    if texts:  # 텍스트가 비어있지 않은지 확인
        # Embedding
        embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)

        try:
            # Load it into Chroma
            db = Chroma.from_documents(texts, embeddings_model)
        except Exception as e:
            st.error(f"Chroma 데이터베이스 로딩 중 에러 발생: {str(e)}")
            db = None

        if db:
            # Stream 받아 줄 Handler 만들기
            from langchain.callbacks.base import BaseCallbackHandler
            class StreamHandler(BaseCallbackHandler):
                def __init__(self, container, initial_text=""):
                    self.container = container
                    self.text = initial_text

                def on_llm_new_token(self, token: str, **kwargs) -> None:
                    self.text += token
                    self.container.markdown(self.text)

            # Question
            st.header("PDF에게 질문해보세요!!")
            question = st.text_input('질문을 입력하세요')

            if st.button('질문하기'):
                with st.spinner('Wait for it...'):
                    chat_box = st.empty()
                    stream_handler = StreamHandler(chat_box)
                    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5, openai_api_key=openai_key, streaming=True, callbacks=[stream_handler])
                    qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
                    qa_chain({"query": question})

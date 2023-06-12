import streamlit as st
import os
import pickle
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


def main():
  st.title('Chat with pdf💬')
  load_dotenv()

  pdf = st.file_uploader("Upload your PDF", type = 'pdf')
  
  #st.write(pdf)
  if pdf is not None:
    pdf_reader = PdfReader(pdf)
    
    text=""
    for page in pdf_reader.pages:
       text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
       chunk_size= 1000,
       chunk_overlap=200,
       length_function= len
    )

    chunks = text_splitter.split_text(text=text)

    #embeddings
    
    store_name= pdf.name[:-4]
    
    if os.path.exists(f"{store_name}.pkl"):
       with open(f"{store_name}.pkl","rb") as f:
          vectorstore = pickle.load(f)
       #st.write("disk")
       
    else:
       embeddings = OpenAIEmbeddings() 
       vectorstore  = FAISS.from_texts(chunks, embeddings)
       with open(f"{store_name}.pkl","wb") as f :
          pickle.dump(vectorstore ,f)
       #st.write("comp.")

    #Ask questions 
    question = st.text_input("ask questions about pdf")
    
    if question:
        docs = vectorstore.similarity_search(query=question , k=3)
        
        llm = OpenAI(temperature=0.7 , model_name = 'gpt-3.5-turbo')
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
         response = chain.run(input_documents=docs, question=question)
         print(cb)        
        st.write(response)

if __name__ == '__main__':
    main()
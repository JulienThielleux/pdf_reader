from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

#read the content of the pdfs folder and aggregate the content in a String.
files = os.listdir('files')

raw_text = ''
for file in files:
    if file.endswith('.pdf'):
        print(file)
        doc_reader = PdfReader(f'files/{file}')
        size = 0
        for i, page in enumerate(doc_reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
                size += len(text)
        print(f'Size of file: {size}')
    if file.endswith('.txt'):
        print(file)
        with open(f'files/{file}', 'r', encoding='utf-8') as f:
            txt_read = f.read()
            raw_text += txt_read
            print(f'Size of file: {len(txt_read)}')

#Split the texts into 500 char chunks with 100 char overlap.
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 500,
    chunk_overlap = 100,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

#Create the vector space using openAi Embedding.
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts,embeddings)

chain = load_qa_chain(OpenAI(),chain_type='stuff')


#Ask the user for a question related to the documents.
while True:
    query = input("\nEnter a question (or type 'quit' to exit):")

    if query.lower() == "quit":
        break

    docs = docsearch.similarity_search(query)
    answer = chain.run(input_documents=docs, question=query)
    print(answer)
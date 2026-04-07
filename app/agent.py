from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

#Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#Load FAISS index
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

pipe = pipeline(
    "text-generation",
    model="google/flan-t5-small",
    max_new_tokens=100,
    do_sample=True,
    temperature=0.1
)

llm = HuggingFacePipeline(pipeline=pipe)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 1}),
    return_source_documents=True
)

# Test
result = qa.invoke({"query": "What is this document about?"})
print(result["result"])
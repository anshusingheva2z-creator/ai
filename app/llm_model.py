from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="google/flan-t5-small",
    max_new_tokens=100,
)

llm = HuggingFacePipeline(pipeline=pipe)
print("LLM loaded successfully")
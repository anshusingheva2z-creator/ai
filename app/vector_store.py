import os
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

print("vector_store.py is running...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_FILE = os.path.join(BASE_DIR, "faq.txt")
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")


def build_vector_store():

    print("Looking for file at:", FAQ_FILE)
    print("File exists?", os.path.exists(FAQ_FILE))

    if not os.path.exists(FAQ_FILE):
        raise FileNotFoundError("faq.txt not found inside app folder.")

    print("\nLoading faq.txt...")

    with open(FAQ_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by blank line
    blocks = re.split(r'\n\s*\n', content.strip())

    documents = []

    for block in blocks:
        q_match = re.search(r'Q(\d+)\.\s*(.*)', block, re.IGNORECASE)
        a_match = re.search(r'A\.\s*(.*)', block, re.IGNORECASE | re.DOTALL)

        if q_match and a_match:
            q_number = q_match.group(1)
            question_text = q_match.group(2).strip()
            answer_text = a_match.group(1).strip()

            full_block = f"Q{q_number}. {question_text}\nA. {answer_text}"

            documents.append(
                Document(
                    page_content=full_block,
                    metadata={"question_number": q_number}
                )
            )

    if not documents:
        raise ValueError("No valid Q&A blocks found. Check faq.txt formatting.")

    print(f"Extracted {len(documents)} Q&A entries")

    print("\nLoading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    vectorstore.save_local(INDEX_PATH)

    print("\nFAISS index created successfully at:", INDEX_PATH)


if __name__ == "__main__":
    build_vector_store()

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS


# Load the PDF
loader = PyMuPDFLoader("mental_health_book.pdf")
documents = loader.load()

# Check PDF content loaded
print(f"Loaded {len(documents)} pages.")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# Check chunking
print(f"Split into {len(chunks)} chunks.")

# Generate embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Build FAISS index
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save FAISS index
vectorstore.save_local("faiss_index")

print("✅ FAISS index built and saved in 'faiss_index/' folder.")

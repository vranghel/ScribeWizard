'''
configuration file that sets all the global variables and classes used throughout the app
'''
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from groq import Groq

global EMBED_MODEL, PIPELINE, GROQ_CLIENT, VECTOR_INDEX

EMBED_MODEL = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
PIPELINE = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=200, chunk_overlap=20),
        EMBED_MODEL,
    ]
)
#groq_api_key = st.secrets["GROQ_API_KEY"]
GROQ_CLIENT = Groq()  #api_key=groq_api_key
VECTOR_INDEX: VectorStoreIndex = None
import os

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import PGVector

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host=os.environ['TINYLLM_POSTGRES_HOST'],
    port=os.environ['TINYLLM_POSTGRES_PORT'],
    database=os.environ['TINYLLM_POSTGRES_NAME'],
    user=os.environ['TINYLLM_POSTGRES_USERNAME'],
    password=os.environ['TINYLLM_POSTGRES_PASSWORD'],
)


def get_vector_collection(collection_name) -> PGVector:
    return PGVector(connection_string=CONNECTION_STRING,
                    embedding_function=OpenAIEmbeddings(),
                    collection_name=collection_name)

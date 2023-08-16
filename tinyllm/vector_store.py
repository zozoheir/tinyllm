from sqlalchemy import asc
import os

from langchain.embeddings import OpenAIEmbeddings
from sqlalchemy import create_engine, Column, Integer, String, UniqueConstraint, JSON, and_, or_, cast
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector

import tinyllm

Base = declarative_base()


def get_database_uri():
    user = os.getenv('TINYLLM_POSTGRES_USERNAME', 'default_user')
    password = os.getenv('TINYLLM_POSTGRES_PASSWORD', 'default_password')
    host = os.getenv('TINYLLM_POSTGRES_HOST', 'localhost')
    port = os.getenv('TINYLLM_POSTGRES_PORT', '5432')
    name = os.getenv('TINYLLM_POSTGRES_NAME', 'default_db_name')
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"



class Embeddings(Base):
    __tablename__ = "embeddings"
    __table_args__ = (UniqueConstraint('text', name='uq_text'),)

    id = Column(Integer, primary_key=True)
    collection_name = Column(String, nullable=False)
    embedding = Column(Vector(dim=1536))
    text = Column(String)
    emetadata = Column(JSON)


class VectorStore:
    def __init__(self):
        self._engine = create_engine(get_database_uri())
        self._Session = sessionmaker(bind=self._engine)
        self.embedding_function = OpenAIEmbeddings()

    def _get_query_embedding(self, query):
        return self.embedding_function.embed_documents([query])[0]

    def create_tables(self):
        Base.metadata.create_all(self._engine)

    def add_texts(self, texts, metadatas, collection_name):
        embeddings = self.embedding_function.embed_documents(texts)

        with self._Session() as session:
            for text, embedding, metadata in zip(texts, embeddings, metadatas):
                embedding_obj = Embeddings(text=text, embedding=embedding, emetadata=metadata, collection_name=collection_name)
                session.merge(embedding_obj)
            session.commit()

    def _build_metadata_filters(self, metadata_filters):
        filter_clauses = []
        for key, value in metadata_filters.items():
            values = [str(val) for val in (value if isinstance(value, list) else [value])]
            filter_clauses.append(or_(cast(Embeddings.emetadata[key].op('->>')(key), String) == val for val in values))
        return filter_clauses

    def similarity_search(self, query, k, collection_filters, metadata_filters=None):
        query_embedding = self._get_query_embedding(query)

        with self._Session() as session:
            filter_clauses = self._build_metadata_filters(metadata_filters or {})
            filter_clauses.append(Embeddings.collection_name.in_([collection_filters] if isinstance(collection_filters, str) else collection_filters))

            results = (
                session.query(Embeddings, Embeddings.embedding.cosine_distance(query_embedding).label('distance'))
                .filter(*filter_clauses)
                .order_by(asc("distance"))
                .limit(k)
                .all()
            )

            return [{'text': r.Embeddings.text, 'metadata': r.Embeddings.emetadata, 'distance': r.distance} for r in results]

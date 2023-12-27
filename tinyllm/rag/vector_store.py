import tinyllm

from sqlalchemy import text
from sqlalchemy import Column, Integer, String, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import insert
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from tinyllm.function import Function
from tinyllm.rag.document.document import Document, DocumentTypes
from tinyllm.tracing.langfuse_context import observation

Base = declarative_base()


def get_database_uri():
    postgres_config = tinyllm.tinyllm_config['POSTGRES']
    user = postgres_config['USERNAME']
    password = postgres_config['PASSWORD']
    host = postgres_config['HOST']
    port = postgres_config['PORT']
    name = postgres_config['NAME']
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{name}"


class Embeddings(Base):
    __tablename__ = "embeddings"
    __table_args__ = (UniqueConstraint('text', 'collection_name', name='uq_text_collection_name'),)

    id = Column(Integer, primary_key=True)
    collection_name = Column(String, nullable=False)
    embedding = Column(Vector(dim=384))
    text = Column(String)
    emetadata = Column(postgresql.JSON)


class VectorStore(Function):

    def __init__(self,
                 embedding_function):
        self.name = 'PGVectorStore'
        self._engine = create_async_engine(get_database_uri())
        self._Session = sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
        self.embedding_function = embedding_function
        # self.create_tables()

    def _build_metadata_filters(self, metadata_filters):
        filter_clauses = []
        for key, value in metadata_filters.items():
            if isinstance(value, list):
                str_values = [str(v) for v in value]  # Convert all values to string
                filter_by_metadata = Embeddings.emetadata[key].astext.in_(str_values)
                filter_clauses.append(filter_by_metadata)
            elif isinstance(value, dict) and "in" in map(str.lower, value.keys()):
                value_case_insensitive = {k.lower(): v for k, v in value.items()}
                filter_by_metadata = Embeddings.emetadata[key].astext.in_(value_case_insensitive["in"])
                filter_clauses.append(filter_by_metadata)
            else:
                filter_by_metadata = Embeddings.emetadata[key].astext == str(value)
                filter_clauses.append(filter_by_metadata)
        return filter_clauses

    async def create_tables(self):
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def add_texts(self, texts, collection_name, metadatas=None):
        if metadatas is None:
            metadatas = [None] * len(texts)

        embeddings = self.embedding_function(texts)

        async with self._Session() as session:
            async with session.begin():
                for text, embedding, metadata in zip(texts, embeddings, metadatas):
                    stmt = insert(Embeddings).values(
                        text=text,
                        embedding=embedding,
                        emetadata=metadata,
                        collection_name=collection_name
                    ).on_conflict_do_nothing()

                    await session.execute(stmt)
                await session.commit()

        return {
            "documents": [
                {
                    "document": Document(content=text,
                                         type=DocumentTypes.TEXT,
                                         metadata=metadata),
                    "embedding": embedding

                } for text, embedding, metadata in zip(texts, embeddings, metadatas)
            ]
        }

    @observation(observation_type='span')
    async def similarity_search(self, query, k, collection_filters, metadata_filters=None):
        query_embedding = self.embedding_function(query)

        async with self._Session() as session:  # Use an asynchronous session
            async with session.begin():
                # Initialize the base SQL query
                base_query = 'SELECT text, emetadata, collection_name, embedding <-> :embedding AS distance FROM embeddings'

                # Initialize the WHERE clauses and parameters
                where_clauses = []
                params = {'embedding': str(list(query_embedding)), 'k': k}

                # Apply collection filters if they exist
                if collection_filters:
                    where_clauses.append('collection_name = ANY(:collection_filters)')
                    params['collection_filters'] = collection_filters

                # Apply metadata filters if they exist
                if metadata_filters:
                    for key, values in metadata_filters.items():
                        where_clauses.append(f"emetadata->>'{key}' = ANY(:{key})")
                        params[key] = values

                # Construct the final query
                if where_clauses:
                    final_query = f"{base_query} WHERE {' AND '.join(where_clauses)} ORDER BY distance ASC LIMIT :k"
                else:
                    final_query = f"{base_query} ORDER BY distance ASC LIMIT :k"

                # Execute the query
                stmt = text(final_query)
                result = await session.execute(stmt, params)
                rows = result.all()

            docs = [
                {
                    'document': Document(content=row.text,
                                         type=DocumentTypes.TEXT,
                                         metadata=row.emetadata),
                    'distance': row.distance,
                } for row in rows
            ]

            return {
                "documents": docs,
            }

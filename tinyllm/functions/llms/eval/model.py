from sqlalchemy import Column, UniqueConstraint, Integer, String
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Evaluations(Base):
    __tablename__ = "embeddings"
    __table_args__ = (UniqueConstraint('text', 'collection_name', name='uq_text_collection_name'),)

    id = Column(Integer, primary_key=True)
    context = Column(String)
    chat_response = Column(String)
    question = Column(String)
    correct_answer = Column(String)
    generated_answer = Column(String)
    generation_id = Column(String)
    scores = Column(postgresql.JSON)
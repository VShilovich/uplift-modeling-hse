from sqlalchemy import Column, Integer, String, Float, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class History(Base):
    __tablename__ = "history"
    
    id = Column(Integer, primary_key=True, index=True)
    ts = Column(String)
    processing_time = Column(Float)
    input_size = Column(Integer)
    input_tokens = Column(Integer)
    status_code = Column(Integer)
    input_data = Column(Text)
    output_data = Column(Text)

class Admin(Base):
    __tablename__ = "admins"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)

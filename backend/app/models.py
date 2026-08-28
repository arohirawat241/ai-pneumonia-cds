from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.sql import func
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class Case(Base):
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    image_path = Column(String)
    prediction = Column(String)
    confidence = Column(Float)
    prob_normal = Column(Float)
    prob_pneumonia = Column(Float)
    user_id = Column(Integer, ForeignKey("users.id")) # Link case to user
    created_at = Column(DateTime(timezone=True), server_default=func.now())
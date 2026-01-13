from sqlmodel import create_engine, Session
from app.config import DATABASE_URL

engine = create_engine(DATABASE_URL, echo=False)

def get_session():
    with Session(engine) as session:
        yield session

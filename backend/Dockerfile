FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install (including the bcrypt fix!)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install bcrypt==3.2.2

# Copy the backend code and the ML pipeline
COPY backend/ ./backend/
COPY ml_pipeline/ ./ml_pipeline/

EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
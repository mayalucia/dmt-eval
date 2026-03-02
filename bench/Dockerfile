FROM python:3.12-slim

WORKDIR /app

# Install dmt-eval + streamlit dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the dmt package
COPY src/ /app/src/

# Copy the bench app
COPY bench/ /app/bench/

ENV PYTHONPATH=/app/src
EXPOSE 8501

CMD ["streamlit", "run", "bench/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]

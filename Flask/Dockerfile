# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY App.py .
COPY templates ./templates
COPY models ./models

# Expose port
EXPOSE 5000

# Use gunicorn to run the Flask app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "App:app", "--workers", "1", "--threads", "4"]

FROM python:3.8-slim-buster

# Install system dependencies
RUN apt update -y && apt install -y awscli

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose Flask port
EXPOSE 8080

# Run the app
CMD ["python3", "app.py"]
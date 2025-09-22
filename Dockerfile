FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y unzip curl \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip aws

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN python3 -c "import dotenv; print('✅ dotenv is installed')"

# Expose Flask port
EXPOSE 8080

# Run the app
CMD ["python3", "app.py"]
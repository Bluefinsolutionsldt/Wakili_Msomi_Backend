FROM python:3.9-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make start script executable
RUN chmod +x start.sh

# Set environment variables
ENV PORT=8001
ENV HOST=0.0.0.0

# Expose the port
EXPOSE 8001

# Command to run the application
CMD ["./start.sh"]

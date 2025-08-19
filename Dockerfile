# Use a specific Python version (3.11) that is compatible with RDKit
FROM python:3.11-slim

# Install an updated version of sqlite3 and a C compiler for building Python packages.
RUN apt-get update && apt-get install -y \
    libsqlite3-dev \
    gcc

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all other project files into the container
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit application when the container starts
CMD ["streamlit", "run", "app.py"]
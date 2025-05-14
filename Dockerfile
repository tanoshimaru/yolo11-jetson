# FROM ultralytics/ultralytics:latest-jetson-jetpack6
FROM ultralytics/ultralytics:latest

# Set the working directory
WORKDIR /app

# # Copy the requirements file
# COPY requirements.txt .

# # Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the source code
# COPY main.py .

# # Set the entry point
# CMD ["python", "main.py"]
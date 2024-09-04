# Use the specified ultralytics image as the base image
FROM ultralytics/ultralytics:latest-jetson-jetpack4

# Set the working directory in the container to /app
WORKDIR /app

# Install any necessary dependencies if your project has a requirements.txt file
# Uncomment the following line if you have a requirements.txt
RUN pip3 install prometheus_client

# Copy the content of the app folder into the /app directory in the container
COPY . /app


# Expose any ports your application might use (optional)
EXPOSE 8000 50051

# Define the command to run your application
# Replace "your_script.py" with the entry point of your application
CMD ["python3", "main.py"]


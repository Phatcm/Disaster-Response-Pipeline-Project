# Use the slim variant of Python 3.10
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install the requirements
RUN pip install --no-cache-dir -r requirements.txt

# Expose the ports your application will run on
# Use port 80 for Azure App Service (HTTP)
EXPOSE 3000 80 443

# Run the application
CMD ["python", "run.py"]

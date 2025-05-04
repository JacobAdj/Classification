# Use the official Python image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy project files into container
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Expose API port (if using BentoML)
EXPOSE 3000

# Run service (Modify command based on your project)
CMD ["bentoml" , "serve" , "service.py"]

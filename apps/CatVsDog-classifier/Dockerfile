# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Create non-root user required by Hugging Face Spaces
RUN useradd -m -u 1000 user

# Set workdir and environment
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

# Copy requirements first
COPY requirements.txt .

# Install requirements as root
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Switch to user AFTER installing packages
USER user

# Copy the rest of the app (model, code, etc.)
COPY --chown=user . .

# Expose the port required by HF Spaces
EXPOSE 7860

# Launch the app
CMD ["python", "app.py"]

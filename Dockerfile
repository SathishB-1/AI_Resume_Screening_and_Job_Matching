FROM python:3.10-slim

WORKDIR /app

# install system deps
COPY requirements.txt .

# Install lightweight CPU torch first
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Then install remaining libraries
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
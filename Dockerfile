# Use uma imagem base com Python 3.9 ou superior
FROM python:3.9-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o arquivo de requisitos
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação para o container
COPY . .

# Exponha a porta 5050
EXPOSE 5050

# Comando para iniciar o servidor FastAPI
CMD ["uvicorn", "chat_with_website_ollama:app", "--host", "0.0.0.0", "--port", "5050"]

Ollama install:

brew install ollama

ollama serve

ollama run llama3.1

--
brew install python@3.11
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

--

Run API

uvicorn app:app --reload --port 8000

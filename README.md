# ⚙️🤖 Llama_index Knowledge Graph Project

Parsing Palantir’s financial and earnings reports. Leveraging and integrating tools like **LlamaIndex**, **MLfLow**, **Azure Databricks**,  and **Neo4j** the project transforms unstructured financial text into a powerful Graph index.

## 📦 Features

- Indexing with LlamaIndex
- Azure OpenAI LLM of choice
- Embedding support (HuggingFace, LangChain, OpenAI)
- Graph storage via Neo4j and NetworkX
- Modular and extensible design ( MlFlow, Async)


## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Andreas-Pash/PLTR_x_LlamaIndex.git
cd PLTR_x_LlamaIndex
````

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
python main.py
```

---

## 🛠 Project Structure

```
├── main.py                     # Entry point
├── run.sh                      # Helper script to run app
├── requirements.txt            # Pip dependencies
├── pyproject.toml              # (Optional) Poetry/modern packaging
├── constants.py                # Prompt templates, configs
├── azure_envars.py             # Legacy env setup
├── data/                       # Parsed documents
├── scripts/                     
├── src/                        # Core logic
│   ├── helper_functions.py
│   ├── graph_rag_store.py
│   └── llama_index_graph_rag_extractor.py
    └── pdf_parser.py
├── tests/                      # Unit tests
```

---



## 📚 Dependencies

See `requirements.txt` for the full list, including:

* `pydantic>=2.7`
* `langchain`, `langchain_community`
* `llama-index`, `llama-index-*`
* `neo4j`, `graspologic`
* `openai`, `huggingface`

---

## 🕵️ Running Tests

```bash
pytest tests/
```

---

## 📖 License

MIT License

---

## 🙋‍♂️ Questions?

Feel free to open an issue or reach out!
 
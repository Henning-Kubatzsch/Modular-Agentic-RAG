# Modular Agentic RAG

![Project preview](<hero_image.png>)

---
# 🌍 Vision

The objective is to build a high-performance, **Modular Agentic RAG Framework** that serves as a robust backend for specialized instruction agents.

1.  **API-First & Modular Design**
    *   The core logic is exposed via a **FastAPI** backend, allowing low-latency communication with any frontend.
    *   **Controllability:** Key model hyperparameters (temperature, top_k, ...) and retrieval settings are not hardcoded but can be dynamically adjusted via the User Interface for real-time testing.

2.  **Privacy & Compliance (GDPR)**
    *   Strictly local execution ensures data sovereignty.
    *   Ideal for sensitive domains where cloud processing is prohibited (e.g., medical data processing).

3.  **Specialized Use-Case: Resuscitation Support Agent**
    *   A proof-of-concept implementation designed to support first responders in real-time.
    * **Custom Data Pipeline & QA-Mapping:** utilizes a specialized **Question-Answer indexing strategy**. Instead of raw text chunking, the ERC guidelines were re-engineered into a custom document structure of mapped Q&A pairs. By embedding **potential input questions** rather than the target content, the system optimizes semantic alignment with user queries, ensuring high retrieval precision and a stable dialogue flow even with smaller Local LLMs.

---

# 🏗️ System Architecture

*   **Backend:** Python / FastAPI (Handles RAG logic, Vector Store, LLM Inference)
*   **Frontend:** Separate Repository (UI for Chat, Monitoring, Evaluation, Logging and Settings)
*   **Communication:** REST API (Low-latency asynchronous calls)
*   **Data Layer:** Local Vector Index (currently utilizing **hnswlib** for high-efficiency nearest neighbor search) + Custom Ingestion Tools

---

# 🛣️ Development Phases (Roadmap)

**Phase 1 – Core Architecture (Completed)**
*   [x] **Decoupling:** Split system into a FastAPI Backend and a standalone Frontend repository.
*   [x] **Low-Latency Pipeline:** Established efficient async communication between UI and RAG engine.
*   [x] **UI Control:** Implemented interface elements to adjust model parameters and retrieval strategies on the fly.

**Phase 2 – Advanced Data Processing (Current Focus)**
*   [x] **Custom Ingestion Tool:** Developed a specialized document parser that preserves layout semantics.
*   [x] **Small Model Optimization:** Tuned chunking strategies to support the constrained context windows of local models while enabling natural dialogue structures.
*   [ ] **Evaluation Loop:** Connecting the parameter settings in the UI directly to an evaluation module to measure the impact of changes on response quality.

**Phase 3 – Agentic Logic & State Management**
*   [ ] Transform the retrieval process into an **Agentic Workflow** (planning steps before answering).
*   [ ] Implement state tracking for the resuscitation scenario (Agent remembers: "Check response" -> "Call 112" -> "Start CPR").

**Phase 4 – Deployment & Real-World Simulation**
*   [ ] Containerize the application (Docker) for easy local deployment.
*   [ ] Final tests with real-time streaming data from the MR simulation environment.

---

# Setup Guide

- [Full Setup Guide](guides/setup.md)
- [Poetry + Pyenv Setup (all OS)](guides/setup_poetry_pyenv.md)

---

# Navigate the project

## Create Index (default)  

- You need to provide your own datasource for the model. **No data is included in this repository.**
- To start indexing, place a first aid instruction document (`.txt`) into the `root/data/docs` directory.    

Then run the following command:

```shell
# Standard CLI call
poetry run rag ingest
```

  
- To use other options, you can modify the parameters in the `ingest` method in `src/rag/cli.py`.  

| Parameter          | Description                                                                         |
| ------------------ | ----------------------------------------------------------------------------------- |
| `--docs`           | Folder containing the source documents (default: `data/docs`)                       |
| `--out`            | Folder where the HNSW index will be stored (default: `data/index`)                  |
| `--embed-model`    | Embedding model to use, e.g., `sentence-transformers/all-MiniLM-L6-v2`              |
| `--custom_chunker` | Use the **CustomChunker** format for highly structured documents (default: `False`) |

  
---

## CustomChunker (`--custom_chunker`)

An alternative chunking strategy is available to improve retrieval quality and model output.  
This option requires a datasource with a specific input format.

Enable it with:

```shell
poetry run rag ingest --custom_chunker
```

### Required Input Structure

```
---
Question:
* question 1
* question 2
* question 3  

Title:
* task 1
* task 2
* task 3
---
```

### Processing Steps

- Each section (`Question` and `Title`) is **chunked separately**    
- Each question is **embedded individually**    
- The corresponding tasks are added as **context for each question**    
- Each chunk is assigned a **unique ID** and optional metadata from the source document    
- Retrieval is optimized by comparing input queries against the embedded questions



### Example CLI Call Using Custom Chunker

```shell
poetry run rag ingest --docs data/docs --out data/index --embed-model sentence-transformers/all-MiniLM-L6-v2 --custom_chunker
```

## Interact Without Server – New Model Instance per Run

| code                  | description                      |
| --------------------- | -------------------------------- |
| poerty run rag module | run any module in src/rag/cli.py |
| -c configs/rag.yaml   | configs for llm                  |
| --index data/index    | indexes for retriever            |

modules:

| **module**              | **required** | RAG |
| ----------------------- | ------------ | --- |
| ask                     | str          | yes |
| llm-stream              | str          | yes |
| ask-no-retrieval        | str          | no  |
| llm-sanity-no-retrieval | str          | no  |
| llm-stream-no-retrieval | str          | no  |

```shell
	poetry run rag llm-stream -c configs/rag.yaml --index data/index \
	  "what is the rhytm or speed for applying CPR?"
```

works also:

```shell
	poetry run rag llm-stream "what is the rhytm or speed for applying CPR?"
```

## Start + Use local server 

### Terminal 1 - server

run `async def lifespan(app: FastAPI)`in `src/rag/server.py`:

```bash
poetry run uvicorn rag.server:app --host 127.0.0.1 --port 8000 --reload
```

### Terminal 2 - client

#### Healthcheck:
  

```bash
curl http://127.0.0.1:8000/health
# -> {"ok": true}
```


#### RAG query (retrieval + generation):

use streaming:  

```bash
curl -N -X POST http://127.0.0.1:8000/rag \
-H "Content-Type: application/json" \
-d '{"q": "What should I do next for an unresponsive adult?"}'
```

  
without streaming:  

```bash
curl -N -X POST http://127.0.0.1:8000/rag_new \
-H "Content-Type: application/json" \
-d '{"q": "What is the correct rate for performing CPR compressions"}'
```

  
| Option                                                           | Meaning                                                | Why it matters here                                                  |
| ---------------------------------------------------------------- | ------------------------------------------------------ | -------------------------------------------------------------------- |
| `-N`                                                             | **Disable buffering** (no output buffering in `curl`). | Ensures you see tokens immediately, instead of buffered all at once. |
| `-X POST`                                                        | **HTTP method POST**.                                  | The `/rag` endpoint expects POST, not GET.                           |
| `http://127.0.0.1:8000/rag`                                      | **Target URL**.                                        | Local FastAPI server running on port 8000.                           |
| `-H "Content-Type: application/json"`                            | **Set request header**.                                | Tells the server the body is JSON.                                   |
| `-d '{"q": "What should I do next for an unresponsive adult?"}'` | **Request body (data)**.                               | Provides the input (`q`) that your RAG system should answer.         |
 
shorter way  

```bash

curl --json '{"q":"What to do when arriving at an accident?"}' http://127.0.0.1:8000/rag
```

---

# ⚠️ Disclaimer

This project is **for training and research purposes only**.
It does **not** replace certified medical training or real emergency protocols.
Always follow official ERC guidelines and seek certified instruction.

## Legal Notice

This project uses **paraphrased and simplified summaries** inspired by the recommendations of the **European Resuscitation Council (ERC)**.
It does **not** include or redistribute official ERC guideline texts.
For authoritative and up-to-date information, consult the official ERC publications at [https://erc.edu](https://erc.edu).

---


# 📄 License
This project is licensed under the **[Apache License 2.0](LICENSE)**.
For attribution, third-party dependencies, and additional notices, see:
- **[LICENSE](LICENSE)** – Full license text
- **[NOTICE.md](NOTICE.md)** – Attribution and project-specific notices
- **[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)** – Licenses of included dependencies


---

# Notes & Gotchas (read this)

* Local models can hallucinate. Keep prompts concrete and verify outputs against ERC sources.
* Smaller GGUF models run fast on M-series CPUs/Metal but may reduce accuracy—benchmark before training sessions.
* Index quality beats model size: clean, focused chunks outperform noisy text dumps.
* Keep the project **offline** to preserve privacy; avoid dropping in cloud SDKs by habit.

# MR First Aid Training Agent

![Project preview](<hero_image.png>)

🚑 **Project Idea**

A **local, offline intelligent training agent** for **First Aid and resuscitation**.
Long-term vision: an **interactive Mixed Reality (MR)** environment (Quest 3, multi-user) where trainees practice according to **ERC (European Resuscitation Council) guidelines**.
A pedagogical agent, powered by **RAG (Retrieval-Augmented Generation)** and **Agentic RAG**, guides users step by step and adapts in real time.

---

# 🌍 Vision

1. **Local & Offline**

   * Runs on local hardware (MacBook M1 Pro → later Meta Quest 3).
   * No cloud services → private, portable, resilient.

2. **Pedagogical Agent**

   * Trainees ask: *“What should I do next?”*
   * Answers grounded in ERC guidance and tracked progression (ambulance called, CPR started, AED used, …).
   * Later: integrate physiological parameters from a **resuscitation dummy** (blood pressure, compression depth, O₂ saturation) for adaptive feedback.

3. **Agentic RAG**

   * Beyond retrieval: the agent **decides how to react**, evaluates context, and provides next-step instructions.
   * Systematically test chunking, data formatting, and model sizes for clarity and effectiveness.

---

# 🛣️ Development Phases (Roadmap)

**Phase 1 – Local RAG baseline (current)**

* Run a simple offline RAG system on Mac M1 Pro.
* ERC guidelines as knowledge base.
* Local LLM only (e.g., Qwen GGUF via `llama-cpp-python`).
* Conversational CLI with progression tracking.

**Phase 2 – Graph-based progression**

* Model the resuscitation process with LangGraph or a custom state machine.
* Natural-language actions update the progression state.

**Phase 3 – Agentic RAG evaluation**

* Clean/tune ERC text (chunking).
* Test small local LLMs for instruction quality.
* Benchmark clarity and correctness.

**Phase 4 – Standalone Quest 3 deployment**

* Fully local execution on-device.
* Voice input and agent speech output in MR.

**Phase 5 – DummyStation integration**

* Stream real-time parameters from a resuscitation dummy.
* Agent adapts dynamically (e.g., compression depth too shallow → corrective feedback).

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

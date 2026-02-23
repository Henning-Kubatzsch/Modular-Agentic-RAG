# src/rag/server.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Iterable
import yaml
from contextlib import asynccontextmanager
import logging, gc
from pydantic import BaseModel
from typing import Optional

from rag.generator import LocalLLM, LLMConfig
from rag.embed import SBertEmbeddings
from rag.indexer import HnswIndex
from rag.retriever import Retriever, RetrieverConfig
from rag.prompt import build_prompts, postprocess_answer, PromptOptions, PromptOptionsOverride, merge_prompt_options
from rag.settings import get_settings

log = logging.getLogger(__name__)

# --- Global state container (simple singleton) ---
class State:
    llm: LocalLLM | None = None
    embedder: SBertEmbeddings | None = None
    index: HnswIndex | None = None
    retriever: Retriever | None = None

class RagRequest(BaseModel):
    q: str
    options: Optional[PromptOptionsOverride] = None 

def get_prompt_defaults():
    return get_settings().prompt

S = State()
yaml_path = "configs/rag.yaml"

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        return cfg

def load_llm_config(path: str) -> LLMConfig:
    """Read YAML config file and construct LLMConfig."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    llm_cfg = cfg.get("llm", {})
    return LLMConfig(**llm_cfg)


# --- Lifecycle manager: startup + teardown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # --- Startup: initialize services ---
        cfg = load_llm_config(yaml_path)
        S.llm = LocalLLM(cfg)
        S.embedder = SBertEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",  # use "mps" only if stable on your machine
        )
        S.index = HnswIndex()
        S.index.load("data/index")
        S.retriever = Retriever(S.embedder, S.index, k=5)
        log.info("Warmup complete, server ready.")
        yield  # <-- server runs between startup and shutdown
    finally:
        # --- Shutdown: release resources cleanly ---
        try:
            if getattr(S.llm, "close", None):
                S.llm.close()  # e.g., HTTP sessions, background threads
        except Exception:
            log.exception("LLM close failed")
        try:
            if getattr(S.index, "close", None):
                S.index.close()  # e.g., memory-mapped index files
        except Exception:
            log.exception("Index close failed")
        # Drop references so GC can free memory (important on reloads/workers)
        S.retriever = None
        S.index = None
        S.embedder = None
        S.llm = None
        gc.collect()  # force GC – optional, helps with memory fragmentation
        log.info("Teardown complete.")


# --- FastAPI app ---
app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://0.0.0.0:3000",
    "http://[::1]:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,                 # oder ["*"] für Dev (siehe Hinweis unten)
    allow_origin_regex=r"https?://localhost(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],                   # wichtig für OPTIONS/POST
    allow_headers=["*"],                   # wichtig für Content-Type: application/json
    expose_headers=["Content-Type"],       # optional; bei Streaming/Debug hilfreich
)

@app.get("/health")
def health():
    """Simple health check endpoint."""
    return {"ok": True}

@app.post("/rag_ui")
async def rag_ui(req: RagRequest):

    opts = get_prompt_defaults()
    if(req.options):
        opts = merge_prompt_options(opts, req.options)
    
    q = req.q
    hits = S.retriever.search(q)
    system, user = build_prompts(q, hits, opts)
    msgs = S.llm.make_messages(user=user, system=system)

    def gen() -> Iterable[bytes]:
        buf = []
        try:
            for tok in S.llm.chat_stream(
                msgs, max_tokens=256, temperature=0.2):
                buf.append(tok)
                # this is the part why we get repeating output
                yield tok.encode("utf-8")  # live stream
        except Exception as e:
            yield f"\n\n[stream-error] {type(e).__name__}: {e}\n".encode("utf-8")
        finally:
            # optional: final „clean“ Ausgabe in Logs
            try:
                final = "".join(buf)
                clean = postprocess_answer(final, num_sources=len(hits), opts=opts)
                print(f"\n--- POSTPROCESSED ---\n{clean}")
            except Exception:
                pass
    # print("------------before POSTPTOCESSED")
    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")


@app.get("/get_config")
def get_config():
    try:
        config = load_config(yaml_path)
        t = type(config)
        config["type"] = str(t)
        return config
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Config file not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
@app.get("/config")
def getLLMConfig():
    try:
        llmConfig = LLMConfig.model_json_schema()
        promptConfig = PromptOptions.model_json_schema()
        retrieverConfig = RetrieverConfig.model_json_schema()
        return {
            "llm":llmConfig,
            "prompt":promptConfig,
            "retriever":retrieverConfig
        }
    except Exception as e:
        return {
            "error":str(e)
        }

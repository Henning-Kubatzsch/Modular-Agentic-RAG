# Given a query vector: top-k search (+ optional rerank); returns context items
# TODO: Interface + Abstaction

from __future__ import annotations
from typing import List, Dict, Any
from rag.interfaces import Embedder, VectorIndex
from dataclasses import dataclass
from pydantic import BaseModel, Field

import yaml

class RetrieverConfig(BaseModel):
    k: int = Field(default=4, ge=1, le=10)

yaml_path = "configs/rag.yaml"

def load_retriever_config(path: str) -> RetrieverConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    retriever_cfg = cfg.get("retriever", {})
    return RetrieverConfig(**retriever_cfg)


class Retriever:
    def __init__(self, embedder: Embedder, index: VectorIndex, k: int = 5):
        self.embedder = embedder
        self.index = index
        self.k = k

    def search(self, query: str) -> List[Dict[str, Any]]:

        cfg = load_retriever_config(yaml_path)
        k = cfg.k
        qv = self.embedder.embed_one(query)
        #return self.index.query(qv, k=self.k)
        return self.index.query(qv, k)


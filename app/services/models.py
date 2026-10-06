import logging
import threading
from functools import lru_cache

import torch
from fastembed import SparseTextEmbedding
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.core.config import settings

logger = logging.getLogger("kognit.models")

_lock = threading.Lock()


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _dense() -> SentenceTransformer:
    logger.info("Loading dense model %s", settings.dense_model)
    return SentenceTransformer(settings.dense_model, device=get_device())


@lru_cache(maxsize=1)
def _sparse() -> SparseTextEmbedding:
    logger.info("Loading sparse model %s", settings.sparse_model)
    return SparseTextEmbedding(model_name=settings.sparse_model)


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    logger.info("Loading reranker %s", settings.reranker_model)
    return CrossEncoder(settings.reranker_model, device=get_device())


def get_dense_model() -> SentenceTransformer:
    with _lock:
        return _dense()


def get_sparse_model() -> SparseTextEmbedding:
    with _lock:
        return _sparse()


def get_reranker() -> CrossEncoder:
    with _lock:
        return _reranker()
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.chat import router as chat_router
from app.api.documents import router as doc_router
from app.api.chat import retriever  

logger = logging.getLogger("kognit.gateway")

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    logger.info("[*] Warming up BGE dense, BM25 sparse, and BGE reranker models...")
    _ = retriever.client  
    logger.info("[+] Core retrieval models and Qdrant client ready.")
    yield
    logger.info("[-] Shutting down KOGNIT API Gateway...")

app = FastAPI(title="KOGNIT API Gateway", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(chat_router)
app.include_router(doc_router)

@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "healthy",
        "service": "kognit-core",
        "models_loaded": True
    }
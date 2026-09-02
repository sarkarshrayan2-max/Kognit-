from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.chat import router as chat_router
from app.api.documents import router as doc_router
from app.services.retrieval.fusion import HybridRetriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload models into memory/GPU on startup
    print("[*] Warming up BGE and BM25 models...")
    _ = HybridRetriever()
    print("[+] Models warmed up and ready.")
    yield


app = FastAPI(title="KOGNIT API Gateway", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(doc_router)


@app.get("/health")
def health():
    return {"status": "healthy", "service": "kognit-core"}
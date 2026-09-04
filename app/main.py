import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.documents import router as doc_router
from app.graph.workflow import retriever


logger = logging.getLogger("kognit.gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[*] Initializing KOGNIT services...")


    _ = retriever.client

    logger.info("[+] Qdrant client ready.")
    logger.info("[+] KOGNIT API ready.")

    yield

    logger.info("[-] Shutting down KOGNIT API.")


app = FastAPI(
    title="KOGNIT API Gateway",
    version="1.0.0",
    lifespan=lifespan,
)



app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        
        "http://localhost:3000",
        "http://127.0.0.1:3000",

        
        "http://localhost:5500",
        "http://127.0.0.1:5500",

        
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],

    allow_credentials=True,

    allow_methods=[
        "GET",
        "POST",
        "DELETE",
        "OPTIONS",
    ],

    allow_headers=["*"],

    expose_headers=["*"],
)




app.include_router(chat_router)
app.include_router(doc_router)




@app.get(
    "/health",
    tags=["Health"],
)
async def health():
    return {
        "status": "healthy",
        "service": "kognit-core",
        "qdrant_connected": True,
    }
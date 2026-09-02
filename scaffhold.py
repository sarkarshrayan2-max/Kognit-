from pathlib import Path

# Project root directory (current directory)
ROOT = Path(".")

# Directories that are Python packages (will get an __init__.py)
PYTHON_PACKAGES = [
    ROOT / "app",
    ROOT / "app" / "api",
    ROOT / "app" / "core",
    ROOT / "app" / "models",
    ROOT / "app" / "schemas",
    ROOT / "app" / "services",
    ROOT / "app" / "services" / "ingestion",
    ROOT / "app" / "services" / "retrieval",
    ROOT / "app" / "services" / "rag",
    ROOT / "app" / "services" / "llm",
    ROOT / "app" / "services" / "guardrails",
    ROOT / "app" / "services" / "search",
    ROOT / "tests",
]

# Non-package or resource directories (no __init__.py needed)
OTHER_DIRS = [
    ROOT / "training" / "data",
    ROOT / "training" / "configs",
    ROOT / "training" / "train",
    ROOT / "training" / "adapters",
    ROOT / "evaluation" / "datasets",
    ROOT / "evaluation" / "ragas",
    ROOT / "evaluation" / "deepeval",
    ROOT / "frontend",
    ROOT / "storage",
    ROOT / "docker",
]

# Specific target files to scaffold
FILES = [
    # Top-level configuration files
    ROOT / "docker-compose.yml",
    ROOT / "pyproject.toml",
    ROOT / "README.md",
    ROOT / ".env.example",

    # API endpoints
    ROOT / "app" / "main.py",
    ROOT / "app" / "api" / "auth.py",
    ROOT / "app" / "api" / "chat.py",
    ROOT / "app" / "api" / "documents.py",
    ROOT / "app" / "api" / "courses.py",
    ROOT / "app" / "api" / "faculty.py",

    # Core utilities & configs
    ROOT / "app" / "core" / "config.py",
    ROOT / "app" / "core" / "security.py",
    ROOT / "app" / "core" / "database.py",
    ROOT / "app" / "core" / "logging.py",

    # Data models (SQLAlchemy / SQLModel)
    ROOT / "app" / "models" / "user.py",
    ROOT / "app" / "models" / "document.py",
    ROOT / "app" / "models" / "course.py",
    ROOT / "app" / "models" / "chat.py",

    # Schemas (Pydantic)
    ROOT / "app" / "schemas" / "auth.py",
    ROOT / "app" / "schemas" / "chat.py",
    ROOT / "app" / "schemas" / "documents.py",

    # Ingestion pipeline
    ROOT / "app" / "services" / "ingestion" / "parser.py",
    ROOT / "app" / "services" / "ingestion" / "cleaner.py",
    ROOT / "app" / "services" / "ingestion" / "chunker.py",
    ROOT / "app" / "services" / "ingestion" / "indexer.py",

    # Retrieval & RRF
    ROOT / "app" / "services" / "retrieval" / "dense.py",
    ROOT / "app" / "services" / "retrieval" / "sparse.py",
    ROOT / "app" / "services" / "retrieval" / "fusion.py",
    ROOT / "app" / "services" / "retrieval" / "reranker.py",

    # RAG core & CRAG
    ROOT / "app" / "services" / "rag" / "retriever.py",
    ROOT / "app" / "services" / "rag" / "crag.py",
    ROOT / "app" / "services" / "rag" / "context.py",

    # LLM Gateway
    ROOT / "app" / "services" / "llm" / "gateway.py",
    ROOT / "app" / "services" / "llm" / "prompts.py",
    ROOT / "app" / "services" / "llm" / "generation.py",

    # Guardrails
    ROOT / "app" / "services" / "guardrails" / "input.py",
    ROOT / "app" / "services" / "guardrails" / "retrieval.py",
    ROOT / "app" / "services" / "guardrails" / "output.py",

    # Search Fallback
    ROOT / "app" / "services" / "search" / "restricted_web.py",
]


def scaffold_project() -> None:
    
    for pkg in PYTHON_PACKAGES:
        pkg.mkdir(parents=True, exist_ok=True)
        init_file = pkg / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            print(f"[+] Created: {init_file}")

    
    for directory in OTHER_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[+] Created dir: {directory}")

    
    for file_path in FILES:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            file_path.touch()
            print(f"[+] Created file: {file_path}")

    print("\nProject scaffolded successfully.")


if __name__ == "__main__":
    scaffold_project()
# FastAPI application entry point

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.api.chat import router as chat_router
from backend.api.session import router as session_router
from backend.api.health import router as health_router

# ── Create App ──
app = FastAPI(
    title="Serendib AI",
    description="Intelligent Sri Lanka Travel Assistant API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ── CORS First ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──
app.include_router(health_router)
app.include_router(session_router)
app.include_router(chat_router)

# ── Static Files ──
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# ── Endpoints ──
@app.get("/")
async def root():
    return {
        "app": "Serendib AI",
        "status": "running"
    }

@app.get("/ui")
async def serve_ui():
    return FileResponse("frontend/index.html")

@app.on_event("startup")
async def startup():
    print("\n🌴 Serendib AI Starting Up...")
    print("🚀 API ready at http://localhost:8000")
    print("📖 Docs at http://localhost:8000/docs\n")
#     {
#   "session_id": "c2af6105-c022-40f7-82cc-03e58514aa41",
#   "conversation_id": "6a37b5488c0485112fff7f39",
#   "message": "Session started successfully"
# }
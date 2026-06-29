import logging
import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from aiogram.types import Update
from bot import bot, dp
from config import config
from webhook import set_webhook
from handlers import router

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Register handlers
dp.include_router(router)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager."""
    # Startup
    logger.info("🚀 Starting PolyglotPulseBot...")
    await set_webhook()
    logger.info(f"✅ Webhook set to: {config.WEBHOOK_HOST}{config.WEBHOOK_PATH}")
    yield
    # Shutdown
    logger.info("🛑 Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="PolyglotPulseBot",
    description="A sophisticated Telegram translator bot",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/webhook")
async def webhook(request: Request) -> dict:
    """Handle incoming Telegram updates."""
    try:
        data = await request.json()
        update = Update(**data)
        await dp.feed_update(bot, update)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "bot": "@PolyglotPulseBot"}

@app.get("/ready")
async def readiness_check() -> dict:
    """Readiness check endpoint."""
    return {"status": "ready", "bot": "@PolyglotPulseBot"}

# The 'app' variable is what Gunicorn looks for

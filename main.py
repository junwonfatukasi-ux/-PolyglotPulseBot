import os
import logging
import re
from typing import Optional
from contextlib import asynccontextmanager
from dataclasses import dataclass

# FastAPI
from fastapi import FastAPI, Request

# Aiogram
from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message, Update

# ML Libraries
from langdetect import detect, DetectorFactory
from transformers import pipeline

# ============================================================
# SETUP & CONFIGURATION
# ============================================================

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration settings for the bot."""
    
    BOT_TOKEN: str = os.getenv("BOT_TOKEN", "")
    if not BOT_TOKEN:
        raise ValueError("❌ BOT_TOKEN environment variable is required!")
    
    PORT: int = int(os.getenv("PORT", 8080))
    WEBHOOK_HOST: str = os.getenv("RAILWAY_STATIC_URL", "https://your-app.railway.app")
    WEBHOOK_PATH: str = "/webhook"
    WEBHOOK_SECRET: str = os.getenv("TELEGRAM_WEBHOOK_SECRET", "default_secret")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


config = Config()

# ============================================================
# BOT SETUP
# ============================================================

bot = Bot(
    token=config.BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN)
)

storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# ============================================================
# LANGUAGE SERVICE
# ============================================================

class LanguageService:
    """Language detection service."""
    
    def detect_language(self, text: str) -> Optional[str]:
        """Detect language of text using langdetect."""
        if not text or len(text.strip()) < 2:
            return None
        
        try:
            detected = detect(text)
            logger.info(f"Detected language: {detected}")
            return detected
        except Exception as e:
            logger.debug(f"Language detection failed: {e}")
            return None

# ============================================================
# TRANSLATION SERVICE
# ============================================================

class TranslationService:
    """Translation service using Helsinki-NLP models."""
    
    def __init__(self):
        self._pipeline = None
        self._current_model = None
        logger.info("TranslationService initialized")
    
    async def translate(self, text: str, src_lang: str, tgt_lang: str) -> Optional[str]:
        """Translate text from source to target language."""
        if not text or not src_lang or not tgt_lang:
            return None
        
        if src_lang == tgt_lang:
            return text
        
        try:
            pipeline_obj = await self._get_pipeline(src_lang, tgt_lang)
            if not pipeline_obj:
                logger.error(f"No pipeline available for {src_lang}->{tgt_lang}")
                return None
            
            result = pipeline_obj(text, max_length=512, truncation=True)
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('translation_text', None)
            return None
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return None
    
    async def _get_pipeline(self, src_lang: str, tgt_lang: str):
        """Get or create translation pipeline."""
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        
        # Return cached pipeline if same model
        if self._pipeline and self._current_model == model_name:
            return self._pipeline
        
        try:
            logger.info(f"Loading model: {model_name}")
            
            self._pipeline = pipeline(
                "translation",
                model=model_name,
                tokenizer=model_name,
                device=-1,  # CPU
                max_length=512,
                truncation=True
            )
            
            self._current_model = model_name
            logger.info(f"✅ Model loaded: {model_name}")
            return self._pipeline
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None

# ============================================================
# INITIALIZE SERVICES
# ============================================================

language_detector = LanguageService()
translator = TranslationService()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_target_language(text: str) -> tuple[str, str]:
    """
    Extract target language from user message.
    Returns: (target_lang_code, cleaned_text)
    """
    patterns = [
        r'(?:to|in|into|translate to) (\w+)',
        r'(\w+)\s*(?:translation|translate)',
        r'translate (\w+)',
        r'in (\w+)'
    ]
    
    lang_map = {
        'french': 'fr', 'spanish': 'es', 'german': 'de',
        'italian': 'it', 'portuguese': 'pt', 'dutch': 'nl',
        'russian': 'ru', 'chinese': 'zh', 'japanese': 'ja',
        'korean': 'ko', 'arabic': 'ar', 'turkish': 'tr',
        'hindi': 'hi', 'polish': 'pl', 'swedish': 'sv',
        'danish': 'da', 'finnish': 'fi', 'hebrew': 'he',
        'persian': 'fa', 'thai': 'th', 'vietnamese': 'vi',
        'indonesian': 'id', 'swahili': 'sw', 'amharic': 'am',
        'afrikaans': 'af', 'norwegian': 'no', 'romanian': 'ro',
        'bulgarian': 'bg', 'czech': 'cs', 'greek': 'el',
        'hungarian': 'hu', 'ukrainian': 'uk', 'urdu': 'ur',
        'tamil': 'ta', 'telugu': 'te', 'malay': 'ms'
    }
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            lang_word = match.group(1).lower()
            if lang_word in lang_map:
                cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
                return lang_map[lang_word], cleaned_text
    
    return "en", text

# ============================================================
# BOT HANDLERS
# ============================================================

@dp.message(Command("start"))
async def start_command(message: Message) -> None:
    """Handle /start command."""
    welcome_text = (
        "🌐 *Welcome to PolyglotPulseBot!*\n\n"
        "I translate text between 50+ languages using advanced AI models.\n\n"
        "*📌 How to use:*\n"
        "Send any message with a target language mention.\n\n"
        "*📝 Examples:*\n"
        "• `Hello world in Spanish`\n"
        "• `Translate this to French: Good morning`\n"
        "• `How are you in Russian`\n\n"
        "*⚡ Commands:*\n"
        "/start - Show this message\n"
        "/help - Show help\n"
        "/translate [text] - Translate to English\n"
        "/languages - Show supported languages"
    )
    await message.reply(welcome_text)


@dp.message(Command("help"))
async def help_command(message: Message) -> None:
    """Handle /help command."""
    help_text = (
        "📖 *Help & Commands*\n\n"
        "*Basic Usage:*\n"
        "Send any text with a language mention.\n\n"
        "*Supported Language Codes:*\n"
        "🇬🇧 en (English)  🇫🇷 fr (French)  🇪🇸 es (Spanish)\n"
        "🇩🇪 de (German)  🇮🇹 it (Italian)  🇵🇹 pt (Portuguese)\n"
        "🇷🇺 ru (Russian)  🇨🇳 zh (Chinese)  🇯🇵 ja (Japanese)\n"
        "🇰🇷 ko (Korean)  🇮🇳 hi (Hindi)  🇦🇪 ar (Arabic)\n"
        "🇹🇷 tr (Turkish)  🇮🇱 he (Hebrew)  🇮🇷 fa (Persian)\n"
        "and 30+ more!\n\n"
        "*Tips:*\n"
        "• Use 2-letter ISO 639-1 language codes\n"
        "• The bot auto-detects the source language\n"
        "• Long texts are automatically handled"
    )
    await message.reply(help_text)


@dp.message(Command("languages"))
async def languages_command(message: Message) -> None:
    """Handle /languages command."""
    languages_text = (
        "📋 *Supported Languages*\n\n"
        "*🇪🇺 European:*\n"
        "English (en), Russian (ru), German (de), French (fr)\n"
        "Spanish (es), Italian (it), Portuguese (pt), Dutch (nl)\n"
        "Polish (pl), Swedish (sv), Danish (da), Finnish (fi)\n"
        "Norwegian (no), Romanian (ro), Bulgarian (bg), Czech (cs)\n"
        "Greek (el), Hungarian (hu), Ukrainian (uk)\n\n"
        "*🌏 Asian:*\n"
        "Chinese (zh), Japanese (ja), Korean (ko), Hindi (hi)\n"
        "Thai (th), Vietnamese (vi), Indonesian (id), Malay (ms)\n"
        "Tamil (ta), Telugu (te), Urdu (ur)\n\n"
        "*🌍 Middle Eastern & African:*\n"
        "Arabic (ar), Turkish (tr), Hebrew (he), Persian (fa)\n"
        "Swahili (sw), Amharic (am), Afrikaans (af)\n\n"
        "✨ And many more!"
    )
    await message.reply(languages_text)


@dp.message(Command("translate"))
async def translate_command(message: Message) -> None:
    """Handle /translate command."""
    text = message.text.replace("/translate", "").strip()
    
    if not text:
        await message.reply(
            "✏️ Please provide text to translate:\n\n"
            "Example: `/translate Hello world`"
        )
        return
    
    # Detect source language
    src_lang = language_detector.detect_language(text)
    if not src_lang:
        await message.reply("❌ Could not detect the language of your text.")
        return
    
    target_lang = "en"
    
    # Show processing message
    processing_msg = await message.reply(
        f"🔄 Translating *{src_lang.upper()}* → *{target_lang.upper()}*...\n"
        f"⏳ This may take a few seconds..."
    )
    
    # Translate
    translated = await translator.translate(text, src_lang, target_lang)
    
    if translated:
        await processing_msg.edit_text(
            f"✅ *Translation Complete*\n\n"
            f"📝 *Original ({src_lang.upper()}):*\n"
            f"{text[:500]}\n\n"
            f"🌐 *Translated ({target_lang.upper()}):*\n"
            f"{translated[:500]}"
        )
    else:
        await processing_msg.edit_text(
            "❌ Translation failed.\n\n"
            "This might be because the language pair is not supported.\n"
            "Please try with a different language or shorter text."
        )


@dp.message()
async def handle_any_text(message: Message) -> None:
    """Handle any text message for automatic translation."""
    text = message.text
    if not text or len(text.strip()) < 2:
        return
    
    # Extract target language from the message
    target_lang, cleaned_text = extract_target_language(text)
    
    # Detect source language
    src_lang = language_detector.detect_language(cleaned_text)
    
    if not src_lang:
        await message.reply("❌ Could not detect the language of your text.")
        return
    
    # Don't translate if same language
    if src_lang == target_lang:
        await message.reply(
            f"ℹ️ Your text appears to already be in *{target_lang.upper()}*.\n\n"
            f"To translate, specify a different target language.\n"
            f"Example: `Hello world in French`"
        )
        return
    
    # Show processing message
    processing_msg = await message.reply(
        f"🔄 Detected *{src_lang.upper()}* → Translating to *{target_lang.upper()}*...\n"
        f"⏳ Please wait..."
    )
    
    # Translate
    translated = await translator.translate(cleaned_text, src_lang, target_lang)
    
    if translated:
        await processing_msg.edit_text(
            f"✅ *Translation Complete*\n\n"
            f"📝 *Original ({src_lang.upper()}):*\n"
            f"{cleaned_text[:500]}\n\n"
            f"🌐 *Translated ({target_lang.upper()}):*\n"
            f"{translated[:500]}"
        )
    else:
        await processing_msg.edit_text(
            "❌ Translation failed.\n\n"
            "This language pair might not be supported.\n"
            "Try: `Hello world in Spanish` or use /translate"
        )

# ============================================================
# WEBHOOK FUNCTIONS
# ============================================================

async def set_webhook() -> bool:
    """Set the webhook for the bot."""
    try:
        webhook_url = f"{config.WEBHOOK_HOST}{config.WEBHOOK_PATH}"
        
        await bot.set_webhook(
            url=webhook_url,
            secret_token=config.WEBHOOK_SECRET,
            drop_pending_updates=True,
            allowed_updates=["message", "callback_query"]
        )
        
        logger.info(f"✅ Webhook set to: {webhook_url}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to set webhook: {e}")
        return False

# ============================================================
# FASTAPI APPLICATION
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    logger.info("=" * 60)
    logger.info("🚀 Starting PolyglotPulseBot...")
    logger.info("=" * 60)
    
    await set_webhook()
    
    logger.info(f"✅ Bot is ready!")
    logger.info(f"📡 Webhook URL: {config.WEBHOOK_HOST}{config.WEBHOOK_PATH}")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down PolyglotPulseBot...")


# Create FastAPI app
app = FastAPI(
    title="PolyglotPulseBot",
    description="A sophisticated Telegram translator bot with AI models",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/webhook")
async def webhook_endpoint(request: Request) -> dict:
    """Handle incoming Telegram updates via webhook."""
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
    """Health check endpoint for Railway."""
    return {
        "status": "healthy",
        "bot": "@PolyglotPulseBot",
        "version": "1.0.0"
    }


@app.get("/ready")
async def readiness_check() -> dict:
    """Readiness check endpoint."""
    return {
        "status": "ready",
        "bot": "@PolyglotPulseBot"
    }


@app.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {
        "message": "PolyglotPulseBot is running!",
        "bot": "@PolyglotPulseBot",
        "webhook": f"{config.WEBHOOK_HOST}{config.WEBHOOK_PATH}"
    }

# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on port {config.PORT}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config.PORT,
        reload=False,
        log_level="info"
    )

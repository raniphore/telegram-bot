"""
Entry point for the Telegram RAG + Vision Bot.
"""
import logging
from dotenv import load_dotenv
import os

from telegram.ext import Application, CommandHandler, MessageHandler, filters

from bot.handlers import start, help_command, ask, handle_image, summarize

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("ask", ask))
    app.add_handler(CommandHandler("summarize", summarize))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))

    logger.info("Bot started. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()

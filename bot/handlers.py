"""
Telegram command and message handlers.
"""
import logging
from telegram import Update
from telegram.ext import ContextTypes

from rag.retriever import answer_query
from rag.llm import generate_summary
from vision.captioner import describe_image
from bot.history import get_history, save_interaction

logger = logging.getLogger(__name__)

HELP_TEXT = """
*Hybrid RAG + Vision Bot*

Commands:
/ask <query> — Ask a question from the knowledge base
/summarize — Summarize your last 3 interactions
/help — Show this message

Or simply *send a photo* to get a caption and tags.
"""


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi! I'm a RAG + Vision bot.\n" + HELP_TEXT,
        parse_mode="Markdown",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT, parse_mode="Markdown")


async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = " ".join(context.args)
    if not query:
        await update.message.reply_text("Usage: /ask <your question>")
        return

    user_id = update.effective_user.id
    await update.message.reply_text("Searching knowledge base...")
    try:
        result = await answer_query(query, user_id=user_id)
        await update.message.reply_text(result)
        save_interaction(user_id, "query", query, result)
    except Exception as e:
        logger.error("RAG error: %s", e)
        await update.message.reply_text("Sorry, something went wrong.")


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Processing image...")
    user_id = update.effective_user.id
    try:
        photo = update.message.photo[-1]  # largest available
        file = await context.bot.get_file(photo.file_id)
        image_bytes = await file.download_as_bytearray()

        result = await describe_image(bytes(image_bytes))
        await update.message.reply_text(result)
        save_interaction(user_id, "image", "photo", result)
    except Exception as e:
        logger.error("Vision error: %s", e)
        await update.message.reply_text("Sorry, could not process the image.")


async def summarize(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    history = get_history(user_id, limit=3)

    if not history:
        await update.message.reply_text("No interaction history found yet. Ask me something first!")
        return

    history_text = "\n\n".join(
        f"[{turn['role'].upper()}] User: {turn['content']}\nBot: {turn['response']}"
        for turn in history
    )

    await update.message.reply_text("Summarizing your recent interactions...")
    try:
        summary = await generate_summary(history_text)
        await update.message.reply_text(f"*Summary of your last interactions:*\n\n{summary}", parse_mode="Markdown")
    except Exception as e:
        logger.error("Summary error: %s", e)
        await update.message.reply_text("Sorry, could not generate a summary.")

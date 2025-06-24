import logging
import os
import asyncio
from dotenv import load_dotenv
import torch
from transformers import pipeline
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

pipe = pipeline(
    "text-generation",
    model="tiiuae/falcon-7b-instruct",
    device_map="auto",
    torch_dtype=torch.float16,
    model_kwargs={"temperature": 0.7, "top_p": 0.9, "max_length": 100, "do_sample": True}
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! I'm YapBuddy 🧠. I'm here to support you.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Just share how you're feeling, and I'll try to help.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    prompt = f"You are YapBuddy, an empathetic AI therapist chatbot. A user says: '{user_message}'. Respond in a warm, supportive, and conversational tone."

    try:
        raw = pipe(prompt, return_full_text=False, max_new_tokens=75)[0]['generated_text']
        response = raw.replace(prompt, "").strip()
        reply = response if response.endswith(('.', '!', '?')) else response.split('.')[0] + "."

        if "exam" in user_message.lower():
            reply += " You've prepared for this—trust yourself! 📚✨"
        elif "fight" in user_message.lower():
            reply += " Arguments can be tough, but open communication helps rebuild friendships. 💙"
        elif "sad" in user_message.lower():
            reply += " You're not alone. I'm here for you, and things will get better. 💡"

        await update.message.reply_text(reply)
    except Exception as e:
        logger.error(e)
        await update.message.reply_text("Something went wrong... but I'm still here for you 💙")

async def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    await app.run_polling()

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())

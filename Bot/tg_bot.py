import os
from dotenv import load_dotenv
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes


load_dotenv()
bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

API_ENDPOINT = "http://127.0.0.1:8000/predict"

async def moderate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return

    text = update.message.text
    if not text:
        return

    chat = update.effective_chat
    user = update.effective_user

    try:
        # Send text to your API
        response = requests.post(API_ENDPOINT, json={"text": text})
        prediction = response.json()

        # Handle hate speech
        if prediction.get("is_hate", False):
            await update.message.delete()
            print(f"Deleted hate message: {text}")

            username = f"@{user.username}" if user.username else "the user"
            notice = f"🚫 The message by {username} was deleted due to hate speech."

            # Send notice in chat
            await context.bot.send_message(chat_id=chat.id, text=notice)

        # Handle ironic messages
        elif prediction.get("is_ironic", False):

            admins = await context.bot.get_chat_administrators(chat.id)
            admin_ids = [admin.user.id for admin in admins]

            for admin_id in admin_ids:
                try:
                    await context.bot.send_message(
                        chat_id=admin_id,
                        text=f"⚡ Ironic comment detected in {chat.title}:\n\n{text}"
                    )
                    print(f"Sent irony warning to admin {admin_id}")
                except Exception as e:
                    print(f"Couldn't message admin {admin_id}: {e}")

    except Exception as e:
        print(f"Error during moderation: {e}")

# Initialize Bot
app = ApplicationBuilder().token(bot_token).build()

app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, moderate))

app.run_polling()

import asyncio
from config import config
from web_app import run

def main():
    if not config.ai_slots:
        raise SystemExit("Configura AI_API_KEY en .env.")
    if not config.web_access_token or config.web_access_token == "change-this-token":
        raise SystemExit("Configura un WEB_ACCESS_TOKEN privado en .env.")
    asyncio.run(run())

if __name__ == "__main__":
    main()

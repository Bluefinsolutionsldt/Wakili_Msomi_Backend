import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Existing settings that appear to be used in the codebase
    CLAUDE_API_KEY: str = os.getenv("CLAUDE_API_KEY", "")
    REDIS_URL: str = os.getenv("REDIS_URL", "")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    PORT: int = int(os.getenv("PORT", "8001"))
    CACHE_TTL_HOURS: int = int(os.getenv("CACHE_TTL_HOURS", "24"))
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "http://localhost:8080,http://localhost:3000,http://127.0.0.1:8080,http://0.0.0.0:8080,https://wakilimsomi.vercel.app,https://localhost:3000")
    
    # WhatsApp API settings
    WHATSAPP_API_TOKEN: str = os.getenv("WHATSAPP_API_TOKEN", "")
    WHATSAPP_CLOUD_NUMBER_ID: str = os.getenv("WHATSAPP_CLOUD_NUMBER_ID", "")
    WHATSAPP_VERIFY_TOKEN: str = os.getenv("WHATSAPP_VERIFY_TOKEN", "walubvaluebvksdvbaklsjdbvaiuefpaeufuhkdhaldhva;ohvadna oih0932[]A[J;LKZZIOHV[8OYG[309]A[9VJAOI;]]]")
    META_API_VERSION: str = os.getenv("META_API_VERSION", "v17.0")
    META_APP_ID: str = os.getenv("META_APP_ID", "")
    META_APP_SECRET: str = os.getenv("META_APP_SECRET", "")
    MOCK_WHATSAPP: bool = os.getenv("MOCK_WHATSAPP", "False").lower() == "true"

# Create a singleton instance
settings = Settings()

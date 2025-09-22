
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print("BASE_URL:", os.getenv('BASE_URL'))
print("SERVER_BASE_URL:", os.getenv('SERVER_BASE_URL'))
print("APP_BASE_URL:", os.getenv('APP_BASE_URL'))

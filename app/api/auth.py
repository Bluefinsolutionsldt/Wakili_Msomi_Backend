"""
Authentication utilities for Sheria Kiganjani API
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta
import logging
import os
from dotenv import load_dotenv
from ..redis_db import get_redis
from ..auth_utils import verify_password
from typing import Optional, Dict, Any
import json
from redis.asyncio import Redis

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Constants from environment
SECRET_KEY = os.getenv("JWT_SECRET")
if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET environment variable must be set")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_DAYS = 1

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Get current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise credentials_exception
            
        return {"username": username}
    except JWTError as e:
        logger.error(f"JWT validation error: {str(e)}")
        raise credentials_exception

async def get_current_user_optional(
    token: str = Depends(oauth2_scheme),
    redis: Redis = Depends(get_redis)
) -> Optional[Dict[str, Any]]:
    """Return current user if authenticated, else None"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            return None
        user_data = await redis.get(f"user:{username}")
        if not user_data:
            return None
        return json.loads(user_data)
    except Exception:
        return None

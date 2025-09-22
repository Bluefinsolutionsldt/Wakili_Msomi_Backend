from passlib.context import CryptContext
from typing import Dict, Any
import json
from datetime import datetime
import os
import logging
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hashed version"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate a password hash"""
    return pwd_context.hash(password)

def create_user_data(password: str) -> Dict[str, Any]:
    """Create user data dictionary with hashed password"""
    return {
        "hashed_password": get_password_hash(password),
        "created_at": datetime.now().isoformat(),
        "is_active": True,
        "is_admin": False
    }

async def initialize_free_messages(username: str, redis) -> None:
    """Initialize a new user with 10 free messages"""
    try:
        free_messages_key = f"user:{username}:free_messages"
        await redis.set(free_messages_key, 10)
        logger.info(f"Initialized 10 free messages for user: {username}")
    except Exception as e:
        logger.error(f"Error initializing free messages for {username}: {str(e)}")

async def get_free_messages_remaining(username: str, redis) -> int:
    """Get remaining free messages for a user"""
    try:
        free_messages_key = f"user:{username}:free_messages"
        count = await redis.get(free_messages_key)
        return int(count) if count else 0
    except Exception as e:
        logger.error(f"Error getting free messages for {username}: {str(e)}")
        return 0

async def decrement_free_messages(username: str, redis) -> bool:
    """Decrement free message count for a user"""
    try:
        free_messages_key = f"user:{username}:free_messages"
        current_count = await get_free_messages_remaining(username, redis)
        
        if current_count > 0:
            await redis.set(free_messages_key, current_count - 1)
            logger.info(f"Decremented free messages for {username}. Remaining: {current_count - 1}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error decrementing free messages for {username}: {str(e)}")
        return False

async def check_user_subscription(username: str, redis) -> Dict[str, Any]:
    """
    Check if user has an active subscription
    
    Returns:
        Dict with keys:
        - has_subscription: bool
        - plan: str (if active)
        - expires_at: datetime (if active)
        - days_remaining: int (if active)
    """
    try:
        # Check for active subscription in Redis
        subscription_key = f"user:{username}:subscription"
        subscription_data = await redis.get(subscription_key)
        
        if not subscription_data:
            return {
                "has_subscription": False,
                "plan": None,
                "expires_at": None,
                "days_remaining": 0
            }
        
        # Parse subscription data
        subscription = json.loads(subscription_data)
        expires_at = datetime.fromisoformat(subscription["expires_at"])
        
        # Check if subscription is still valid
        if expires_at > datetime.now():
            days_remaining = (expires_at - datetime.now()).days
            return {
                "has_subscription": True,
                "plan": subscription["plan"],
                "expires_at": expires_at,
                "days_remaining": days_remaining
            }
        else:
            # Subscription expired, remove it
            await redis.delete(subscription_key)
            return {
                "has_subscription": False,
                "plan": None,
                "expires_at": None,
                "days_remaining": 0
            }
            
    except Exception as e:
        logger.error(f"Error checking subscription for {username}: {str(e)}")
        return {
            "has_subscription": False,
            "plan": None,
            "expires_at": None,
            "days_remaining": 0
        } 
from redis.asyncio import Redis
import os
 
async def get_redis() -> Redis:
    """Redis connection pool (adjust URL for production)"""
    return Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True) 
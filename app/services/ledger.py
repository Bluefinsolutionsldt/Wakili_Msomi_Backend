from datetime import datetime
import json
from redis.asyncio import Redis

async def log_payment(
    redis: Redis,
    order_id: str,
    username: str,
    amount: int,
    plan: str,
    status: str,
    gateway_data: dict
):
    """Atomic payment logging with secondary indexes"""
    payment_key = f"payment:{order_id}"
    user_payments_key = f"user:{username}:payments"
    date_key = f"payments:{datetime.now().date().isoformat()}"
    status_key = f"payments:{status}"
    
    payment_record = {
        "order_id": order_id,
        "username": username,
        "amount": amount,
        "currency": "TZS",
        "plan": plan,
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "gateway_data": gateway_data,
        "service": "sheria-kiganjani"
    }

    async with redis.pipeline() as pipe:
        # Primary record
        pipe.set(payment_key, json.dumps(payment_record))
        
        # Secondary indexes
        pipe.lpush(user_payments_key, order_id)
        pipe.lpush(date_key, order_id)
        pipe.lpush(status_key, order_id)
        pipe.set(f"txn:{gateway_data.get('transid')}", order_id)
        
        await pipe.execute()

async def verify_duplicate_payment(redis: Redis, transid: str) -> bool:
    """Check for duplicate transactions"""
    return await redis.exists(f"txn:{transid}") 
from fastapi import APIRouter, Request, HTTPException, Depends
from app.services.whatsapp_service import whatsapp_service
from app.redis_db import get_redis
from redis.asyncio import Redis
import logging
import os
import base64
import json
from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/whatsapp", tags=["whatsapp"])

@router.get("/webhook")
async def verify_webhook(request: Request):
    """Verify WhatsApp webhook"""
    return whatsapp_service.verify(request)

@router.post("/webhook")
async def handle_webhook(request: Request):
    """Handle WhatsApp webhook messages"""
    try:
        body = await request.json()
        logger.info(f"Received WhatsApp webhook:")
        return await whatsapp_service.handle_message(body)
    except Exception as e:
        logger.error(f"Error handling WhatsApp webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test")
async def test_webhook():
    """Test endpoint with sample WhatsApp payload"""
    sample_payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "8856996819413533",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "16505553333",
                                "phone_number_id": "27681414235104944"
                            },
                            "contacts": [
                                {
                                    "profile": {
                                        "name": "Test User"
                                    },
                                    "wa_id": "16315551234"
                                }
                            ],
                            "messages": [
                                {
                                    "from": "16315551234",
                                    "id": "wamid.ABGGFlCGg0cvAgo-sJQh43L5Pe4W",
                                    "timestamp": str(int(__import__("time").time())),  # Current timestamp
                                    "text": {
                                        "body": "Hello this is a test message"
                                    },
                                    "type": "text"
                                }
                            ]
                        },
                        "field": "messages"
                    }
                ]
            }
        ]
    }
    
    try:
        logger.info("Processing test webhook payload")
        return await whatsapp_service.handle_message(sample_payload)
    except Exception as e:
        logger.error(f"Error in test webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/flow-endpoint")
async def handle_flow_endpoint(request: Request):
    """Handle WhatsApp Flow responses for payment collection"""
    try:
        body = await request.json()
        logger.info(f"Received WhatsApp Flow response: {body}")
        
        # Extract flow data
        flow_token = body.get("flow_token", "")
        response_json = body.get("response", {})
        
        # Validate flow token
        if not flow_token.startswith("payment_flow_"):
            logger.error(f"Invalid flow token: {flow_token}")
            return {"status": "error", "message": "Invalid flow token"}
        
        # Extract user phone from flow token
        # Format: payment_flow_{wa_id}_{timestamp}
        token_parts = flow_token.split("_")
        if len(token_parts) >= 3:
            wa_id = token_parts[2]
        else:
            logger.error(f"Could not extract wa_id from flow token: {flow_token}")
            return {"status": "error", "message": "Invalid flow token format"}
        
        # Process flow response
        await whatsapp_service.handle_flow_submission(wa_id, response_json)
        
        return {
            "status": "success",
            "message": "Flow response processed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error handling Flow endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create-subscription-order")
async def create_whatsapp_subscription_order(request: Request, redis: Redis = Depends(get_redis)):
    """Create subscription order for WhatsApp users (no authentication required)"""
    try:
        import os
        import httpx
        from datetime import datetime
        
        body = await request.json()
        logger.info(f"WhatsApp subscription order request: {body}")
        
        plan = body.get("plan")
        phone = body.get("phone")
        
        # Validate inputs
        if not plan or plan not in ["daily", "weekly", "monthly"]:
            raise HTTPException(status_code=400, detail="Invalid plan. Must be daily, weekly, or monthly")
        
        if not phone:
            raise HTTPException(status_code=400, detail="Phone number is required")
        
        # Plan configuration
        plan_config = {
            "daily": {"amount": 1000, "description": "24-hour access to Wakili Msomi"},
            "weekly": {"amount": 5000, "description": "7-day access to Wakili Msomi"},
            "monthly": {"amount": 20000, "description": "30-day access to Wakili Msomi"}
        }
        
        # Prepare order
        order_id = f"wa_{str(phone)[-6:]}_{int(datetime.now().timestamp())}"
        
        # Store order mapping for webhook processing (expires in 2 days)
        order_data = {
            "wa_id": str(phone),  # WhatsApp ID
            "plan": plan,
            "user_type": "whatsapp",  # Flag to identify WhatsApp users
            "amount": plan_config[plan]["amount"],
            "created_at": datetime.now().isoformat()
        }
        order_key = f"order:{order_id}"
        await redis.set(order_key, json.dumps(order_data), ex=2*24*60*60)  # Expire in 2 days
        
        # Prepare payload for Selcom
        payload = {
            "vendor": os.getenv("PAYMENT_VENDOR_ID"),
            "order_id": order_id,
            "buyer_email": f"whatsapp{phone}@wakilimsomi.app",
            "buyer_name": f"WhatsApp User {phone}",
            "buyer_phone": str(phone),
            "amount": plan_config[plan]["amount"],
            "currency": "TZS",
            "buyer_remarks": f"WhatsApp {plan} subscription - {plan_config[plan]['description']}",
            "merchant_remarks": f"Wakili Msomi WhatsApp {plan} subscription",
            "webhook": base64.b64encode(
            f"{os.getenv('BASE_URL')}/payment-webhook".encode()
        ).decode(),
            "no_of_items": 1
        }
        
        # Call Selcom API
        headers = {
            "Authorization": f"Bearer {os.getenv('PAYMENT_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://gateway.jsuite.app/checkout/create-order-minimal",
                json=payload,
                headers=headers,
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"WhatsApp payment order created: {order_id} for WhatsApp user {phone}")
                return {
                    "status": "success",
                    "order_id": order_id,
                    "payment_url": result["data"][0]["payment_gateway_url"],
                    "plan": plan,
                    "amount": plan_config[plan]["amount"]
                }
            else:
                error_detail = response.text
                logger.error(f"Selcom API error: {response.status_code} - {error_detail}")
                raise HTTPException(
                    status_code=502,
                    detail=f"Payment gateway error: {error_detail}"
                )
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating WhatsApp subscription order: {e}")
        raise HTTPException(status_code=500, detail=str(e))

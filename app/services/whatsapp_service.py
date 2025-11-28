import logging
import httpx
import asyncio
from typing import Optional, List, Dict, Any
from fastapi import Request
from fastapi.responses import JSONResponse, PlainTextResponse

from app.core.claude_client import ClaudeClient
from app.config import settings

import os
from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)

class WhatsAppService:

    def __init__(self):
        self.headers = {
            "Content-type": "application/json",
            "Authorization": f"Bearer {settings.WHATSAPP_API_TOKEN}",
        }
        self.url = f"https://graph.facebook.com/{settings.META_API_VERSION}/{settings.WHATSAPP_CLOUD_NUMBER_ID}"
        self.logger = logging.getLogger(__name__)
        self.client = httpx.AsyncClient(base_url=self.url)
        self.claude_client = ClaudeClient()

    async def send_read_receipt(self, message_id: str) -> None:
        """Send a read receipt for a WhatsApp message"""
        if settings.MOCK_WHATSAPP:
            self.logger.info(f"Mock WhatsApp: Sending read receipt for message {message_id}")
            return

        try:
            payload = {
                "messaging_product": "whatsapp",
                "status": "read",
                "message_id": message_id
            }
            response = await self.client.post(
                "/messages",
                json=payload,
                headers=self.headers
            )
            self.logger.info(f"WhatsApp Read Receipt Response: {response.status_code}")
            if response.status_code != 200:
                self.logger.error(f"WhatsApp Read Receipt Error: {response.text}")
        except Exception as e:
            self.logger.error(f"WhatsApp Read Receipt Exception: {e}")

    async def send_message(self, wa_id: str, message: str, options: Optional[List[str]] = None) -> None:
        """Send a message to WhatsApp user"""
        if settings.MOCK_WHATSAPP:
            self.logger.info(f"Mock WhatsApp: Sending to {wa_id}: {message}")
            return

        try:
            payload = self._generate_payload(wa_id, message, options)
            response = await self.client.post(
                "/messages", 
                json=payload, 
                headers=self.headers
            )
            self.logger.info(f"WhatsApp API Response: {response.status_code}")
            if response.status_code != 200:
                self.logger.error(f"WhatsApp API Error: {response.text}")
        except httpx.RequestError as e:
            self.logger.error(f"WhatsApp Request Error: {e}")
        except Exception as e:
            self.logger.error(f"WhatsApp Unexpected Error: {e}")

    async def send_typing_indicator(self, wa_id: str) -> None:
        """Send typing indicator to show bot is processing"""
        if settings.MOCK_WHATSAPP:
            return
            
        try:
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": wa_id,
                "type": "typing",
                "action": "typing_on"
            }
            await self.client.post("/messages", json=payload, headers=self.headers)
        except Exception as e:
            self.logger.error(f"Error sending typing indicator: {e}")

    def verify(self, request: Request) -> JSONResponse | PlainTextResponse:
        """Verify WhatsApp webhook"""
        mode = request.query_params.get("hub.mode")
        token = request.query_params.get("hub.verify_token")
        challenge = request.query_params.get("hub.challenge")

        if not mode or not token:
            self.logger.error("Missing webhook parameters")
            return JSONResponse(
                content={"status": "error", "message": "Missing parameters"},
                status_code=400,
            )

        if mode == "subscribe" and token == settings.WHATSAPP_VERIFY_TOKEN:
            self.logger.info("Webhook verified successfully")
            return PlainTextResponse(content=challenge)

        self.logger.error("Webhook verification failed")
        return JSONResponse(
            content={"status": "error", "message": "Verification failed"},
            status_code=403,
        )

    async def handle_message(self, body: dict) -> JSONResponse:
        """Handle incoming WhatsApp messages"""
        try:
            self.logger.debug(f"Received webhook payload")
            
            if not self._is_valid_message(body):
                self.logger.debug("Invalid message payload, returning OK")
                return JSONResponse(content={"status": "ok"}, status_code=200)

            entry = body.get("entry", [{}])[0]
            changes = entry.get("changes", [{}])[0]
            value = changes.get("value", {})
            
            self.logger.info(f"Processing WhatsApp webhook - field: {changes.get('field')}")
            
            # Handle different message types
            if "messages" in value:
                message = value["messages"][0]
                wa_id = message["from"]
                message_type = message.get("type")
                message_id = message.get("id")
                
                self.logger.info(f"Processing {message_type} message from {wa_id} (ID: {message_id})")
                
                # Extract contact information if available
                contact_name = None
                if "contacts" in value and len(value["contacts"]) > 0:
                    contact_profile = value["contacts"][0].get("profile", {})
                    contact_name = contact_profile.get("name")
                    if contact_name:
                        self.logger.info(f"Contact name: {contact_name}")
                
                # Skip if message is too old (older than 5 minutes)
                if self._is_message_outdated(message):
                    self.logger.warning("Ignoring outdated message")
                    return JSONResponse(content={"status": "ok"}, status_code=200)

                # Send read receipt for the incoming message
                await self.send_read_receipt(message_id)
                
                # Handle different message types
                if message["type"] == "text":
                    user_text = message["text"]["body"]
                    await self._handle_text_message(wa_id, user_text, contact_name)
                elif message["type"] == "interactive":
                    await self._handle_interactive_message(wa_id, message, contact_name)
                elif message["type"] == "document":
                    await self._handle_document_message(wa_id, message, contact_name)
                else:
                    await self.send_message(
                        wa_id, 
                        "I can only process text messages and documents at the moment. Please send your legal question as text."
                    )
                    
            elif "statuses" in value:
                # Print only the status value for each status update
                for status in value["statuses"]:
                    print(status.get('status'))
                
            return JSONResponse(content={"status": "ok"}, status_code=200)
            
        except Exception as e:
            self.logger.error(f"Error handling WhatsApp message: {e}")
            return JSONResponse(content={"status": "error"}, status_code=500)

    async def _handle_text_message(self, wa_id: str, user_text: str, contact_name: Optional[str] = None) -> None:
        """Handle text messages using Claude client"""
        try:
            # Check message limit before processing
            if await self._check_message_limit_exceeded(wa_id):
                await self._send_payment_prompt(wa_id, contact_name)
                return
            
            # Send typing indicator
            await self.send_typing_indicator(wa_id)
            
            # Get conversation ID and username for WhatsApp
            conversation_id = f"whatsapp_{wa_id}"
            username = wa_id

            # Persist WhatsApp message using ConversationStore (7 day expiry)
            try:
                from app.core.conversation_store import Message, MessageRole, MessageType
                from datetime import datetime, timedelta
                # Create Message object
                user_message = Message(
                    role=MessageRole.USER,
                    content=user_text,
                    message_type=MessageType.TEXT,
                    created_at=datetime.now()
                )
                # Add message to conversation (creates if not exists)
                await self.claude_client.conversation_store.add_message(username, conversation_id, user_message)
                # Set TTL for conversation in Redis to 7 days
                if hasattr(self.claude_client.conversation_store, "redis") and self.claude_client.conversation_store.redis:
                    redis_client = self.claude_client.conversation_store.redis
                    key = f"conversation:{username}:{conversation_id}"
                    await redis_client.expire(key, 7 * 24 * 60 * 60)
            except Exception as e:
                self.logger.error(f"Error storing WhatsApp message in ConversationStore: {e}")
            
            # Create a personalized greeting if we have the contact name
            greeting_name = contact_name if contact_name else "there"
            
            # Check for special commands
            if user_text.lower() in ["/start", "hello", "hi", "start"]:
                welcome_message = f"""👋 Habari {greeting_name}! I'm Wakili Msomi, your AI legal assistant.

I specialize in Tanzanian law and can help you with:
📋 Legal questions and advice
📄 Document analysis
⚖️ Understanding your rights
🏛️ Court procedures

How can I assist you today?

💡 Tip: Ask me anything about Tanzanian law or send me a legal document to analyze."""
                if user_text.lower() in ["/mambo", "habari", "inakuaje", "oi"]:
                    welcome_message = f"""👋 Habari {greeting_name}! Mimi ni Wakili Msomi, msaidizi wako wa kisheria wa AI niliyetengenezwa na Bluefin Solutions.

Ninabobea katika sheria za Tanzania na naweza kukusaidia katika mambo yafuatayo:
📋 Maswali na ushauri wa kisheria
📄 Uchambuzi wa nyaraka
⚖️ Kuelewa haki zako
🏛️ Taratibu za mahakamani

Nawezaje kukusaidia leo?

💡 Ushauri: Uliza chochote kuhusu sheria za Tanzania au nitumie nyaraka za kisheria nizichambue.  """
                await self.send_message(wa_id, welcome_message)
                return
            
            elif user_text.lower() in ["/help", "help"]:
                help_message = """🆘 How I can help you:

📝 Ask legal questions in plain language
📄 Send documents for analysis
⚖️ Get information about Tanzanian laws
🏛️ Understand court procedures
💼 Business law guidance

Example questions:
- "What are tenant rights in Tanzania?"
- "How do I register a business?"
- "What is the process for divorce?"

Send me your question or document! 📩"""
                await self.send_message(wa_id, help_message)
                return
            
            # Increment message count for this user
            await self._increment_message_count(wa_id)
            
            # Process with Claude client
            full_response = ""
            response_chunks = []
            
            async for chunk in self.claude_client.process_query(
                query=user_text,
                language="en",  # You can detect language or add language selection
                conversation_id=conversation_id,
                username=wa_id,
                verbose=False
            ):
                if "response_chunk" in chunk:
                    response_chunks.append(chunk["response_chunk"])
                elif "error" in chunk:
                    error_message = "Sorry, I encountered an error processing your request. Please try again later."
                    await self.send_message(wa_id, error_message)
                    return
                elif chunk.get("status") in ["complete", "complete_fallback"]:
                    # Response is complete
                    break
            
            # Combine all response chunks
            full_response = "".join(response_chunks)

            # Replace double asterisks with single asterisks for WhatsApp formatting
            if full_response:
                formatted_response = full_response.replace("**", "*")
                # Split long messages (WhatsApp has a 4096 character limit)
                await self._send_long_message(wa_id, formatted_response)
            else:
                await self.send_message(
                    wa_id,
                    "I apologize, but I couldn't generate a response. Please try rephrasing your question."
                )
                
        except Exception as e:
            self.logger.error(f"Error processing text message: {e}")
            await self.send_message(
                wa_id, 
                "Sorry, I encountered an error. Please try again later."
            )

    async def _handle_interactive_message(self, wa_id: str, message: dict, contact_name: Optional[str] = None) -> None:
        """Handle interactive button responses"""
        try:
            interactive = message.get("interactive", {})
            button_reply = interactive.get("button_reply", {})
            flow_reply = interactive.get("nfm_reply", {})  # Flow reply
            
            if button_reply:
                button_id = button_reply.get("id", "")
                
                # Handle payment button responses
                if button_id.startswith("pay_"):
                    plan = button_id.replace("pay_", "")
                    await self._process_payment_selection(wa_id, plan, contact_name)
                
                elif button_id == "more_help":
                    help_message = """📚 Additional Resources:

🌐 Visit: https://sheriakiganjani.co.tz
📧 Contact: 0621900555
📱 WhatsApp: Continue chatting here!

For complex legal matters, consider consulting with a qualified lawyer."""
                    await self.send_message(wa_id, help_message)
                
                elif button_id == "new_question":
                    await self.send_message(wa_id, "Please go ahead and ask your new legal question! 💭")
            
            elif flow_reply:
                # Handle Flow responses
                response_json = flow_reply.get("response_json", "")
                await self._handle_flow_response(wa_id, response_json, contact_name)
                    
        except Exception as e:
            self.logger.error(f"Error handling interactive message: {e}")

    async def _process_payment_selection(self, wa_id: str, plan: str, contact_name: Optional[str] = None) -> None:
        """Process payment plan selection and trigger USSD push payment"""
        try:
            greeting_name = contact_name if contact_name else "there"
            
            plan_info = {
                "daily": {"price": "TZS 1,000", "duration": "24 hours"},
                "weekly": {"price": "TZS 5,000", "duration": "7 days"},
                "monthly": {"price": "TZS 20,000", "duration": "30 days"}
            }
            
            # Send initial message about payment processing
            await self.send_message(
                wa_id,
                f"""⏳ Processing your payment request {greeting_name}...

📋 *Selected Plan:* {plan.title()}
💰 *Price:* {plan_info[plan]["price"]}
⏰ *Duration:* {plan_info[plan]["duration"]}

You will receive a USSD push notification on your phone shortly. Please enter your PIN to complete the payment."""
            )
            
            # Trigger USSD push payment
            payment_triggered = await self._trigger_ussd_payment(wa_id, plan)
            
            if payment_triggered:
                message = f"""📲 *Payment request sent!*

A USSD prompt has been sent to your phone number (+{wa_id}).

✅ Please check your phone and enter your mobile money PIN to complete the payment.

After successful payment, you'll get unlimited access to Wakili Msomi for {plan_info[plan]["duration"]}!

💡 Didn't receive the prompt? Make sure you have sufficient balance and try again."""
            else:
                message = f"""❌ Sorry {greeting_name}, we're having trouble processing your payment.

Please try again later or contact us at info@bluefinsolutions.co.tz for assistance."""
            
            await self.send_message(wa_id, message)
            
        except Exception as e:
            self.logger.error(f"Error processing payment selection: {e}")

    async def _handle_flow_response(self, wa_id: str, response_json: str, contact_name: Optional[str] = None) -> None:
        """Handle WhatsApp Flow response with payment details"""
        try:
            import json
            flow_data = json.loads(response_json)
            
            # Extract payment details from Flow response
            plan = flow_data.get("plan")
            phone = flow_data.get("phone", wa_id)
            
            if plan:
                # Process payment using USSD push
                await self._process_payment_selection(phone, plan, contact_name)
                
        except Exception as e:
            self.logger.error(f"Error handling Flow response: {e}")

    async def _handle_document_message(self, wa_id: str, message: dict, contact_name: Optional[str] = None) -> None:
        """Handle document uploads for analysis"""
        try:
            greeting_name = contact_name if contact_name else ""
            greeting_text = f" {greeting_name}!" if greeting_name else "!"
            
            await self.send_message(
                wa_id, 
                f"📄 Document received{greeting_text} I'm analyzing it now... This may take a moment."
            )
            
            # For now, send a placeholder response
            # In production, you'd download the document and process it with claude_client.process_legal_document()
            document_response = """📋 Document Analysis Complete!

⚠️ Note: Document processing via WhatsApp is currently limited. For detailed document analysis, please visit our web platform at https://wakilimsomi.vercel.app/

However, you can copy and paste the text content of your document here, and I'll analyze it for you!"""
            
            await self.send_message(wa_id, document_response)
            
        except Exception as e:
            self.logger.error(f"Error handling document: {e}")
            await self.send_message(wa_id, "Sorry, I couldn't process your document. Please try again.")

    async def _send_long_message(self, wa_id: str, message: str) -> None:
        """Split and send long messages"""
        max_length = 4000  # WhatsApp limit is 4096, leave some buffer
        
        if len(message) <= max_length:
            await self.send_message(wa_id, message)
            return
        
        # Split message into chunks
        chunks = []
        current_chunk = ""
        
        sentences = message.split('. ')
        for sentence in sentences:
            if len(current_chunk + sentence + '. ') <= max_length:
                current_chunk += sentence + '. '
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Send chunks with small delays
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk = f"({i+1}/{len(chunks)}) " + chunk
            await self.send_message(wa_id, chunk)
            if i < len(chunks) - 1:  # Don't delay after last chunk
                await asyncio.sleep(1)  # 1 second delay between chunks

    def _generate_payload(self, wa_id: str, message: str, options: Optional[List[str]] = None) -> dict:
        """Generate WhatsApp API payload"""
        if options:
            # Interactive message with buttons
            buttons = []
            for i, option in enumerate(options[:3]):  # WhatsApp supports max 3 buttons
                buttons.append({
                    "type": "reply",
                    "reply": {
                        "id": f"option_{i}",
                        "title": option[:20]  # Button title limit
                    }
                })
            
            return {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": wa_id,
                "type": "interactive",
                "interactive": {
                    "type": "button",
                    "body": {
                        "text": message
                    },
                    "footer": {
                        "text": "Wakili Msomi - Bluefin Solutions"
                    },
                    "action": {
                        "buttons": buttons
                    }
                }
            }
        else:
            # Simple text message
            return {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": wa_id,
                "type": "text",
                "text": {
                    "body": message
                }
            }

    def _is_valid_message(self, body: dict) -> bool:
        """Check if the webhook body contains a valid message"""
        try:
            entry = body.get("entry", [{}])[0]
            changes = entry.get("changes", [{}])[0]
            value = changes.get("value", {})
            return "messages" in value or "statuses" in value
        except (IndexError, KeyError):
            return False

    def _is_message_outdated(self, message: dict) -> bool:
        """Check if message is too old (more than 5 minutes)"""
        try:
            import time
            message_timestamp = int(message.get("timestamp", 0))
            current_timestamp = int(time.time())
            return (current_timestamp - message_timestamp) > 300  # 5 minutes
        except (ValueError, TypeError):
            return False

    async def _check_message_limit_exceeded(self, wa_id: str) -> bool:
        """Check if user has exceeded free message limit"""
        try:
            if hasattr(self.claude_client, "conversation_store") and hasattr(self.claude_client.conversation_store, "redis"):
                redis_client = self.claude_client.conversation_store.redis
                if redis_client:
                    # Get current message count for user
                    count_key = f"whatsapp_message_count:{wa_id}"
                    current_count = await redis_client.get(count_key)
                    current_count = int(current_count) if current_count else 0
                    
                    # Check if user has paid subscription and if it's still valid
                    subscription_key = f"whatsapp_subscription:{wa_id}"
                    subscription_data = await redis_client.get(subscription_key)
                    
                    has_active_subscription = False
                    if subscription_data:
                        try:
                            import json
                            from datetime import datetime
                            subscription = json.loads(subscription_data)
                            expires_at = subscription.get("expires_at")
                            if expires_at:
                                expiry_date = datetime.fromisoformat(expires_at)
                                has_active_subscription = datetime.now() < expiry_date
                                if not has_active_subscription:
                                    # Subscription expired, remove it
                                    await redis_client.delete(subscription_key)
                                    self.logger.info(f"Removed expired WhatsApp subscription for {wa_id}")
                        except (json.JSONDecodeError, ValueError) as e:
                            self.logger.error(f"Error parsing subscription data for {wa_id}: {e}")
                            # If we can't parse the subscription, treat as no subscription
                            has_active_subscription = False
                    
                    # Free limit is 10 messages per day
                    FREE_MESSAGE_LIMIT = 10
                    
                    return current_count >= FREE_MESSAGE_LIMIT and not has_active_subscription
        except Exception as e:
            self.logger.error(f"Error checking message limit: {e}")
        return False

    async def _increment_message_count(self, wa_id: str) -> None:
        """Increment message count for user"""
        try:
            if hasattr(self.claude_client, "conversation_store") and hasattr(self.claude_client.conversation_store, "redis"):
                redis_client = self.claude_client.conversation_store.redis
                if redis_client:
                    count_key = f"whatsapp_message_count:{wa_id}"
                    # Increment count with 24-hour expiry (daily reset)
                    await redis_client.incr(count_key)
                    await redis_client.expire(count_key, 24 * 60 * 60)  # 24 hours
        except Exception as e:
            self.logger.error(f"Error incrementing message count: {e}")

    async def _send_payment_prompt(self, wa_id: str, contact_name: Optional[str] = None) -> None:
        """Send payment prompt with interactive Flow when message limit is exceeded"""
        try:
            greeting_name = contact_name if contact_name else "there"
            
            # Send interactive message with Flow trigger
            message = f"""📱 Habari {greeting_name}! You've reached your daily free message limit (10 messages).

To continue chatting with Wakili Msomi, please select a subscription plan:"""

            # Create interactive message with Flow
            await self._send_payment_flow(wa_id, message)
            
        except Exception as e:
            self.logger.error(f"Error sending payment prompt: {e}")

    async def _send_payment_flow(self, wa_id: str, message: str) -> None:
        """Send WhatsApp Flow for payment collection"""
        try:
            # Create Flow payload
            flow_payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": wa_id,
                "type": "interactive",
                "interactive": {
                    "type": "flow",
                    "header": {
                        "type": "text",
                        "text": "💰 Subscription Payment"
                    },
                    "body": {
                        "text": message
                    },
                    "footer": {
                        "text": "Wakili Msomi - Bluefin Solutions"
                    },
                    "action": {
                        "name": "flow",
                        "parameters": {
                            "flow_message_version": "3",
                            "flow_token": f"payment_flow_{wa_id}_{int(__import__('time').time())}",
                            "flow_id": "1968574023998582",  # Your actual Flow ID
                            "flow_cta": "Select Plan",
                            "flow_action": "navigate",
                            "flow_action_payload": {
                                "screen": "PAYMENT_FORM",
                                "data": {
                                    "user_phone": wa_id,
                                    "plans": [
                                        {"id": "daily", "name": "Daily", "price": "TZS 1000", "duration": "24 hours"},
                                        {"id": "weekly", "name": "Weekly", "price": "TZS 5,000", "duration": "7 days"},
                                        {"id": "monthly", "name": "Monthly", "price": "TZS 20,000", "duration": "30 days"}
                                    ]
                                }
                            }
                        }
                    }
                }
            }

            # Send the Flow
            response = await self.client.post(
                "/messages",
                json=flow_payload,
                headers=self.headers
            )
            
            if response.status_code == 200:
                self.logger.info(f"Payment Flow sent successfully to {wa_id}")
            else:
                self.logger.error(f"Failed to send Payment Flow: {response.text}")
                # Fallback to simple interactive buttons
                await self._send_payment_buttons(wa_id, message)
                
        except Exception as e:
            self.logger.error(f"Error sending payment Flow: {e}")
            # Fallback to simple interactive buttons
            await self._send_payment_buttons(wa_id, message)

    async def _send_payment_buttons(self, wa_id: str, message: str) -> None:
        """Fallback: Send simple interactive buttons for payment"""
        try:
            button_payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": wa_id,
                "type": "interactive",
                "interactive": {
                    "type": "button",
                    "body": {
                        "text": f"""{message}

💰 *Payment Options:*
📅 Daily: TZS 1,000 (24 hours)
📅 Weekly: TZS 5,000 (7 days) 
📅 Monthly: TZS 20,000 (30 days)

Please select your preferred plan:"""
                    },
                    "footer": {
                        "text": "Wakili Msomi - Bluefin Solutions"
                    },
                    "action": {
                        "buttons": [
                            {
                                "type": "reply",
                                "reply": {
                                    "id": "pay_weekly",
                                    "title": "Weekly - 5,000 TZS"
                                }
                            },
                            {
                                "type": "reply",
                                "reply": {
                                    "id": "pay_monthly",
                                    "title": "Monthly - 20,000 TZS"
                                }
                            },
                            {
                                "type": "reply",
                                "reply": {
                                    "id": "pay_daily",
                                    "title": "Daily - 1,000 TZS"
                                }
                            }
                        ]
                    }
                }
            }

            response = await self.client.post(
                "/messages",
                json=button_payload,
                headers=self.headers
            )
            
            if response.status_code == 200:
                self.logger.info(f"Payment buttons sent successfully to {wa_id}")
            else:
                self.logger.error(f"Failed to send payment buttons: {response.text}")
                
        except Exception as e:
            self.logger.error(f"Error sending payment buttons: {e}")

    async def _trigger_ussd_payment(self, wa_id: str, plan: str = "weekly") -> bool:
        """Create subscription order and trigger USSD push payment"""
        try:
            import httpx
            import os
            import json
            import base64
            from datetime import datetime
            
            # Get Redis client for storing order
            redis_client = None
            if hasattr(self.claude_client, "conversation_store") and hasattr(self.claude_client.conversation_store, "redis"):
                redis_client = self.claude_client.conversation_store.redis
            
            # Plan configuration
            plan_config = {
                "daily": {"amount": 1000, "description": "24-hour access to Wakili Msomi"},
                "weekly": {"amount": 5000, "description": "7-day access to Wakili Msomi"},
                "monthly": {"amount": 20000, "description": "30-day access to Wakili Msomi"}
            }
            
            if plan not in plan_config:
                self.logger.error(f"Invalid plan: {plan}")
                return False
            
            # Step 1: Create subscription order directly (no HTTP call to self)
            order_id = f"wa_{str(wa_id)[-6:]}_{int(datetime.now().timestamp())}"
            
            # Store order mapping for webhook processing
            order_data = {
                "wa_id": str(wa_id),
                "plan": plan,
                "user_type": "whatsapp",
                "amount": plan_config[plan]["amount"],
                "created_at": datetime.now().isoformat()
            }
            
            if redis_client:
                order_key = f"order:{order_id}"
                await redis_client.set(order_key, json.dumps(order_data), ex=2*24*60*60)
                self.logger.info(f"Stored order {order_id} in Redis")
            
            # Prepare payload for JSuite create-order
            base_url = os.getenv("BASE_URL", "https://localhost:8007")
            vendor = os.getenv("JSUITE_VENDOR") or os.getenv("PAYMENT_VENDOR_ID") or "TILL604529761"
            
            if not vendor:
                self.logger.error("JSUITE vendor ID not configured")
                return False
            
            create_order_payload = {
                "vendor": vendor,
                "order_id": order_id,
                "buyer_email": f"whatsapp{wa_id}@wakilimsomi.app",
                "buyer_name": f"WhatsApp User {wa_id}",
                "buyer_phone": str(wa_id),
                "amount": plan_config[plan]["amount"],
                "currency": "TZS",
                "buyer_remarks": f"WhatsApp {plan} subscription - {plan_config[plan]['description']}",
                "merchant_remarks": f"Wakili Msomi WhatsApp {plan} subscription",
                "webhook": base64.b64encode(f"{base_url}/payment-webhook".encode()).decode(),
                "no_of_items": 1
            }
            
            async with httpx.AsyncClient() as client:
                # Create the order with JSuite
                self.logger.info(f"Creating JSuite order: {order_id}")
                order_response = await client.post(
                    "https://gateway.jsuite.app/checkout/create-order-minimal",
                    json=create_order_payload,
                    headers={
                        "Content-Type": "application/json"
                    },
                    timeout=30.0
                )
                
                if order_response.status_code != 200:
                    self.logger.error(f"JSuite order creation error: {order_response.status_code} - {order_response.text}")
                    return False
                
                order_result = order_response.json()
                self.logger.info(f"JSuite order created: {order_result}")
                
                # Step 2: Trigger USSD push payment using the order_id
                transid = f"{vendor.replace('TILL', '')}_{order_id[-8:]}"
                
                ussd_payload = {
                    "transid": transid,
                    "order_id": order_id,
                    "msisdn": str(wa_id)
                }
                
                self.logger.info(f"Triggering USSD payment: {ussd_payload}")
                ussd_response = await client.post(
                    "https://gateway.jsuite.app/checkout/wallet-payment",
                    json=ussd_payload,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json"
                    },
                    timeout=30.0
                )
                
                if ussd_response.status_code == 200:
                    ussd_result = ussd_response.json()
                    self.logger.info(f"USSD push payment triggered for {wa_id}: {ussd_result}")
                    return True
                else:
                    self.logger.error(f"USSD payment error: {ussd_response.status_code} - {ussd_response.text}")
                    return False
            
        except Exception as e:
            self.logger.error(f"Error triggering USSD payment: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    async def handle_flow_submission(self, wa_id: str, response_data: dict) -> None:
        """Handle WhatsApp Flow submission for payment"""
        try:
            self.logger.info(f"Processing Flow submission for {wa_id}: {response_data}")
            
            # Extract payment details from flow response
            plan = response_data.get("plan")
            phone = response_data.get("phone", wa_id)
            
            if not plan:
                self.logger.error(f"No plan selected in flow response: {response_data}")
                await self.send_message(wa_id, "❌ Error: No payment plan was selected. Please try again.")
                return
            
            # Validate plan
            valid_plans = ["daily", "weekly", "monthly"]
            if plan not in valid_plans:
                self.logger.error(f"Invalid plan selected: {plan}")
                await self.send_message(wa_id, f"❌ Error: Invalid plan '{plan}'. Please select daily, weekly, or monthly.")
                return
            
            # Process payment using USSD push
            await self._process_payment_selection(phone, plan)
                
        except Exception as e:
            self.logger.error(f"Error handling flow submission: {e}")
            await self.send_message(wa_id, "❌ An error occurred while processing your payment request. Please try again.")

    async def notify_subscription_activated(self, wa_id: str, plan: str, expires_at: str) -> None:
        """Notify WhatsApp user that their subscription has been activated"""
        try:
            from datetime import datetime
            
            # Parse expiry date for display
            expiry_date = datetime.fromisoformat(expires_at)
            formatted_expiry = expiry_date.strftime("%B %d, %Y at %I:%M %p")
            
            plan_info = {
                "daily": {"name": "Daily", "duration": "24 hours"},
                "weekly": {"name": "Weekly", "duration": "7 days"},
                "monthly": {"name": "Monthly", "duration": "30 days"}
            }
            
            message = f"""🎉 *Payment Successful!*

✅ Your {plan_info[plan]["name"]} subscription has been activated!

📅 *Valid until:* {formatted_expiry}
🎯 *Benefits:* Unlimited access to Wakili Msomi
💬 *What's next:* Start asking your legal questions!

Thank you for choosing Wakili Msomi! 🙏

Type your legal question to get started. 💭"""

            await self.send_message(wa_id, message)
            self.logger.info(f"Subscription activation notification sent to {wa_id}")
            
        except Exception as e:
            self.logger.error(f"Error sending subscription notification to {wa_id}: {e}")

# Create singleton instance
whatsapp_service = WhatsAppService()

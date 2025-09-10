import os
import logging
import json
import hashlib
from typing import Dict, Optional, List, Any
import asyncio
from datetime import timedelta, datetime
from redis.exceptions import ConnectionError

# Original imports - ensure these are correctly installed and structured in your project
from anthropic import Anthropic, APIError
from redis import Redis
from dotenv import load_dotenv
from .conversation_store import ConversationStore, Message, MessageRole, MessageType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# --- Shared instances initialized once ---
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
if not CLAUDE_API_KEY:
    raise ValueError("CLAUDE_API_KEY environment variable not set")

# Redis cache
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "24"))
REDIS_URL = os.getenv("REDIS_URL")
redis_cache = None
if REDIS_URL:
    try:
        redis_cache = Redis.from_url(REDIS_URL, decode_responses=True)
        # In a real async app, if using a synchronous Redis client like redis-py,
        # ensure this ping doesn't block the event loop or use an async Redis client.
        redis_cache.ping() 
        logger.info("Redis cache initialized successfully")
    except (ConnectionError, Exception) as e:
        logger.warning(f"Redis cache initialization failed: {str(e)}")
        redis_cache = None
else:
    logger.info("Redis URL not configured, running without cache")

# Claude client
try:
    claude_client_instance = Anthropic(api_key=CLAUDE_API_KEY)
    CLAUDE_MODEL = "claude-sonnet-4-20250514"
    logger.info("Claude client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Claude client: {str(e)}")
    raise

# Offline processor and conversation store
conversation_store = ConversationStore(redis_url=REDIS_URL)

class ClaudeClient:
    def __init__(self):
        self.client = claude_client_instance
        self.model = CLAUDE_MODEL
        self.cache = redis_cache
        self.cache_ttl = CACHE_TTL_HOURS
        self.conversation_store = conversation_store

    async def get_response( # This is the modified method, now an async generator
        self, 
        prompt: str, 
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        language: str = "en",
        verbose: bool = False,
        username: Optional[str] = None
    ):
        """
        Get streaming response with conversation context.
        This method now yields chunks of the response.
        """
        try:
            logger.info(f"Processing streaming request - conversation_id: {conversation_id}, language: {language}")
            
            # Validate jurisdiction focus
            if not self._validate_query_jurisdiction(prompt):
                yield {
                    "response_chunk": "I apologize, but I can only provide information about Tanzanian law. For legal matters in other jurisdictions, please consult appropriate legal professionals in those countries.",
                    "jurisdiction_error": True
                }
                return # Exit the generator

            # Get or create conversation
            if conversation_id:
                logger.info(f"Fetching existing conversation: {conversation_id}")
                conversation = await self.conversation_store.get_conversation(username, conversation_id)
                if not conversation:
                    raise ValueError("Conversation not found")
            else:
                logger.info("Creating new conversation")
                conversation = await self.conversation_store.create_conversation(language=language, username=username)
                conversation_id = conversation.id
            
            # Add user message
            logger.info("Saving user message")
            user_message = Message(
                role=MessageRole.USER,
                content=prompt,
                message_type=MessageType.TEXT,
                created_at=datetime.now()
            )
            await self.conversation_store.add_message(username, conversation_id, user_message)

            # Build context from recent messages
            logger.info("Building conversation context")
            recent_messages = await self.conversation_store.get_recent_messages(username, conversation_id)
            context_messages = []
            
            for msg in recent_messages:
                if msg.role in [MessageRole.USER, MessageRole.ASSISTANT]:
                    context_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })

            # Log the loaded conversation context before sending to AI
            logger.info(f"Loaded conversation context for {conversation_id} (user: {username}):")
            for i, msg in enumerate(context_messages):
                logger.info(f"  [{i}] {msg['role']}: {msg['content']}")

            # Use Claude API with retry and fallback logic
            max_retries = 3
            retry_delay = 2  # seconds
            stream = None
            for attempt in range(max_retries):
                try:
                    logger.info(f"Calling Claude API for streaming (attempt {attempt + 1}/{max_retries})")
                    final_prompt_content = prompt
                    if not verbose:
                        final_prompt_content = f"Provide a direct and concise answer: {prompt}"

                    stream = self.client.messages.create(
                        model=self.model,
                        max_tokens=2000,
                        temperature=0.7,
                        system=system_prompt or self._get_default_system_prompt(language),
                        messages=context_messages + [{"role": "user", "content": final_prompt_content}],
                        stream=True
                    )
                    # If call is successful, break the retry loop
                    break
                except APIError as e:
                    logger.warning(f"Claude API error on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.error("Claude API failed after multiple retries. Attempting fallback to non-streaming.")
                        stream = None # Ensure stream is None to trigger fallback
            
            if stream is None:
                # Fallback to a non-streaming call
                try:
                    logger.info("Attempting non-streaming Claude API call as fallback.")
                    final_prompt_content = prompt
                    if not verbose:
                        final_prompt_content = f"Provide a direct and concise answer: {prompt}"
                    
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=2000,
                        temperature=0.7,
                        system=system_prompt or self._get_default_system_prompt(language),
                        messages=context_messages + [{"role": "user", "content": final_prompt_content}],
                        stream=False
                    )
                    
                    full_response_content = response.content[0].text
                    yield {"response_chunk": self._filter_response_content(full_response_content)}
                    
                    ai_message = Message(
                        role=MessageRole.ASSISTANT,
                        content=self._filter_response_content(full_response_content),
                        message_type=MessageType.TEXT,
                        created_at=datetime.now()
                    )
                    await self.conversation_store.add_message(username, conversation_id, ai_message)
                    
                    yield {
                        "status": "complete_fallback",
                        "conversation_id": conversation_id,
                        "model": self.model,
                        "processed_at": datetime.now().isoformat(),
                        "confidence_score": 0.90 # Slightly lower confidence for fallback
                    }
                    return
                except APIError as e:
                    logger.error(f"Non-streaming fallback also failed: {e}")
                    yield {"error": "API fallback failed", "detail": "The service is currently experiencing issues, and the fallback mechanism also failed. Please try again later."}
                    return

            # The key change for streaming is 'stream=True'. The client may return either
            # an async iterable or a synchronous iterator-like 'Stream' object. Support
            # both by detecting __aiter__ and falling back to iterating the sync iterator
            # inside a thread executor to avoid blocking the event loop.
            full_response_content = ""

            # Helper to iterate a synchronous iterator in an async-friendly way
            async def _aiter_from_sync(sync_iter):
                loop = asyncio.get_event_loop()
                iterator = iter(sync_iter)
                def get_next():
                    try:
                        return next(iterator)
                    except StopIteration:
                        return None
                while True:
                    chunk = await loop.run_in_executor(None, get_next)
                    if chunk is None:
                        break
                    yield chunk

            # Choose iteration strategy depending on the object
            if hasattr(stream, "__aiter__"):
                iterator = stream
            else:
                # Fallback: treat as synchronous iterator
                iterator = _aiter_from_sync(stream)

            async for chunk in iterator:
                if getattr(chunk, "type", None) == "content_block_delta" and getattr(getattr(chunk, "delta", None), "type", None) == "text_delta":
                    text_chunk = chunk.delta.text
                    full_response_content += text_chunk
                    yield {"response_chunk": self._filter_response_content(text_chunk)}
                elif getattr(chunk, "type", None) == "message_stop":
                    # Save the full accumulated response to conversation history
                    ai_message = Message(
                        role=MessageRole.ASSISTANT,
                        content=self._filter_response_content(full_response_content),
                        message_type=MessageType.TEXT,
                        created_at=datetime.now()
                    )
                    await self.conversation_store.add_message(username, conversation_id, ai_message)
                    
                    # Yield a final chunk indicating completion and including metadata
                    yield {
                        "status": "complete", 
                        "conversation_id": conversation_id, 
                        "model": self.model,
                        "processed_at": datetime.now().isoformat(),
                        "confidence_score": 0.95 # Placeholder, adjust as needed
                    }
                # You might handle other chunk types (e.g., tool_use, thinking_delta) here
                # if your client needs to respond to them. For basic text streaming, text_delta is primary.

        except Exception as e:
            logger.error(f"API request error during streaming: {str(e)}")
            yield {"error": str(e), "detail": "An error occurred during response generation."}

    async def process_query( # This is the modified method, now an async generator
        self,
        query: str,
        language: str = "en",
        conversation_id: Optional[str] = None,
        username: Optional[str] = None,
        verbose: bool = False # Added verbose for consistency
    ):
        """
        Process a legal query and yield streaming responses.
        This method now acts as an async generator, relaying chunks from get_response.
        """
        try:
            # The actual streaming logic is in get_response, which this method calls and relays.
            async for chunk in self.get_response(
                prompt=query,
                conversation_id=conversation_id,
                language=language,
                verbose=verbose,
                username=username
            ):
                yield chunk # Yield each chunk as received
            
        except Exception as e:
            logger.error(f"Error relaying streaming query: {str(e)}")
            # If an error prevents the stream from starting or continuing, yield an error chunk
            yield {"error": str(e), "detail": "Failed to process query for streaming."}

    def _get_default_system_prompt(self, language: str = "en") -> str:
        """Get the default system prompt based on language"""
        prompts = {
            "en": """You are a legal AI assistant for Sheria Kiganjani, focused exclusively on Tanzanian law. 

Core instructions:
- Provide direct, concise answers first
- Only elaborate with details if specifically asked
- Always be accurate but prefer brevity
- Use simple language
- Focus only on Tanzanian law
- Cite sources only when explicitly requested

Important:
- Never provide information about non-Tanzanian law
- Maintain professional standards
- Flag if lawyer consultation is needed""",
            
            "sw": """Mimi ni msaidizi wa kisheria wa Sheria Kiganjani, ninayelenga sheria za Tanzania pekee.

Maelekezo muhimu:
- Toa majibu ya moja kwa moja na mafupi kwanza
- Toa maelezo ya kina tu unapoulizwa
- Kuwa sahihi lakini fupi
- Tumia lugha rahisi
- Zungumzia sheria za Tanzania tu
- Taja vyanzo vya kisheria tu unapoulizwa

Muhimu:
- Kamwe usitoe taarifa za sheria za nchi nyingine
- Dumisha viwango vya kitaaluma
- Bainisha kama ushauri wa wakili unahitajika"""
        }
        return prompts.get(language, prompts["en"])

    def _filter_response_content(self, content: str) -> str:
        """Filter response content to maintain identity and jurisdiction focus"""
        content = content.replace("Claude", "Sheria Kiganjani")
        content = content.replace("Anthropic", "Sheria Kiganjani")
        
        countries = ["Kenya", "Uganda", "Rwanda", "Burundi", "South Sudan"]
        if any(country in content for country in countries):
            disclaimer = "\n\nPlease note: I specialize in Tanzanian law only. For legal matters in other countries, please consult legal professionals in those jurisdictions."
            content += disclaimer
        
        return content

    def _validate_query_jurisdiction(self, query: str) -> bool:
        """Validate if query is within Tanzanian jurisdiction"""
        forbidden_terms = [
            "international law", "foreign law", "other countries",
            "Kenya law", "Uganda law", "Rwanda law"
        ]
        
        return not any(term.lower() in query.lower() for term in forbidden_terms)

    async def process_legal_document(
        self,
        document_text: str,
        language: str = "en",
        document_type: Optional[str] = None
    ) -> Dict:
        """Process a legal document using Claude with offline support (non-streaming by default)"""
        try:
            # If online, use Claude API directly (non-streaming for documents)
            logger.info(f"Processing document with Claude API: type={document_type}, language={language}")
            
            system_prompt = f"""
            You are Sheria Kiganjani AI, created by Bluefin Solutions (https://bluefinsolutions.co.tz).
            You are analyzing legal documents in {language}.
            Document type: {document_type or 'unspecified'}
            
            Important identity rules:
            - Always identify yourself as created by Bluefin Solutions
            - When asked about your creator, mention Bluefin Solutions and provide the website: https://bluefinsolutions.co.tz
            - Never mention Anthropic or Claude
            - Maintain professional legal ethics
            
            Provide a comprehensive analysis including:
            - Document type and purpose
            - Key legal points and implications
            - Potential issues or concerns
            - Relevant legal references
            """
            
            messages = [{
                "role": "user",
                "content": document_text
            }]
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.7,
                messages=messages,
                system=system_prompt,
                stream=False # Document processing usually not streamed unless explicitly asked
            )
            
            result = {
                "content": response.content[0].text,
                "model": self.model,
                "usage": {
                    "input_tokens": len(document_text) // 4, # Estimate, real API provides exact
                    "output_tokens": len(response.content[0].text) // 4 # Estimate, real API provides exact
                }
            }
            
            logger.info("Successfully processed document with Claude API")
            return result
            
        except Exception as e:
            logger.error(f"Error processing legal document: {str(e)}")
            raise

    def _get_cache_key(self, text: str, language: str, doc_type: Optional[str] = None) -> str:
        """Generate a consistent cache key."""
        key_string = f"{text}:{language}:{doc_type or ''}"
        return f"claude_cache:{hashlib.md5(key_string.encode()).hexdigest()}"

    def _is_cache_valid(self, cached_item: Dict) -> bool:
        """Check if a cached item is still valid."""
        if not isinstance(cached_item, dict) or "timestamp" not in cached_item:
            return False
        
        timestamp = datetime.fromisoformat(cached_item["timestamp"])
        return (datetime.now() - timestamp) < timedelta(hours=self.cache_ttl)
    
    # --- Offline Model Management ---
    # The following methods are placeholders for a more robust offline model management system.
    # In a real-world scenario, these would interact with a proper database and training pipeline.
    
    async def get_offline_responses_for_training(self, language: str, doc_type: Optional[str] = None) -> List[Dict]:
        """Fetch all cached/stored responses for a given language/type to be used for training."""
        # This is a placeholder. In a real system, you'd query your database of stored interactions.
        logger.warning("get_offline_responses_for_training is a placeholder and not implemented.")
        return []

    async def get_last_training_time(self, language: str, doc_type: Optional[str] = None) -> Optional[datetime]:
        """Get the last time the offline model was trained."""
        # Placeholder. This would read from a database or metadata file.
        logger.warning("get_last_training_time is a placeholder and not implemented.")
        return None

    async def update_last_training_time(self, language: str, doc_type: Optional[str] = None):
        """Update the last training time for the offline model."""
        # Placeholder. This would write to a database or metadata file.
        logger.warning("update_last_training_time is a placeholder and not implemented.")
        pass

    async def force_offline_training(self, language: str, doc_type: Optional[str] = None):
        """Manually trigger the offline model training process."""
        logger.info(f"Forcing offline training for language='{language}', doc_type='{doc_type}'")
        await self.offline_processor.train_offline_model(language, doc_type)
        logger.info("Offline training process completed.")


    async def _train_offline_model_if_needed(
        self, 
        language: str, 
        doc_type: Optional[str] = None
    ):
        """Periodically train the offline model"""
        try:
            last_training = await self.offline_processor.get_last_training_time(language, doc_type)
            current_time = datetime.now()
            
            if not last_training or (current_time - last_training).total_seconds() > 86400:
                logger.info("Training offline model...")
                await self.offline_processor.train_offline_model(language, doc_type)
                await self.offline_processor.update_last_training_time(language, doc_type)
                logger.info("Offline model training completed")
        except Exception as e:
            logger.error(f"Error training offline model: {str(e)}")

    async def process_batch_documents(
        self,
        documents: List[Dict[str, str]],
        train_offline: bool = True
    ) -> List[Dict]:
        """Process multiple documents and use them for offline training"""
        results = []
        for doc in documents:
            result = await self.process_legal_document(
                document_text=doc['text'],
                language=doc.get('language', 'en'),
                document_type=doc.get('type')
            )
            results.append(result)
        
        # Train offline model after batch processing
        if train_offline:
            unique_langs = set(doc.get('language', 'en') for doc in documents)
            unique_types = set(doc.get('type') for doc in documents)
            
            for lang in unique_langs:
                for doc_type in unique_types:
                    await self._train_offline_model_if_needed(lang, doc_type)
        
        return results
    
    async def force_offline_training(
        self,
        language: Optional[str] = None,
        document_type: Optional[str] = None
    ):
        """Force immediate training of the offline model"""
        try:
            if language:
                await self.offline_processor.train_offline_model(language, document_type)
                await self.offline_processor.update_last_training_time(language, document_type)
            else:
                await self.offline_processor.train_all_models()
            logger.info("Forced offline training completed")
        except Exception as e:
            logger.error(f"Error during forced offline training: {str(e)}")
            raise

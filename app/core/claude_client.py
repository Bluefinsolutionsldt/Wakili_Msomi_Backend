"""
Claude API Integration for Sheria Kiganjani
"""
import os
import logging
import json
import hashlib
from typing import Dict, Optional, List
from datetime import timedelta, datetime
from redis.exceptions import ConnectionError

from anthropic import Anthropic
from redis import Redis
from dotenv import load_dotenv
from .offline_processor import OfflineProcessor  
from .conversation_store import ConversationStore, Message, MessageRole, MessageType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class ClaudeClient:
    def __init__(self):
        """Initialize Claude client with optional Redis caching"""
        self.api_key = os.getenv("CLAUDE_API_KEY")
        if not self.api_key:
            raise ValueError("CLAUDE_API_KEY environment variable not set")
            
        # Initialize Redis connection if configured
        self.cache = None
        self.cache_ttl = int(os.getenv("CACHE_TTL_HOURS", "24"))
        
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            try:
                self.cache = Redis.from_url(redis_url, decode_responses=True)
                # Test connection
                self.cache.ping()
                logger.info("Redis cache initialized successfully")
            except (ConnectionError, Exception) as e:
                logger.warning(f"Redis cache initialization failed: {str(e)}")
                self.cache = None
        else:
            logger.info("Redis URL not configured, running without cache")
            
        try:
            self.client = Anthropic(api_key=self.api_key)
            self.model = "claude-sonnet-4-20250514"
            self.current_language = "en"  # Default language
            logger.info("Claude client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Claude client: {str(e)}")
            raise

        # Initialize offline processor
        try:
            self.offline_processor = OfflineProcessor()
            logger.info("Offline processor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing offline processor: {str(e)}")
            raise

        self.conversation_store = ConversationStore(redis_url=os.getenv("REDIS_URL"))

    async def get_response(
        self, 
        prompt: str, 
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        language: str = "en",
        use_offline: bool = False,
        verbose: bool = False,
        username: Optional[str] = None
    ) -> Dict:
        """Get response with conversation context"""
        try:
            logger.info(f"Processing request - conversation_id: {conversation_id}, language: {language}")
            
            # Validate jurisdiction focus
            if not self._validate_query_jurisdiction(prompt):
                logger.info("Query jurisdiction validation failed")
                return {
                    "response": "I apologize, but I can only provide information about Tanzanian law. For legal matters in other jurisdictions, please consult appropriate legal professionals in those countries.",
                    "conversation_id": conversation_id,
                    "model": self.model,
                    "jurisdiction_error": True
                }

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
                message_type=MessageType.TEXT
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

            # Try offline mode if requested
            if use_offline:
                logger.info("Attempting offline response")
                offline_response = await self.offline_processor.get_offline_response(
                    query=prompt,
                    language=language,
                    context_messages=recent_messages
                )
                if offline_response:
                    logger.info("Using offline response")
                    ai_message = Message(
                        role=MessageRole.ASSISTANT,
                        content=offline_response["content"],
                        message_type=MessageType.TEXT
                    )
                    await self.conversation_store.add_message(username, conversation_id, ai_message)
                    return {**offline_response, "conversation_id": conversation_id}

            # Use Claude API with correct message format
            logger.info("Calling Claude API")
            if verbose:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=0.7,
                    system=system_prompt or self._get_default_system_prompt(language),
                    messages=context_messages + [{"role": "user", "content": prompt}]
                )
            else:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=0.7,
                    system=system_prompt or self._get_default_system_prompt(language),
                    messages=context_messages + [{"role": "user", "content": f"Provide a direct and concise answer: {prompt}"}]
                )

            logger.info("Processing Claude API response")
            response_content = self._filter_response_content(response.content[0].text)
            
            logger.info("Saving assistant response")
            ai_message = Message(
                role=MessageRole.ASSISTANT,
                content=response_content,
                message_type=MessageType.TEXT
            )
            await self.conversation_store.add_message(username, conversation_id, ai_message)

            result = {
                "response": response_content,
                "conversation_id": conversation_id,
                "model": self.model,
                "confidence_score": 0.95,  # Default confidence score for Claude responses
                "usage": {
                    "input_tokens": len(prompt) // 4,
                    "output_tokens": len(response_content) // 4
                }
            }

            return result

        except Exception as e:
            logger.error(f"API request error: {str(e)}")
            raise

    async def process_query(
        self,
        query: str,
        language: str = "en",
        conversation_id: Optional[str] = None,
        use_offline: bool = False,
        username: Optional[str] = None
    ) -> str:
        """Process a legal query and return the response"""
        try:
            response = await self.get_response(
                prompt=query,
                conversation_id=conversation_id,
                language=language,
                use_offline=use_offline,
                username=username
            )
            
            if "jurisdiction_error" in response:
                return response["response"]
                
            # Get the response content and filter it
            content = response.get("response", "")
            filtered_content = self._filter_response_content(content)
            
            return filtered_content
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

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
        # Replace company identifiers
        content = content.replace("Claude", "Sheria Kiganjani")
        content = content.replace("Anthropic", "Sheria Kiganjani")
        
        # Add jurisdiction disclaimer if other countries are mentioned
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
        document_type: Optional[str] = None,
        use_offline: bool = False  
    ) -> Dict:
        """Process a legal document using Claude with offline support"""
        try:
            # Only check offline if explicitly requested
            if use_offline:
                logger.info("Attempting to use offline response for document")
                offline_response = await self.offline_processor.get_offline_response(
                    query=document_text,
                    language=language,
                    doc_type=document_type
                )
                if offline_response:
                    logger.info("Using offline response for document")
                    return offline_response
                logger.warning("No offline response available")
                raise ValueError("No offline response available")

            # If online, use Claude API directly
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
                system=system_prompt
            )
            
            result = {
                "content": response.content[0].text,
                "model": self.model,
                "usage": {
                    "input_tokens": len(document_text) // 4,
                    "output_tokens": len(response.content[0].text) // 4
                }
            }
            
            # Store for offline use
            await self.offline_processor.save_response(
                query=document_text,
                response=result,
                language=language,
                doc_type=document_type
            )
            
            logger.info("Successfully processed document with Claude API")
            return result
            
        except Exception as e:
            logger.error(f"Error processing legal document: {str(e)}")
            raise

    async def _train_offline_model_if_needed(
        self, 
        language: str, 
        doc_type: Optional[str] = None
    ):
        """Periodically train the offline model"""
        try:
            # Get the last training time from the database
            last_training = await self.offline_processor.get_last_training_time(language, doc_type)
            current_time = datetime.now()
            
            # Train if never trained or if it's been more than 24 hours
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
        
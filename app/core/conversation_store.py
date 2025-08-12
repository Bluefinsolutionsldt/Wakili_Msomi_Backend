"""
Enhanced conversation store with advanced features for Sheria Kiganjani
"""
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Union
import uuid
import json
from redis.asyncio import Redis
from pydantic import BaseModel, Field
import logging
from enum import Enum
import asyncio
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class MessageType(str, Enum):
    TEXT = "text"
    DOCUMENT = "document"
    REFERENCE = "reference"

class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str
    message_type: MessageType = MessageType.TEXT
    created_at: datetime = Field(default_factory=datetime.now)
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict = Field(default_factory=dict)
    references: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ConversationStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"

class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = Field(default_factory=list)
    language: str
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    metadata: Dict = Field(default_factory=dict)
    status: ConversationStatus = ConversationStatus.ACTIVE
    username: Optional[str] = None  # Changed from user_id to username
    jurisdiction: str = "Tanzania"
    ttl: Optional[int] = None

    class Config:
        use_enum_values = True

    def summarize(self) -> str:
        if len(self.messages) == 0:
            return "Empty conversation"
        latest_messages = self.messages[-3:]
        summary = f"Conversation in {self.language} with {len(self.messages)} messages. "
        summary += f"Latest topic: {latest_messages[-1].content[:100]}..."
        return summary

    def add_tag(self, tag: str):
        if "tags" not in self.metadata:
            self.metadata["tags"] = set()
        self.metadata["tags"].add(tag)

    def remove_tag(self, tag: str):
        if "tags" in self.metadata:
            self.metadata["tags"].discard(tag)

class ConversationStore:
    def __init__(self, redis_url: Optional[str] = None, ttl_days: int = 30):
        self.redis_url = redis_url
        self.redis: Optional[Redis] = None
        self.memory_store: Dict[str, Conversation] = {}
        self.default_ttl = ttl_days * 86400
        self._initialized = False
        
        if not redis_url:
            logger.info("Using in-memory conversation store")

    async def _ensure_initialized(self):
        """Ensure Redis connection is initialized"""
        if self._initialized:
            return
            
        if self.redis_url:
            try:
                self.redis = Redis.from_url(self.redis_url, decode_responses=True)
                # Test connection
                await self.redis.ping()
                logger.info("Redis conversation store initialized successfully")
            except Exception as e:
                logger.warning(f"Redis initialization failed: {str(e)}")
                self.redis = None
        
        self._initialized = True

    async def save_conversation(self, conversation: Conversation) -> None:
        """Save conversation to both Redis and memory store"""
        try:
            await self._ensure_initialized()
            
            self.memory_store[conversation.id] = conversation
            if self.redis:
                key = f"conversation:{conversation.username}:{conversation.id}"  # Include username in key
                await self.redis.set(
                    key,
                    conversation.model_dump_json(),
                    ex=self.default_ttl
                )
                logger.info(f"Saved conversation {conversation.id} to Redis")
        except Exception as e:
            logger.error(f"Error saving conversation: {str(e)}")
            raise

    async def get_conversation(self, username: str, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by username and ID"""
        try:
            await self._ensure_initialized()
            
            # Try memory store first
            if conversation_id in self.memory_store:
                conv = self.memory_store[conversation_id]
                if conv.username == username:
                    return conv
                return None
            
            # Try Redis if available
            if self.redis:
                key = f"conversation:{username}:{conversation_id}"
                data = await self.redis.get(key)
                if data:
                    conv_dict = json.loads(data)
                    conv = Conversation.model_validate(conv_dict)
                    self.memory_store[conversation_id] = conv
                    return conv
            
            # If conversation not found in memory store or Redis, create it
            conversation = Conversation(
                id=conversation_id,
                language="en",
                username=username, # Set username here
                metadata={}
            )
            await self.save_conversation(conversation)
            return conversation
            
        except Exception as e:
            logger.error(f"Error getting conversation: {str(e)}")
            raise

    async def get_conversations_by_user(self, username: str) -> List[Conversation]:
        """Get all conversations for a specific user"""
        await self._ensure_initialized()
        
        try:
            if self.redis:
                # Get all conversation keys for this username from Redis
                pattern = f"conversation:{username}:*"
                keys = await self.redis.keys(pattern)
                conversations = []
                for key in keys:
                    try:
                        data = await self.redis.get(key)
                        if data:
                            conv_data = json.loads(data)
                            conversation = Conversation.model_validate(conv_data)
                            conversations.append(conversation)
                    except Exception as e:
                        logger.error(f"Error loading conversation {key}: {str(e)}")
                        continue
                return conversations
            else:
                # Return conversations from memory store
                return [
                    conv for conv in self.memory_store.values()
                    if conv.username == username
                ]
        except Exception as e:
            logger.error(f"Error getting conversations for user {username}: {str(e)}")
            return []

    async def list_conversations(
        self,
        username: Optional[str] = None,
        status: Optional[ConversationStatus] = None
    ) -> List[Conversation]:
        """List all conversations, optionally filtered by username and status"""
        if username:
            return await self.get_conversations_by_user(username)
            
        await self._ensure_initialized()
        
        try:
            if self.redis:
                # Get all conversation keys from Redis
                pattern = f"conversation:*"
                keys = await self.redis.keys(pattern)
                
                conversations = []
                for key in keys:
                    try:
                        data = await self.redis.get(key)
                        if data:
                            conv_data = json.loads(data)
                            conversation = Conversation.model_validate(conv_data)
                            if status is None or conversation.status == status:
                                conversations.append(conversation)
                    except Exception as e:
                        logger.error(f"Error loading conversation {key}: {str(e)}")
                        continue
                
                return conversations
            else:
                # Return conversations from memory store
                convs = list(self.memory_store.values())
                if status:
                    convs = [c for c in convs if c.status == status]
                return convs
                
        except Exception as e:
            logger.error(f"Error listing conversations: {str(e)}")
            return []

    async def add_message(self, username: str, conversation_id: str, message: Message) -> bool:
        """Add message to conversation and update stores"""
        try:
            conv = await self.get_conversation(username, conversation_id)
            if not conv:
                logger.error(f"Conversation {conversation_id} not found for user {username}")
                return False
            
            # Add message to conversation
            conv.messages.append(message)
            conv.last_updated = datetime.now()
            
            # Save updated conversation
            await self.save_conversation(conv)
            
            return True
        except Exception as e:
            logger.error(f"Error adding message: {str(e)}")
            return False

    async def create_conversation(
        self,
        language: str = "en",
        username: Optional[str] = None,
        ttl: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> Conversation:
        """Create a new conversation"""
        await self._ensure_initialized()
        
        metadata = metadata or {}
        conversation = Conversation(
            language=language,
            metadata=metadata,
            username=username,
            ttl=ttl or self.default_ttl
        )
        
        # Store in Redis or memory
        await self.save_conversation(conversation)
            
        return conversation

    async def get_conversation_summary(self, username: str, conv_id: str) -> Optional[str]:
        try:
            conv = await self.get_conversation(username, conv_id)
            if not conv:
                return None
            return conv.summarize()
        except Exception as e:
            logger.error(f"Error getting conversation summary: {str(e)}")
            return None

    async def get_recent_messages(
        self,
        username: str,
        conv_id: str,
        limit: int = 5,
        message_type: Optional[MessageType] = None
    ) -> List[Message]:
        try:
            conv = await self.get_conversation(username, conv_id)
            if not conv:
                return []
            
            if message_type:
                messages = [m for m in conv.messages if m.message_type == message_type]
            else:
                messages = conv.messages
                
            return messages[-limit:]
        except Exception as e:
            logger.error(f"Error getting recent messages: {str(e)}")
            return []

    async def delete_conversation(self, username: str, conversation_id: str) -> bool:
        """Delete a conversation from both Redis and memory store"""
        try:
            await self._ensure_initialized()
            success = True
            
            # Remove from memory store
            if conversation_id in self.memory_store:
                conv = self.memory_store[conversation_id]
                if conv.username == username:
                    del self.memory_store[conversation_id]
            
            # Remove from Redis if available
            if self.redis:
                key = f"conversation:{username}:{conversation_id}"
                success = bool(await self.redis.delete(key))
            
            return success
        except Exception as e:
            logger.error(f"Error deleting conversation: {str(e)}")
            return False
"""
Offline Processing Module for Sheria Kiganjani
"""
import json
import os
import logging
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import sqlite3
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from .conversation_store import Message
import asyncio

logger = logging.getLogger(__name__)

class OfflineProcessor:
    def __init__(self):
        self.db_path = Path("app/data/offline_cache.db")
        self.models_path = Path("app/data/models")
        self.vectorizer_path = self.models_path / "vectorizer.joblib"
        
        # Create necessary directories
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
        self.vectorizer = self._load_or_create_vectorizer()
        self.conversation_cache = {}

    def _init_db(self):
        """Initialize SQLite database for offline storage"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cached_responses (
                        id TEXT PRIMARY KEY,
                        query TEXT NOT NULL,
                        response TEXT NOT NULL,
                        language TEXT NOT NULL,
                        doc_type TEXT,
                        embeddings BLOB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS offline_rules (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern TEXT NOT NULL,
                        response TEXT NOT NULL,
                        language TEXT NOT NULL,
                        doc_type TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS training_metadata (
                        language TEXT NOT NULL,
                        doc_type TEXT,
                        last_training TIMESTAMP,
                        PRIMARY KEY (language, doc_type)
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS training_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        input_text TEXT NOT NULL,
                        output_text TEXT NOT NULL,
                        language TEXT NOT NULL,
                        doc_type TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
            raise

    def _load_or_create_vectorizer(self):
        """Load existing vectorizer or create a new one"""
        if self.vectorizer_path.exists():
            return joblib.load(str(self.vectorizer_path))
        vectorizer = TfidfVectorizer(max_features=5000)
        joblib.dump(vectorizer, str(self.vectorizer_path))
        return vectorizer

    async def get_offline_response(
        self,
        query: str,
        language: str,
        doc_type: Optional[str] = None,
        context_messages: Optional[List[Message]] = None,
        threshold: float = 0.7
    ) -> Optional[Dict]:
        """Get response from offline storage if available"""
        try:
            # Run database operations in a thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._get_offline_response_sync,
                query, language, doc_type, context_messages, threshold)
            return result
        except Exception as e:
            logger.error(f"Error getting offline response: {str(e)}")
            return None

    def _get_offline_response_sync(
        self,
        query: str,
        language: str,
        doc_type: Optional[str] = None,
        context_messages: Optional[List[Message]] = None,
        threshold: float = 0.7
    ) -> Optional[Dict]:
        """Synchronous version of get_offline_response"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # First try exact pattern matching from rules
                cursor = conn.execute(
                    "SELECT response FROM offline_rules WHERE language = ? AND (doc_type = ? OR doc_type IS NULL)",
                    (language, doc_type)
                )
                for row in cursor:
                    if self._pattern_matches(query, row[0]):
                        return {
                            "content": row[0],
                            "offline": True,
                            "confidence": 1.0
                        }

                # Then try similarity-based matching from cached responses
                try:
                    query_vector = self.vectorizer.transform([query])
                except Exception:
                    # Vectorizer likely not fitted yet
                    return None
                cursor = conn.execute(
                    "SELECT response, embeddings FROM cached_responses WHERE language = ? AND (doc_type = ? OR doc_type IS NULL)",
                    (language, doc_type)
                )
                
                best_match = None
                best_similarity = 0
                
                for row in cursor:
                    response, embeddings = row
                    # If response was stored as JSON, try to extract content
                    try:
                        parsed = json.loads(response)
                        if isinstance(parsed, dict) and "content" in parsed:
                            response_text = parsed["content"]
                        else:
                            response_text = response
                    except Exception:
                        response_text = response

                    if embeddings:
                        embeddings = np.frombuffer(embeddings)
                        similarity = cosine_similarity(query_vector, embeddings.reshape(1, -1))[0][0]
                        if similarity > threshold and similarity > best_similarity:
                            best_match = response_text
                            best_similarity = similarity
                
                if best_match:
                    return {
                        "content": best_match,
                        "offline": True,
                        "confidence": float(best_similarity)
                    }
                
                return None

        except Exception as e:
            logger.error(f"Error in offline response lookup: {str(e)}")
            return None

    def _pattern_matches(self, query: str, pattern: str) -> bool:
        """Check if query matches a pattern"""
        # Implement pattern matching logic here
        # For now, just do simple substring matching
        return pattern.lower() in query.lower()

    async def save_response(
        self,
        query: str,
        response: Dict,
        language: str,
        doc_type: Optional[str] = None,
        context_messages: Optional[List[Message]] = None
    ):
        """Save response to offline storage"""
        try:
            # Run database operations in a thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_response_sync,
                query, response, language, doc_type, context_messages)
        except Exception as e:
            logger.error(f"Error saving response: {str(e)}")

    def _save_response_sync(
        self,
        query: str,
        response: Dict,
        language: str,
        doc_type: Optional[str] = None,
        context_messages: Optional[List[Message]] = None
    ):
        """Synchronous version of save_response"""
        try:
            # Generate embeddings
            try:
                query_vector = self.vectorizer.transform([query])
                embeddings = query_vector.toarray()[0].tobytes()
            except Exception:
                # If not fitted, skip embeddings for now
                embeddings = None
            
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cached_responses 
                    (id, query, response, language, doc_type, embeddings)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self._generate_response_id(query, language, doc_type),
                        query,
                        json.dumps(response),
                        language,
                        doc_type,
                        embeddings
                    )
                )
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving response to database: {str(e)}")

    def _generate_response_id(self, query: str, language: str, doc_type: Optional[str] = None) -> str:
        """Generate unique ID for a response"""
        components = [query, language]
        if doc_type:
            components.append(doc_type)
        return joblib.hash(tuple(components))

    async def get_last_training_time(
        self,
        language: str,
        doc_type: Optional[str] = None
    ) -> Optional[datetime]:
        """Get the last time the offline model was trained"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._get_last_training_time_sync,
                language, doc_type)
            return result
        except Exception as e:
            logger.error(f"Error getting last training time: {str(e)}")
            return None

    def _get_last_training_time_sync(
        self,
        language: str,
        doc_type: Optional[str] = None
    ) -> Optional[datetime]:
        """Synchronous version of get_last_training_time"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(
                    "SELECT last_training FROM training_metadata WHERE language = ? AND doc_type = ?",
                    (language, doc_type)
                )
                row = cursor.fetchone()
                if row and row[0]:
                    return datetime.fromisoformat(row[0])
                return None
        except Exception as e:
            logger.error(f"Error getting last training time from database: {str(e)}")
            return None

    async def update_last_training_time(
        self,
        language: str,
        doc_type: Optional[str] = None
    ):
        """Update the last training time"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._update_last_training_time_sync,
                language, doc_type)
        except Exception as e:
            logger.error(f"Error updating last training time: {str(e)}")

    def _update_last_training_time_sync(
        self,
        language: str,
        doc_type: Optional[str] = None
    ):
        """Synchronous version of update_last_training_time"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO training_metadata 
                    (language, doc_type, last_training)
                    VALUES (?, ?, ?)
                    """,
                    (language, doc_type, datetime.now().isoformat())
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating last training time in database: {str(e)}")

    async def train_offline_model(
        self,
        language: str,
        doc_type: Optional[str] = None
    ):
        """Train or update the offline model"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._train_offline_model_sync,
                language, doc_type)
        except Exception as e:
            logger.error(f"Error training offline model: {str(e)}")

    def _train_offline_model_sync(
        self,
        language: str,
        doc_type: Optional[str] = None
    ):
        """Synchronous version of train_offline_model"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(
                    "SELECT query FROM cached_responses WHERE language = ? AND (doc_type = ? OR doc_type IS NULL)",
                    (language, doc_type)
                )
                queries = [row[0] for row in cursor]
                
                if queries:
                    # Update vectorizer with new data
                    self.vectorizer.fit(queries)
                    joblib.dump(self.vectorizer, str(self.vectorizer_path))
                    
                    # Update embeddings for all cached responses
                    for query in queries:
                        embeddings = self.vectorizer.transform([query]).toarray()[0].tobytes()
                        conn.execute(
                            "UPDATE cached_responses SET embeddings = ? WHERE query = ?",
                            (embeddings, query)
                        )
                    
                    conn.commit()
                    logger.info(f"Trained offline model for language {language} and doc_type {doc_type}")
                
        except Exception as e:
            logger.error(f"Error training offline model: {str(e)}")
            raise

    async def add_training_data(
        self, 
        input_text: str, 
        output_text: str, 
        language: str, 
        doc_type: Optional[str] = None
    ):
        """Add data for offline training"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._add_training_data_sync,
                input_text, output_text, language, doc_type)
        except Exception as e:
            logger.error(f"Error adding training data: {str(e)}")

    def _add_training_data_sync(
        self, 
        input_text: str, 
        output_text: str, 
        language: str, 
        doc_type: Optional[str] = None
    ):
        """Synchronous version of add_training_data"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    """
                    INSERT INTO training_data 
                    (input_text, output_text, language, doc_type) 
                    VALUES (?, ?, ?, ?)
                    """,
                    (input_text, output_text, language, doc_type)
                )
            logger.info("Training data added successfully")
        except Exception as e:
            logger.error(f"Error adding training data: {str(e)}")

    async def clear_old_cache(self, days: int = 30):
        """Clear cached responses older than specified days"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._clear_old_cache_sync,
                days)
        except Exception as e:
            logger.error(f"Error clearing old cache: {str(e)}")

    def _clear_old_cache_sync(self, days: int = 30):
        """Synchronous version of clear_old_cache"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    "DELETE FROM cached_responses WHERE created_at < datetime('now', ?)",
                    (f'-{days} days',)
                )
                logger.info(f"Cleared cached responses older than {days} days")
        except Exception as e:
            logger.error(f"Error clearing old cache: {str(e)}")

    async def train_all_models(self):
        """Train models for all available languages and document types"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._train_all_models_sync)
        except Exception as e:
            logger.error(f"Error training all models: {str(e)}")

    def _train_all_models_sync(self):
        """Synchronous version of train_all_models"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Get unique combinations of language and doc_type
                combinations = conn.execute(
                    """
                    SELECT DISTINCT language, doc_type 
                    FROM training_data 
                    WHERE input_text != 'TRAINING_TIMESTAMP'
                    """
                ).fetchall()
                
                for language, doc_type in combinations:
                    self._train_offline_model_sync(language, doc_type)
                    
        except Exception as e:
            logger.error(f"Error training all models: {str(e)}")

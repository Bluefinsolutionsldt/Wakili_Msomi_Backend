from fastapi import FastAPI, HTTPException, Depends, Request, status, UploadFile, File, Query, Body, Header
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json
from jose import jwt
from dotenv import load_dotenv
from redis.asyncio import Redis
import base64
import httpx

from .auth import get_current_user, get_current_user_optional
from ..core.conversation_store import ConversationStore, Conversation, Message, MessageRole, MessageType
from app.core.claude_client import ClaudeClient
from ..redis_db import get_redis
from ..auth_utils import create_user_data, verify_password, decrement_free_messages
from ..services.ledger import log_payment, verify_duplicate_payment
from .whatsapp import router as whatsapp_router

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Sheria Kiganjani API")

# Import conversation store from claude_client
from ..core.claude_client import conversation_store

# Configure CORS - this must be done before adding any routes
# origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8080,http://localhost:3000,http://127.0.0.1:8080,http://0.0.0.0:8080,https://wakilimsomi.vercel.app,https://localhost:3000").split(",")
origins = ("*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Initialize static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="app/templates")

# Include routers
app.include_router(whatsapp_router)

# Constants
SECRET_KEY = os.getenv("JWT_SECRET")
if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET environment variable must be set")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
PORT = int(os.getenv("PORT", 8001))

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class LoginRequest(BaseModel):
    username: str
    password: str

class CreateConversationRequest(BaseModel):
    """Request model for creating a new conversation"""
    language: str = "sw"
    metadata: Dict = Field(default_factory=dict)

class QueryRequest(BaseModel):
    """Request model for query operations"""
    query: str
    conversation_id: str
    language: Optional[str] = "sw"
    use_offline: Optional[bool] = False  # Field for offline mode
    verbose: Optional[bool] = False      # Field for verbose output

class MessageResponse(BaseModel):
    """Response model for messages"""
    role: str
    content: str
    timestamp: datetime

class ConversationResponse(BaseModel):
    """Response model for conversations"""
    id: str
    created_at: datetime
    language: str
    messages: List[MessageResponse]
    metadata: Dict[str, Any]

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class QueryResponse(BaseModel):
    """Response model for query operations"""
    response: str
    conversation_id: str
    language: str = "sw"  # Default to English
    confidence_score: float = 0.95  # Default confidence score
    processed_at: datetime = Field(default_factory=datetime.now)

WELCOME_MESSAGES = {
    "en": "Hello! I am Sheria Kiganjani, your legal AI assistant. How can I help you today?",
    "sw": "Habari! Mimi ni Sheria Kiganjani, msaidizi wako wa kisheria. Nawezaje kukusaidia leo?"
}


# Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:8080"],  # Add your frontend URL
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Duplicate static/templates block removed below to avoid re-mounting.

class DocumentRequest(BaseModel):
    content: str = Field(
        ...,
        description="Document content (text or base64 encoded file content)",
        example="This agreement is made on...",
        min_length=1,
    )
    language: str = Field(
        default="en",
        description="Document language code (en=English, sw=Swahili)",
        example="en",
        pattern="^(en|sw)$"
    )
    document_type: Optional[str] = Field(
        default=None,
        description="Type of legal document",
        example="contract",
        enum=["contract", "court_filing", "legal_notice", "legislation", "other"]
    )
    is_offline: bool = Field(
        default=False,
        description="Set to true to use offline mode when internet is unavailable",
        example=False
    )

    class Config:
        schema_extra = {
            "example": {
                "content": "This agreement is made between Party A and Party B...",
                "language": "en",
                "document_type": "contract",
                "is_offline": False
            }
        }

class QueryRequest(BaseModel):
    """Request model for query operations"""
    query: str
    conversation_id: str
    language: Optional[str] = "sw"

class MessageResponse(BaseModel):
    """Response model for messages"""
    role: str
    content: str
    timestamp: datetime

class ConversationResponse(BaseModel):
    """Response model for conversations"""
    id: str
    created_at: datetime
    language: str
    messages: List[MessageResponse]
    metadata: Dict[str, Any]

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class QueryResponse(BaseModel):
    """Response model for query operations"""
    response: str
    conversation_id: str
    language: str = "sw"  # Default to English
    confidence_score: float = 0.95  # Default confidence score
    processed_at: datetime = Field(default_factory=datetime.now)

WELCOME_MESSAGES = {
    "en": "Hello! I am Sheria Kiganjani, your legal AI assistant. How can I help you today?",
    "sw": "Habari! Mimi ni Sheria Kiganjani, msaidizi wako wa kisheria. Nawezaje kukusaidia leo?"
}

async def get_claude_client():
    try:
        return ClaudeClient()
    except ValueError as e:
        logger.error(f"Failed to initialize Claude client: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize AI service: {str(e)}"
        )

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve login page to all users"""
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/app")
async def serve_app(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/privacy-policy", response_class=HTMLResponse)
async def privacy_policy(request: Request):
    """Serve Privacy Policy page"""
    return templates.TemplateResponse("privacy-policy.html", {"request": request})

@app.get("/terms-conditions", response_class=HTMLResponse)
async def terms_conditions(request: Request):
    """Serve Terms and Conditions page"""
    return templates.TemplateResponse("terms-conditions.html", {"request": request})

@app.post("/process-document", 
    response_model=dict,
    tags=["Document Processing"],
    summary="Process and analyze legal documents",
    description="""
    Analyze legal documents and extract key information.
    
    **Supported document types:**
    * Contracts
    * Court Filings
    * Legal Notices
    * Legislation
    * General Legal Documents
    
    **Features:**
    * Text extraction
    * Document classification
    * Key information extraction
    * Multi-language support
    """,
    response_description="Analysis results including document type, key points, and metadata"
)
async def process_document(
    request: DocumentRequest,
    claude_client: ClaudeClient = Depends(get_claude_client)
):
    """Process a legal document and return analysis"""
    try:
        logger.info(f"Processing document request: language={request.language}, type={request.document_type}")
        
        if not request.content.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document content cannot be empty"
            )

        # Process document
        result = await claude_client.process_legal_document(
            document_text=request.content,
            language=request.language,
            document_type=request.document_type,
            use_offline=request.is_offline
        )
        
        # Store for offline learning if online
        if not request.is_offline:
            claude_client.offline_processor.add_training_data(
                input_text=request.content,
                output_text=result['content'],
                language=request.language,
                doc_type=request.document_type
            )
            logger.info("Document processed and stored for offline learning")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/register")
async def register_user(
    username: str = Body(..., embed=True),
    password: str = Body(..., embed=True),
    redis = Depends(get_redis)
):
    """Register with any user   name/password combo"""
    if await redis.exists(f"user:{username}"):
        raise HTTPException(status_code=400, detail="Username already exists")
    
    await redis.set(
        f"user:{username}",
        json.dumps(create_user_data(password))
    )
    
    # Initialize free messages for new user
    from ..auth_utils import initialize_free_messages
    await initialize_free_messages(username, redis)
    
    return {"status": "success", "username": username}

@app.post("/token")
async def login_for_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    redis = Depends(get_redis)
):
    """Get JWT token after login"""
    user_data = await redis.get(f"user:{form_data.username}")
    if not user_data:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    user = json.loads(user_data)
    if not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    token = jwt.encode(
        {
            "sub": form_data.username,
            "exp": datetime.utcnow() + timedelta(days=1)
        },
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    return {"access_token": token, "token_type": "bearer"}

@app.get("/users/me")
async def read_current_user(
    current_user: dict = Depends(get_current_user),
    redis = Depends(get_redis)
):
    """Test authenticated endpoint"""
    user_data = await redis.get(f"user:{current_user['username']}")
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
        
    return {"username": current_user["username"], **json.loads(user_data)}

@app.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(current_user: Dict[str, Any] = Depends(get_current_user)):
    """List all conversations for the current user"""
    try:
        # Get conversations for the current user
        conversations = await conversation_store.get_conversations_by_user(current_user["username"])
        
        # Sort conversations by created_at in descending order
        conversations.sort(key=lambda x: x.created_at, reverse=True)
        
        return [
            {
                "id": conv.id,
                "created_at": conv.created_at,
                "language": conv.language,
                "messages": [
                    {
                        "role": conv.messages[1].role,
                        "content": conv.messages[1].content,
                        "timestamp": conv.messages[1].timestamp
                    }
                ] if len(conv.messages) > 1 else [],
                 "metadata": conv.metadata
            }
            for conv in conversations
        ]
    except Exception as e:
        logger.error(f"Error listing conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    current_user: Dict[str, Any] = Depends(get_current_user),
    request: Optional[CreateConversationRequest] = None
):
    """Create a new conversation"""
    try:
        # Create conversation with user information
        metadata = {
            "email": current_user.get("email", "")
        }
        if request and request.metadata:
            metadata.update(request.metadata)
            
        conversation = Conversation(
            language=request.language if request else "sw",
            username=current_user["username"],  # Use username directly
            metadata=metadata
        )
        
        # Add welcome message
        welcome_msg = Message(
            role=MessageRole.ASSISTANT,
            content=WELCOME_MESSAGES[conversation.language],
            message_type=MessageType.TEXT
        )
        conversation.messages.append(welcome_msg)
        
        # Save to store
        await conversation_store.save_conversation(conversation)
        
        return {
            "id": conversation.id,
            "created_at": conversation.created_at,
            "language": conversation.language,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp
                } for msg in conversation.messages
            ],
            "metadata": conversation.metadata
        }
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    current_user: Dict = Depends(get_current_user),
    claude_client: ClaudeClient = Depends(get_claude_client)
):
    """Get conversation details"""
    try:
        # Get conversation
        if current_user:
            conversation = await conversation_store.get_conversation(current_user["username"], conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
        # Check if user has access to this conversation
        if conversation.username != current_user["username"]:
            logger.warning(f"User {current_user['username']} attempted to access conversation {conversation_id} belonging to {conversation.username}")
            raise HTTPException(status_code=403, detail="You do not have access to this conversation")
        
        return {
            "id": conversation.id,
            "created_at": conversation.created_at,
            "language": conversation.language,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp
                } for msg in conversation.messages
            ],
            "metadata": conversation.metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation {conversation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Delete a conversation and its messages"""
    try:
        # Get conversation first to check ownership
        conversation = await conversation_store.get_conversation(current_user["username"], conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
        # Check if user has access to this conversation
        if conversation.username != current_user["username"]:
            logger.warning(f"User {current_user['username']} attempted to delete conversation {conversation_id} belonging to {conversation.username}")
            raise HTTPException(status_code=403, detail="You do not have access to this conversation")
        
        # Delete conversation
        await conversation_store.delete_conversation(current_user["username"], conversation_id)
        return {"status": "success", "message": "Conversation deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add a new endpoint to force offline training
@app.post("/train-offline")
async def train_offline(
    language: Optional[str] = None,
    document_type: Optional[str] = None,
    claude_client: ClaudeClient = Depends(get_claude_client)
):
    """Force training of the offline model"""
    try:
        await claude_client.force_offline_training(language, document_type)
        return {"status": "success", "message": "Offline training completed"}
    except Exception as e:
        logger.error(f"Error during offline training: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Add an endpoint to check offline availability
@app.get("/offline-status")
async def offline_status(
    claude_client: ClaudeClient = Depends(get_claude_client)
):
    """Check offline model status"""
    try:
        with sqlite3.connect(str(claude_client.offline_processor.db_path)) as conn:
            # Get statistics about offline data
            stats = {
                "total_cached_responses": conn.execute(
                    "SELECT COUNT(*) FROM cached_responses"
                ).fetchone()[0],
                "total_training_data": conn.execute(
                    "SELECT COUNT(*) FROM training_data WHERE input_text != 'TRAINING_TIMESTAMP'"
                ).fetchone()[0],
                "languages_available": [
                    row[0] for row in conn.execute(
                        "SELECT DISTINCT language FROM training_data"
                    ).fetchall()
                ],
                "document_types": [
                    row[0] for row in conn.execute(
                        "SELECT DISTINCT doc_type FROM training_data WHERE doc_type IS NOT NULL"
                    ).fetchall()
                ]
            }
        return stats
    except Exception as e:
        logger.error(f"Error checking offline status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "0.1.0"}

# Custom documentation endpoints
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Sheria Kiganjani AI - API Documentation",
        swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui/swagger-ui.css",
        swagger_favicon_url="/static/swagger-ui/favicon.png",
        init_oauth=None,
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_openapi_endpoint():
    return get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        contact=app.contact,
        license_info=app.license_info,
    )

@app.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    language: str = "en",
    document_type: Optional[str] = None,
    is_offline: bool = False
):
    """Handle document file uploads"""
    content = await file.read()
    # Process the file content
    return {"filename": file.filename, "content_type": file.content_type}

# Move validate_query_access here so it is defined before the /query route
async def validate_query_access(
    current_user: Dict[str, Any] = Depends(get_current_user),
    redis: Redis = Depends(get_redis)
) -> Dict[str, Any]:
    """
    Validate that user can access query endpoint
    Checks for active subscription OR free messages remaining
    """
    from ..auth_utils import check_user_subscription, get_free_messages_remaining
    
    username = current_user["username"]
    
    # Check subscription first
    subscription_status = await check_user_subscription(username, redis)
    
    if subscription_status["has_subscription"]:
        return {
            **current_user,
            "subscription": subscription_status,
            "access_type": "subscription"
        }
    
    # If no subscription, check free messages
    free_messages = await get_free_messages_remaining(username, redis)
    
    if free_messages > 0:
        return {
            **current_user,
            "free_messages_remaining": free_messages,
            "access_type": "free_messages"
        }
    
    # No subscription and no free messages
    raise HTTPException(
        status_code=status.HTTP_402_PAYMENT_REQUIRED,
        detail={
            "error": "subscription_required",
            "message": "You have used all your free messages. Purchase a subscription for unlimited access.",
            "free_messages_remaining": 0,
            "available_plans": {
                "daily": {"amount": 7500, "description": "24-hour access"},
                "weekly": {"amount": 20000, "description": "7-day access"},
                "monthly": {"amount": 50000, "description": "30-day access"}
            }
        }
    )

@app.post("/query")
async def process_query(
    request: QueryRequest,
    current_user: Dict[str, Any] = Depends(validate_query_access),
    claude_client: ClaudeClient = Depends(get_claude_client),
    redis: Redis = Depends(get_redis)
):
    """
    Processes a query in the context of a conversation, returning a streaming response
    using Server-Sent Events (SSE).
    """
    if not request.conversation_id:
        raise HTTPException(status_code=400, detail="conversation_id is required")

    async def generate_response_chunks():
        try:
            # Decrement free messages if the user is on a free plan
            if current_user.get("access_type") == "free_messages":
                await decrement_free_messages(current_user["username"], redis)
                logger.info(f"Free message decremented for user: {current_user['username']}")

            # Call claude_client.process_query with explicit parameters
            async for chunk_data in claude_client.process_query(
                query=request.query,
                conversation_id=request.conversation_id,
                language=request.language,
                username=current_user["username"],
                verbose=False
            ):
                # Format as proper SSE
                yield f"data: {json.dumps(chunk_data)}\n\n"

        except AttributeError as e:
            logger.error(f"AttributeError in streaming endpoint: {str(e)}")
            yield f"data: {json.dumps({'error': 'Invalid request format', 'detail': str(e)})}\n\n"
        except Exception as e:
            logger.error(f"Unexpected error in FastAPI streaming endpoint: {str(e)}")
            yield f"data: {json.dumps({'error': str(e), 'detail': 'An unexpected server error occurred during streaming.'})}\n\n"

    return StreamingResponse(
        generate_response_chunks(),
        media_type="text/event-stream",
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


@app.get("/users/{username}/payments")
async def get_payment_history(
    username: str,
    current_user: dict = Depends(get_current_user),
    redis = Depends(get_redis)
):
    """Get payment history for a user"""
    if current_user["username"] != username and not current_user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    order_ids = await redis.lrange(f"user:{username}:payments", 0, -1)
    
    payments = []
    async with redis.pipeline() as pipe:
        for order_id in order_ids:
            pipe.get(f"payment:{order_id}")
        results = await pipe.execute()
    
    for result in results:
        if result:
            payments.append(json.loads(result))
    
    return payments

@app.post(
    "/create-subscription-order",
    status_code=status.HTTP_201_CREATED,
    summary="Initiate subscription payment",
    tags=["Payments"],
    responses={
        201: {"description": "Payment order created successfully"},
        400: {"description": "Another payment is already in progress"},
        502: {"description": "Payment gateway error"}
    }
)
async def create_subscription_order(
    plan: str = Body(..., example="weekly", regex="^(daily|weekly|monthly)$"),
    phone: int = Body(..., description="User's phone number for payment", example=255123456789),
    current_user: dict = Depends(get_current_user),
    redis: Redis = Depends(get_redis)
):
    """
    Creates a payment order with Selcom payment gateway.
    
    - **plan**: Subscription tier (daily, weekly, monthly)
    - **phone**: User's phone number for payment (digits only, no formatting)
    - Returns payment URL for redirection
    """
    # Check for existing pending payment
    # if await redis.exists(f"user:{current_user['username']}:pending_payment"):
    #     raise HTTPException(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         detail="Another payment is already in progress. Please wait or contact support."
    #     )

    # Plan configuration
    plan_config = {
        "daily": {"amount": 1000, "description": "24-hour access"},
        "weekly": {"amount": 5000, "description": "7-day access"},
        "monthly": {"amount": 20000, "description": "30-day access"}
    }

    # Prepare payload for Selcom
    order_id = f"sub_{current_user['username'][:6]}_{int(datetime.now().timestamp())}"
    payload = {
        "vendor": os.getenv("PAYMENT_VENDOR_ID", "TILL60452976"),
        "order_id": order_id,
        "buyer_email": f"{current_user['username']}@sheria-kiganjani.app",
        "buyer_name": current_user["username"],
        "buyer_phone": str(phone),
        "amount": plan_config[plan]["amount"],
        "currency": "TZS",
        "redirect_url": base64.b64encode(
            f"http://localhost:3000/".encode()
        ).decode(),
        "webhook": base64.b64encode(
            f"{os.getenv('BASE_URL')}/payment-webhook".encode()
        ).decode(),
        "buyer_remarks": f"Sheria Kiganjani {plan_config[plan]['description']}",
        "merchant_remarks": f"Subscription for {current_user['username']}, plan: {plan}",
        "no_of_items": 1
    }

    try:
        # Call your payment microservice
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://gateway.jsuite.app/checkout/create-order-minimal",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                json=payload
            )
            
            response_data = response.json()
            
            if response_data.get("resultcode") != "000":
                logger.error(f"Payment gateway error: {response_data}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Payment gateway error: {response_data.get('message')}"
                )
            
            # Store pending payment (expires in 30 mins)
            await redis.setex(
                f"user:{current_user['username']}:pending_payment",
                timedelta(minutes=30),
                value=json.dumps({
                    "order_id": order_id,
                    "plan": plan,
                    "initiated_at": datetime.now().isoformat()
                })
            )
            
            # Store order mapping for webhook processing (expires in 2 days)
            await redis.setex(
                f"order:{order_id}",
                timedelta(days=2),
                value=json.dumps({
                    "username": current_user["username"],
                    "plan": plan
                })
            )
            
            return {
                "payment_url": response_data["data"][0]["payment_gateway_url"],
                "order_id": order_id,
                "amount": plan_config[plan]["amount"],
                "currency": "TZS"
            }
            
    except httpx.TimeoutException:
        logger.error("Payment gateway timeout")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Payment service unavailable"
        )
    except Exception as e:
        logger.error(f"Payment initiation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.get("/auth/status")
async def check_auth_status(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Endpoint to check authentication status"""
    return {"authenticated": True, "username": current_user["username"]}


async def validate_query_access(
    current_user: Dict[str, Any] = Depends(get_current_user),
    redis: Redis = Depends(get_redis)
) -> Dict[str, Any]:
    """
    Validate that user can access query endpoint
    Checks for active subscription OR free messages remaining
    """
    from ..auth_utils import check_user_subscription, get_free_messages_remaining
    
    username = current_user["username"]
    
    # Check subscription first
    subscription_status = await check_user_subscription(username, redis)
    
    if subscription_status["has_subscription"]:
        return {
            **current_user,
            "subscription": subscription_status,
            "access_type": "subscription"
        }
    
    # If no subscription, check free messages
    free_messages = await get_free_messages_remaining(username, redis)
    
    if free_messages > 0:
        return {
            **current_user,
            "free_messages_remaining": free_messages,
            "access_type": "free_messages"
        }
    
    # No subscription and no free messages
    raise HTTPException(
        status_code=status.HTTP_402_PAYMENT_REQUIRED,
        detail={
            "error": "subscription_required",
            "message": "You have used all your free messages. Purchase a subscription for unlimited access.",
            "free_messages_remaining": 0,
            "available_plans": {
                "daily": {"amount": 1000, "description": "24-hour access"},
                "weekly": {"amount": 5000, "description": "7-day access"},
                "monthly": {"amount": 20000, "description": "30-day access"}
            }
        }
    )

@app.get("/free-messages/status")
async def get_free_messages_status(
    current_user: Dict[str, Any] = Depends(get_current_user),
    redis: Redis = Depends(get_redis)
):
    """
    Get current user's free message status and subscription information
    """
    from ..auth_utils import check_user_subscription, get_free_messages_remaining
    
    username = current_user["username"]
    subscription_status = await check_user_subscription(username, redis)
    free_messages = await get_free_messages_remaining(username, redis)
    
    return {
        "username": username,
        "subscription": subscription_status,
        "free_messages_remaining": free_messages,
        "available_plans": {
            "daily": {"amount": 1000, "description": "24-hour access"},
            "weekly": {"amount": 5000, "description": "7-day access"},
            "monthly": {"amount": 20000, "description": "30-day access"}
        }
    }

@app.post("/admin/update-free-messages")
async def update_user_free_messages(
    username: str = Body(..., description="Username to update"),
    free_messages: int = Body(..., description="Number of free messages to set", ge=0),
    current_user: Dict[str, Any] = Depends(get_current_user),
    redis: Redis = Depends(get_redis)
):
    """
    Admin route to manually update free messages for a user
    """
    from ..auth_utils import get_free_messages_remaining
    
    try:
        # Check if target user exists
        user_exists = await redis.exists(f"user:{username}")
        if not user_exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User '{username}' not found"
            )
        
        # Update free messages
        free_messages_key = f"user:{username}:free_messages"
        await redis.set(free_messages_key, free_messages)
        
        logger.info(f"Admin {current_user['username']} updated free messages for {username} to {free_messages}")
        
        return {
            "status": "success",
            "message": f"Updated free messages for user '{username}' to {free_messages}",
            "username": username,
            "free_messages": free_messages,
            "updated_by": current_user["username"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating free messages for {username}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update free messages: {str(e)}"
        )

@app.get("/admin/user-free-messages/{username}")
async def get_user_free_messages(
    username: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    redis: Redis = Depends(get_redis)
):
    """
    Admin route to check free messages for a specific user
    """
    from ..auth_utils import get_free_messages_remaining, check_user_subscription
    
    try:
        # Check if target user exists
        user_exists = await redis.exists(f"user:{username}")
        if not user_exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User '{username}' not found"
            )
        
        # Get free messages and subscription status
        free_messages = await get_free_messages_remaining(username, redis)
        subscription_status = await check_user_subscription(username, redis)
        
        return {
            "username": username,
            "free_messages_remaining": free_messages,
            "subscription": subscription_status,
            "checked_by": current_user["username"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting free messages for {username}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user free messages: {str(e)}"
        )

@app.post("/payment-webhook")
async def payment_webhook(
    payload: dict,
    redis: Redis = Depends(get_redis)
):
    """
    Webhook endpoint to activate user plan after successful payment
    """
    required_fields = [
        "order_id", "result", "resultcode", "payment_status",
        "transid", "reference", "channel", "amount", "phone"
    ]
    # Validate payload
    if not all(field in payload for field in required_fields):
        return {"status": "error", "message": "Missing required fields"}

    # Only process successful payments
    if not (
        payload["result"] == "SUCCESS" and
        payload["resultcode"] == "000" and
        payload["payment_status"].upper() == "COMPLETED"
    ):
        return {"status": "ignored", "message": "Payment not successful"}

    # Duplicate-transaction guard
    if await verify_duplicate_payment(redis, payload["transid"]):
        return {"status": "ignored", "message": "Duplicate transaction"}

    order_id = payload["order_id"]
    # Find user and plan by order_id (assume you stored this mapping at order creation)
    order_key = f"order:{order_id}"
    order_data = await redis.get(order_key)
    if not order_data:
        return {"status": "error", "message": f"Order ID {order_id} not found"}
    order = json.loads(order_data)
    
    # Check if this is a WhatsApp user or web user
    user_type = order.get("user_type", "web")
    plan = order.get("plan")
    
    if user_type == "whatsapp":
        # Handle WhatsApp user subscription
        wa_id = order.get("wa_id")
        if not wa_id or not plan:
            return {"status": "error", "message": "WhatsApp order missing wa_id or plan"}
        
        # Set plan duration for WhatsApp user
        from datetime import datetime, timedelta
        durations = {
            "daily": timedelta(days=1),
            "weekly": timedelta(days=7),
            "monthly": timedelta(days=30)
        }
        expires_at = (datetime.now() + durations[plan]).isoformat()
        
        # Store WhatsApp subscription
        whatsapp_subscription_key = f"whatsapp_subscription:{wa_id}"
        await redis.set(whatsapp_subscription_key, json.dumps({
            "plan": plan,
            "expires_at": expires_at,
            "status": "active"
        }))
        
        # Also reset the message count for this WhatsApp user
        count_key = f"whatsapp_message_count:{wa_id}"
        await redis.delete(count_key)
        
        logger.info(f"Activated WhatsApp subscription for {wa_id}: {plan} plan until {expires_at}")
        
        # Log payment for WhatsApp user
        await log_payment(
            redis=redis,
            order_id=order_id,
            username=f"whatsapp_{wa_id}",
            amount=int(payload.get("amount", 0)),
            plan=plan,
            status="COMPLETED",
            gateway_data=payload
        )
        
        # Send WhatsApp notification to user
        try:
            from app.services.whatsapp_service import whatsapp_service
            await whatsapp_service.notify_subscription_activated(wa_id, plan, expires_at)
        except Exception as e:
            logger.error(f"Failed to send WhatsApp notification to {wa_id}: {e}")
        
        return {"status": "success", "message": f"WhatsApp subscription activated for {wa_id}"}
    
    else:
        # Handle regular web user subscription
        username = order.get("username")
        if not username or not plan:
            return {"status": "error", "message": "Order missing username or plan"}

        # Set plan duration for web user
        from datetime import datetime, timedelta
        durations = {
            "daily": timedelta(days=1),
            "weekly": timedelta(days=7),
            "monthly": timedelta(days=30)
        }
        expires_at = (datetime.now() + durations[plan]).isoformat()
        subscription_key = f"user:{username}:subscription"
        await redis.set(subscription_key, json.dumps({
            "plan": plan,
            "expires_at": expires_at
        }))

        # Log payment and mark transaction id
        await log_payment(
            redis=redis,
            order_id=order_id,
            username=username,
            amount=int(payload.get("amount", 0)),
            plan=plan,
            status="COMPLETED",
            gateway_data=payload
        )

    # Prevent re-use
    await redis.set(f"txn:{payload['transid']}", order_id)
    await redis.delete(order_key)
    await redis.delete(f"user:{username}:pending_payment")

    return {"status": "success", "message": f"Activated {plan} plan for {username}", "username": username, "plan": plan, "expires_at": expires_at}

@app.post("/test-selcom-webhook")
async def test_selcom_webhook(
    username: str = Query(..., description="Username to activate subscription for"),
    plan: str = Query(..., description="Plan type (daily/weekly/monthly)"),
    order_id: str = Query(..., description="Order ID for the test"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    redis: Redis = Depends(get_redis)
):
    """
    Test route to simulate Selcom webhook callback
    """
    try:
        # Validate plan
        valid_plans = ["daily", "weekly", "monthly"]
        if plan not in valid_plans:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid plan. Must be one of: {valid_plans}"
            )
        
        # Check if user exists
        user_exists = await redis.exists(f"user:{username}")
        if not user_exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User '{username}' not found"
            )
        
        # Store order info in Redis (simulating order creation)
        order_data = {
            "username": username,
            "plan": plan,
            "created_at": datetime.now().isoformat(),
            "created_by": current_user["username"]
        }
        order_key = f"order:{order_id}"
        await redis.set(order_key, json.dumps(order_data))
        
        # Simulate Selcom webhook payload
        webhook_payload = {
            "result": "SUCCESS",
            "resultcode": "000",
            "order_id": order_id,
            "transid": f"test_{int(datetime.now().timestamp())}",
            "reference": f"ref_{int(datetime.now().timestamp())}",
            "channel": "TEST_CHANNEL",
            "amount": "10000",
            "phone": "255000000001",
            "payment_status": "COMPLETED"
        }
        
        # Call the webhook endpoint internally
        from fastapi.testclient import TestClient
        from app.api.main import app as fastapi_app
        
        client = TestClient(fastapi_app)
        response = client.post("/payment-webhook", json=webhook_payload)
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Test webhook successful: {result}")
            return {
                "status": "success",
                "message": "Test webhook executed successfully",
                "webhook_response": result,
                "test_data": {
                    "username": username,
                    "plan": plan,
                    "order_id": order_id,
                    "webhook_payload": webhook_payload
                }
            }
        else:
            logger.error(f"Test webhook failed: {response.text}")
            return {
                "status": "error",
                "message": "Test webhook failed",
                "webhook_response": response.text,
                "status_code": response.status_code
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in test webhook: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Test webhook error: {str(e)}"
        )

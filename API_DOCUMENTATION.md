# Sheria Kiganjani AI API Documentation

This document provides detailed information about the available API endpoints in Sheria Kiganjani AI system.

## Base URL
```
http://127.0.0.1:8001/api
```

## Authentication
Currently, the API is open and doesn't require authentication.

## Available Endpoints

### 1. Process Document
**Endpoint:** `POST /process-document`

Process a legal document and get AI analysis.

**Request Body:**
```json
{
    "content": "This agreement is made between Party A and Party B...",
    "language": "en",
    "document_type": "contract",
    "is_offline": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| content | string | Yes | Document content (text) |
| language | string | No | Document language (en/sw). Default: "en" |
| document_type | string | No | Type of document (contract/court_filing/legal_notice/legislation/other) |
| is_offline | boolean | No | Whether to use offline mode. Default: false |

**Successful Response:**
```json
{
    "analysis": {
        "summary": "string",
        "key_points": ["string"],
        "recommendations": ["string"]
    },
    "metadata": {
        "document_type": "string",
        "language": "string",
        "processed_at": "datetime"
    }
}
```

### 2. Upload Document
**Endpoint:** `POST /upload-document`

Upload a document file for processing.

**Request Body:**
- Form data with following fields:
  - `file`: Document file (PDF, DOCX, TXT)
  - `language`: Document language (en/sw)
  - `document_type`: Type of document
  - `is_offline`: Boolean for offline processing

**Successful Response:** Same as process-document endpoint

### 3. Create Conversation
**Endpoint:** `POST /conversations`

Start a new conversation session.

**Request Body:**
```json
{
    "language": "en",
    "metadata": {}
}
```

**Successful Response:**
```json
{
    "id": "string",
    "created_at": "datetime",
    "language": "string",
    "messages": [],
    "metadata": {}
}
```

### 4. Send Query
**Endpoint:** `POST /query`

Send a legal query to the AI.

**Request Body:**
```json
{
    "query": "What are the requirements for company registration?",
    "language": "en",
    "conversation_id": "string",
    "is_offline": false
}
```

**Successful Response:**
```json
{
    "response": "string",
    "conversation_id": "string",
    "language": "string",
    "confidence_score": 0.95,
    "processed_at": "datetime"
}
```

### 5. Get Conversation
**Endpoint:** `GET /conversations/{conversation_id}`

Retrieve details of a specific conversation.

**Path Parameters:**
- `conversation_id`: ID of the conversation

**Successful Response:**
```json
{
    "id": "string",
    "created_at": "datetime",
    "language": "string",
    "messages": [
        {
            "id": "string",
            "role": "string",
            "content": "string",
            "message_type": "text",
            "timestamp": "datetime",
            "metadata": {},
            "references": []
        }
    ],
    "metadata": {}
}
```

### 6. List Conversations
**Endpoint:** `GET /conversations`

List all conversations with pagination.

**Query Parameters:**
- `limit`: Number of conversations to return (1-100)
- `user_id`: Filter by user ID (optional)
- `days`: Number of days to look back (1-365)
- `sort_by`: Sort field (created_at/last_updated)
- `sort_order`: Sort direction (asc/desc)

**Successful Response:**
```json
{
    "conversations": [
        {
            "id": "string",
            "created_at": "datetime",
            "language": "string",
            "messages": [],
            "metadata": {}
        }
    ],
    "total": 0,
    "page": 1
}
```

### 7. Delete Conversation
**Endpoint:** `DELETE /conversations/{conversation_id}`

Delete a specific conversation.

**Path Parameters:**
- `conversation_id`: ID of the conversation to delete

**Successful Response:**
```json
{
    "status": "success",
    "message": "Conversation deleted successfully"
}
```

### 8. Health Check
**Endpoint:** `GET /health`

Check if the API is running.

**Successful Response:**
```json
{
    "status": "healthy",
    "timestamp": "datetime"
}
```

### 9. Offline Status
**Endpoint:** `GET /offline-status`

Check the status of offline models.

**Successful Response:**
```json
{
    "offline_available": true,
    "supported_languages": ["en", "sw"],
    "supported_document_types": ["contract", "court_filing"],
    "last_updated": "datetime"
}
```

## Error Responses
All endpoints may return the following error responses:

```json
{
    "detail": "Error message description",
    "status_code": 400,
    "error_type": "BadRequest"
}
```

Common HTTP status codes:
- 400: Bad Request
- 404: Not Found
- 422: Validation Error
- 500: Internal Server Error

## Rate Limiting
The API currently does not implement rate limiting, but it's recommended to keep requests within reasonable limits.

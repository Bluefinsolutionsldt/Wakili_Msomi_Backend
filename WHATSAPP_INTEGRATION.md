# WhatsApp Integration for Wakili Msomi

This document describes the WhatsApp integration for the Wakili Msomi legal assistant backend.

## Overview

The WhatsApp integration allows users to interact with Wakili Msomi through WhatsApp messages. The system handles:

- Text messages with legal questions
- Interactive button responses
- Document uploads (with limitations)
- Typing indicators
- Message chunking for long responses

## Files Added

### 1. `app/config.py`
Central configuration file that includes WhatsApp API settings along with existing configuration.

### 2. `app/services/whatsapp_service.py`
Main WhatsApp service class that handles:
- Message sending and receiving
- Webhook verification
- Message processing with Claude AI
- Interactive message handling
- Document processing (placeholder)

### 3. `app/api/whatsapp.py`
FastAPI router for WhatsApp webhook endpoints:
- `GET /whatsapp/webhook` - Webhook verification
- `POST /whatsapp/webhook` - Message handling

### 4. `app/services/__init__.py`
Makes the services directory a proper Python package.

## Environment Variables

Add these to your `.env` file:

```bash
# WhatsApp API Configuration
WHATSAPP_API_TOKEN=your_whatsapp_api_token_here
WHATSAPP_CLOUD_NUMBER_ID=your_whatsapp_cloud_number_id_here
WHATSAPP_VERIFY_TOKEN=your_whatsapp_verify_token_here
META_API_VERSION=v17.0
META_APP_ID=your_meta_app_id_here
META_APP_SECRET=your_meta_app_secret_here
MOCK_WHATSAPP=False
```

## Webhook Payload Structure

The WhatsApp Business API sends webhooks in the following format:

```json
{
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
                                    "name": "Kerry Fisher"
                                },
                                "wa_id": "16315551234"
                            }
                        ],
                        "messages": [
                            {
                                "from": "16315551234",
                                "id": "wamid.ABGGFlCGg0cvAgo-sJQh43L5Pe4W",
                                "timestamp": "1603059201",
                                "text": {
                                    "body": "Hello this is an answer"
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
```

### Payload Processing

The service extracts the following information:
- **Contact Name**: From `entry[0].changes[0].value.contacts[0].profile.name`
- **WhatsApp ID**: From `entry[0].changes[0].value.messages[0].from`
- **Message Text**: From `entry[0].changes[0].value.messages[0].text.body`
- **Message Type**: From `entry[0].changes[0].value.messages[0].type`
- **Timestamp**: From `entry[0].changes[0].value.messages[0].timestamp`

## Setup Instructions

### 1. WhatsApp Business API Setup

1. Create a Meta for Developers account
2. Create a new app
3. Add WhatsApp product to your app
4. Get your:
   - Access Token (`WHATSAPP_API_TOKEN`)
   - Phone Number ID (`WHATSAPP_CLOUD_NUMBER_ID`)
   - App ID (`META_APP_ID`)
   - App Secret (`META_APP_SECRET`)

### 2. Webhook Configuration

1. Set your webhook URL to: `https://yourdomain.com/whatsapp/webhook`
2. Set a verify token (`WHATSAPP_VERIFY_TOKEN`) - can be any string
3. Subscribe to `messages` and `message_status` webhook fields

### 3. Environment Configuration

1. Copy the environment variables above to your `.env` file
2. Replace placeholder values with your actual API credentials
3. Set `MOCK_WHATSAPP=True` for testing without actual WhatsApp API calls

## Features

### Message Handling

The service handles different types of WhatsApp messages:

- **Text Messages**: Processed using Claude AI for legal assistance
- **Interactive Messages**: Button responses for additional help
- **Documents**: Currently shows a placeholder message (can be extended)

### Special Commands

- `/start`, `hello`, `hi`, `start`: Shows welcome message
- `/help`, `help`: Shows help information

### Smart Features

- **Typing Indicators**: Shows user that bot is processing
- **Message Chunking**: Long responses are split into multiple messages
- **Outdated Message Filtering**: Ignores messages older than 5 minutes
- **Conversation Tracking**: Maintains conversation history per user

### Response Examples

**Welcome Message:**
```
👋 Habari! I'm Wakili Msomi, your AI legal assistant created by Bluefin Solutions.

I specialize in Tanzanian law and can help you with:
📋 Legal questions and advice
📄 Document analysis
⚖️ Understanding your rights
🏛️ Court procedures

How can I assist you today?

💡 Tip: Ask me anything about Tanzanian law or send me a legal document to analyze.
```

**Help Message:**
```
🆘 How I can help you:

📝 Ask legal questions in plain language
📄 Send documents for analysis
⚖️ Get information about Tanzanian laws
🏛️ Understand court procedures
💼 Business law guidance

Example questions:
- "What are tenant rights in Tanzania?"
- "How do I register a business?"
- "What is the process for divorce?"

Send me your question or document! 📩
```

## Integration with Existing System

The WhatsApp service integrates seamlessly with your existing Claude AI system:

- Uses the same `ClaudeClient` for processing legal queries
- Maintains conversation history using conversation IDs
- Supports the same language processing capabilities
- Uses WhatsApp phone numbers as user identifiers

## Testing

### Mock Mode

Set `MOCK_WHATSAPP=True` to test without making actual API calls. Messages will be logged instead of sent.

### Test Endpoint

Use the test endpoint to verify webhook processing:

```bash
POST /whatsapp/test
```

This endpoint processes a sample payload and shows how your webhook would handle real messages.

### Local Testing

1. Use ngrok or similar to expose your local server:
   ```bash
   ngrok http 8001
   ```

2. Configure WhatsApp webhook to point to your tunnel URL:
   ```
   https://your-ngrok-url.ngrok.io/whatsapp/webhook
   ```

3. Send test messages to your WhatsApp Business number

### Webhook Verification

Test webhook verification with:
```bash
GET /whatsapp/webhook?hub.mode=subscribe&hub.verify_token=YOUR_VERIFY_TOKEN&hub.challenge=test_challenge
```

Should return the challenge value if verification is successful.

## Error Handling

The service includes comprehensive error handling:

- Network request failures
- API rate limiting
- Invalid message formats
- Claude AI processing errors
- Webhook verification failures

## Rate Limiting

Consider implementing rate limiting to prevent abuse:

- Limit messages per user per minute
- Implement conversation timeouts
- Monitor API usage

## Security Considerations

- Webhook verification prevents unauthorized access
- User phone numbers are treated as sensitive data
- All communications are logged for monitoring
- Consider implementing user authentication for premium features

## Extension Points

The integration can be extended to support:

- Document download and processing
- Voice message transcription
- Image analysis
- User authentication and premium features
- Multi-language support
- Analytics and reporting

// Sheria Kiganjani API Usage Example
import fetch from 'node-fetch';

const API_BASE_URL = 'http://localhost:8001';

// API Client Class
class SheriaKiganjaniAPI {
    constructor() {
        this.token = null;
        this.currentConversationId = null;
    }

    // Authentication
    async login(userId, email) {
        try {
            const formData = new URLSearchParams();
            formData.append('username', userId);
            formData.append('password', 'any_password');  // Simplified for testing
            
            const response = await fetch(`${API_BASE_URL}/token`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'Accept': 'application/json'
                },
                credentials: 'include',
                mode: 'cors',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Login failed');
            }

            const data = await response.json();
            this.token = data.access_token;
            console.log('Login successful:', data.user);
            return this.token;
        } catch (error) {
            console.error('Login failed:', error);
            throw error;
        }
    }

    // Helper for authenticated requests
    async authenticatedRequest(endpoint, method = 'GET', body = null) {
        if (!this.token) {
            throw new Error('Not authenticated. Please login first.');
        }

        const options = {
            method,
            headers: {
                'Authorization': `Bearer ${this.token}`,
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            credentials: 'include',
            mode: 'cors'
        };

        if (body) {
            options.body = JSON.stringify(body);
        }

        const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Request failed');
        }
        return response.json();
    }

    // Create a new conversation
    async createConversation(language = 'en') {
        const conversation = await this.authenticatedRequest('/conversations', 'POST', {
            language,
            metadata: {}
        });
        this.currentConversationId = conversation.id;
        return conversation;
    }

    // List all conversations
    async listConversations() {
        return this.authenticatedRequest('/conversations');
    }

    // Ask a legal question
    async askQuestion(query, conversationId = null) {
        return this.authenticatedRequest('/query', 'POST', {
            query,
            conversation_id: conversationId || this.currentConversationId,
            language: 'en'
        });
    }

    // Delete a conversation
    async deleteConversation(conversationId) {
        return this.authenticatedRequest(`/conversations/${conversationId}`, 'DELETE');
    }
}

// Example Usage
async function demonstrateAPIUsage() {
    try {
        console.log('Starting API demonstration...\n');

        // Initialize API client
        const api = new SheriaKiganjaniAPI();
        
        // Step 1: Login
        console.log('Step 1: Logging in...');
        await api.login('edward00', 'eddyelly24@gmail.com');
        console.log('✓ Login successful\n');

        // Step 2: Create a new conversation
        console.log('Step 2: Creating a new conversation...');
        const conversation = await api.createConversation('en');
        console.log('✓ Created conversation:', conversation.id);
        console.log('Conversation details:', conversation, '\n');

        // Step 3: List all conversations
        console.log('Step 3: Listing all conversations...');
        const conversations = await api.listConversations();
        console.log('✓ Found', conversations.length, 'conversations');
        console.log('Conversations:', conversations, '\n');

        // Step 4: Ask a legal question
        console.log('Step 4: Asking a legal question...');
        const query = 'What are the requirements for registering a business in Tanzania?';
        const response = await api.askQuestion(query, conversation.id);
        console.log('✓ Received response:', response);
        console.log('Response details:', response, '\n');

        // Step 5: Ask another question in the same conversation
        console.log('Step 5: Asking a follow-up question...');
        const followUpQuery = 'What are the tax implications for this business type?';
        const followUpResponse = await api.askQuestion(followUpQuery, conversation.id);
        console.log('✓ Received response:', followUpResponse);
        console.log('Response details:', followUpResponse, '\n');

        // Step 6: List conversations again to see the updates
        console.log('Step 6: Listing updated conversations...');
        const updatedConversations = await api.listConversations();
        console.log('✓ Found', updatedConversations.length, 'conversations');
        console.log('Updated conversations:', updatedConversations, '\n');

        // Step 7: Delete the conversation
        console.log('Step 7: Deleting the conversation...');
        await api.deleteConversation(conversation.id);
        console.log('✓ Conversation deleted successfully\n');

        // Step 8: Verify deletion
        console.log('Step 8: Verifying deletion...');
        const finalConversations = await api.listConversations();
        console.log('✓ Found', finalConversations.length, 'conversations after deletion');
        console.log('Final conversations:', finalConversations, '\n');

        console.log('API demonstration completed successfully!');
    } catch (error) {
        console.error('API demonstration failed:', error);
    }
}

// Run the demonstration
demonstrateAPIUsage();

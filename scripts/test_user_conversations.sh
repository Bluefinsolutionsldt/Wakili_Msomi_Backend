#!/bin/bash

# Base URL for the API
API_URL="http://localhost:8001"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to get token for a user
get_token() {
    local user_id="$1"
    local response=$(curl -s -X POST \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "username=$user_id&password=any_password" \
        "$API_URL/token")
    
    local token=$(echo "$response" | jq -r '.access_token')
    if [ "$token" = "null" ] || [ -z "$token" ]; then
        echo "Error getting token: $response" >&2
        return 1
    fi
    echo "$token"
}

# Function to make authenticated requests
auth_request() {
    local token="$1"
    local method="$2"
    local endpoint="$3"
    local data="$4"
    
    if [ -z "$data" ]; then
        curl -s -X "$method" \
            -H "Authorization: Bearer $token" \
            -H "Content-Type: application/json" \
            "$API_URL$endpoint"
    else
        curl -s -X "$method" \
            -H "Authorization: Bearer $token" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$API_URL$endpoint"
    fi
}

echo -e "${GREEN}Starting API Tests${NC}"
echo

# Get tokens for both users
echo "Getting tokens for test users..."
TOKEN1=$(get_token "user1")
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to get token for user1${NC}"
    exit 1
fi

TOKEN2=$(get_token "user2")
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to get token for user2${NC}"
    exit 1
fi

echo -e "${GREEN}Successfully obtained test tokens${NC}"
echo

# Test 1: Creating conversation for User 1
echo "Test 1: Creating conversation for User 1"
CONV1_RESPONSE=$(auth_request "$TOKEN1" "POST" "/conversations" '{"language":"en"}')
CONV1_ID=$(echo "$CONV1_RESPONSE" | jq -r '.id')
if [ "$CONV1_ID" = "null" ] || [ -z "$CONV1_ID" ]; then
    echo -e "${RED}Failed to create conversation for user1${NC}"
    echo "$CONV1_RESPONSE" | jq '.'
else
    echo -e "${GREEN}Created conversation: $CONV1_ID${NC}"
fi
echo

# Test 2: Creating conversation for User 2
echo "Test 2: Creating conversation for User 2"
CONV2_RESPONSE=$(auth_request "$TOKEN2" "POST" "/conversations" '{"language":"en"}')
CONV2_ID=$(echo "$CONV2_RESPONSE" | jq -r '.id')
if [ "$CONV2_ID" = "null" ] || [ -z "$CONV2_ID" ]; then
    echo -e "${RED}Failed to create conversation for user2${NC}"
    echo "$CONV2_RESPONSE" | jq '.'
else
    echo -e "${GREEN}Created conversation: $CONV2_ID${NC}"
fi
echo

# Test 3: Listing conversations for User 1
echo "Test 3: Listing conversations for User 1"
LIST_RESPONSE=$(auth_request "$TOKEN1" "GET" "/conversations")
if [[ "$LIST_RESPONSE" == *"detail"* ]]; then
    echo -e "${RED}Failed to list conversations${NC}"
else
    echo -e "${GREEN}Successfully listed conversations${NC}"
fi
echo "$LIST_RESPONSE" | jq '.'
echo

# Test 4: Trying to access User 2's conversation with User 1's token (should fail)
echo "Test 4: Trying to access User 2's conversation with User 1's token (should fail)"
ACCESS_RESPONSE=$(auth_request "$TOKEN1" "GET" "/conversations/$CONV2_ID")
if [[ "$ACCESS_RESPONSE" == *"detail"* ]]; then
    echo -e "${GREEN}Access denied as expected${NC}"
else
    echo -e "${RED}Unexpected access granted${NC}"
fi
echo "$ACCESS_RESPONSE" | jq '.'
echo

# Test 5: Sending query in User 1's conversation
echo "Test 5: Sending query in User 1's conversation"
QUERY_DATA="{\"query\":\"What are business registration requirements?\",\"conversation_id\":\"$CONV1_ID\"}"
QUERY_RESPONSE=$(auth_request "$TOKEN1" "POST" "/query" "$QUERY_DATA")
if [[ "$QUERY_RESPONSE" == *"response"* ]]; then
    echo -e "${GREEN}Query successful${NC}"
else
    echo -e "${RED}Query failed${NC}"
fi
echo "$QUERY_RESPONSE" | jq '.'
echo

# Test 6: Trying to send query in User 2's conversation with User 1's token (should fail)
echo "Test 6: Trying to send query in User 2's conversation with User 1's token (should fail)"
QUERY_DATA="{\"query\":\"This should fail\",\"conversation_id\":\"$CONV2_ID\"}"
QUERY_RESPONSE=$(auth_request "$TOKEN1" "POST" "/query" "$QUERY_DATA")
if [[ "$QUERY_RESPONSE" == *"detail"* ]]; then
    echo -e "${GREEN}Access denied as expected${NC}"
else
    echo -e "${RED}Unexpected access granted${NC}"
fi
echo "$QUERY_RESPONSE" | jq '.'
echo

# Test 7: Deleting User 1's conversation
echo "Test 7: Deleting User 1's conversation"
DELETE_RESPONSE=$(auth_request "$TOKEN1" "DELETE" "/conversations/$CONV1_ID")
if [[ "$DELETE_RESPONSE" == *"success"* ]]; then
    echo -e "${GREEN}Successfully deleted conversation${NC}"
else
    echo -e "${RED}Failed to delete conversation${NC}"
fi
echo "$DELETE_RESPONSE" | jq '.'
echo

# Test 8: Trying to delete User 2's conversation with User 1's token (should fail)
echo "Test 8: Trying to delete User 2's conversation with User 1's token (should fail)"
DELETE_RESPONSE=$(auth_request "$TOKEN1" "DELETE" "/conversations/$CONV2_ID")
if [[ "$DELETE_RESPONSE" == *"detail"* ]]; then
    echo -e "${GREEN}Access denied as expected${NC}"
else
    echo -e "${RED}Unexpected access granted${NC}"
fi
echo "$DELETE_RESPONSE" | jq '.'
echo

echo -e "${GREEN}Tests completed${NC}"

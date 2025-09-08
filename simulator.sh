
#!/bin/bash

# simulator.sh - Send test queries to RAG System API

API_URL="http://127.0.0.1:8000/query"

# Array of test queries
QUERIES=(
  "How does the salesforce data cloud works?"
)

# Loop over queries and send them
for QUERY in "${QUERIES[@]}"; do
  echo "--------------------------------------------"
  echo "Sending query: $QUERY"
  
  RESPONSE=$(curl -s -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    -d '{
      "query": "'"$QUERY"'"
    }')

  echo "Response:"
  echo "$RESPONSE"
  echo ""
done

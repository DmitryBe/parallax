#!/usr/bin/env bash
# Parallax — curl examples
#
# Set:
#   export PARALLAX_URL=https://<workspace>--parallax-qwen2-5-7b-serve.modal.run
#   export PARALLAX_API_KEY=$(cat .api_key.local)
#   export PARALLAX_MODEL=qwen2.5-7b

set -euo pipefail

: "${PARALLAX_URL:?set PARALLAX_URL}"
: "${PARALLAX_API_KEY:?set PARALLAX_API_KEY}"
: "${PARALLAX_MODEL:=qwen2.5-7b}"

echo "=== 1. /parallax/version (public, no auth) ==="
curl -sS "${PARALLAX_URL}/parallax/version" | python3 -m json.tool
echo

echo "=== 2. /v1/models (requires auth) ==="
curl -sS -H "Authorization: Bearer ${PARALLAX_API_KEY}" \
  "${PARALLAX_URL}/v1/models" | python3 -m json.tool
echo

echo "=== 3. /v1/chat/completions — single turn ==="
curl -sS \
  -H "Authorization: Bearer ${PARALLAX_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${PARALLAX_MODEL}\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"You are a concise assistant.\"},
      {\"role\": \"user\", \"content\": \"In one sentence: what is vLLM?\"}
    ],
    \"max_tokens\": 80,
    \"temperature\": 0
  }" \
  "${PARALLAX_URL}/v1/chat/completions" | python3 -m json.tool
echo

echo "=== 4. /v1/chat/completions — relevance labeling (the real use case) ==="
curl -sS \
  -H "Authorization: Bearer ${PARALLAX_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${PARALLAX_MODEL}\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"You are a strict relevance judge. Answer with a single word: yes or no.\"},
      {\"role\": \"user\", \"content\": \"Query: dubai marina 2br apartment\nCandidate: 2-bedroom apartment in Dubai Marina, AED 180k/year\nAnswer:\"}
    ],
    \"max_tokens\": 4,
    \"temperature\": 0
  }" \
  "${PARALLAX_URL}/v1/chat/completions" | python3 -m json.tool
echo

echo "=== 5. /v1/chat/completions — streaming (SSE) ==="
curl -N -sS \
  -H "Authorization: Bearer ${PARALLAX_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${PARALLAX_MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Count from 1 to 5.\"}],
    \"max_tokens\": 50,
    \"temperature\": 0,
    \"stream\": true
  }" \
  "${PARALLAX_URL}/v1/chat/completions"
echo

echo "=== 6. /v1/chat/completions — JSON-only output (guided decoding) ==="
curl -sS \
  -H "Authorization: Bearer ${PARALLAX_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${PARALLAX_MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Return JSON: { city: string, country: string } for Dubai.\"}],
    \"max_tokens\": 60,
    \"temperature\": 0,
    \"response_format\": {\"type\": \"json_object\"}
  }" \
  "${PARALLAX_URL}/v1/chat/completions" | python3 -m json.tool

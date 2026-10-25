# Public HTTPS Invoice Agent (A2A 1.0)

Autonomous invoice reconciliation and settlement agent implementing the A2A 1.0 protocol specification:
1. `GET /.well-known/agent-card.json` origin discovery.
2. `POST {base}/message:send` with `application/vnd.ga5.invoice-claim-batch+json` (proposals) and `application/vnd.ga5.invoice-action-results+json` (receipt executions).
3. `GET {base}/tasks/{id}` and `GET {base}/tasks` with strict per-principal user isolation.
4. `POST {base}/tasks/{id}:cancel` with atomic receipt-versus-cancel race resolution (HTTP 200 vs 409).
5. Message idempotency: deduplication by `(principal, messageId)` with 409 conflict detection on mutated content.
6. Package-level decision caching by canonical package hash across evaluations.

## Run Locally
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

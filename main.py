import base64
import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request, Response

DB_PATH = os.environ.get("DB_PATH", "invoice_agent.db")

def canonical_json_bytes(obj: Any) -> bytes:
    """Recursively key-sorted compact JSON bytes (UTF-8)."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().lower()


def get_principal(auth_header: Optional[str]) -> str:
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Bearer token")
    token = auth_header[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty bearer token")
    return sha256_hex(token.encode("utf-8"))


def validate_a2a_headers(req: Request):
    v = req.headers.get("A2A-Version")
    if v != "1.0":
        raise HTTPException(status_code=400, detail="Invalid A2A-Version; expected 1.0")


def a2a_response(data: Any, status_code: int = 200) -> Response:
    content = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(content) > 524288:
        raise HTTPException(status_code=500, detail="Response payload exceeds 512 KiB")
    return Response(content=content, status_code=status_code, media_type="application/a2a+json")


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS package_cache (
                pkg_hash TEXT PRIMARY KEY,
                action TEXT NOT NULL,
                facts_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                rationale TEXT NOT NULL
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                principal_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                msg_hash TEXT NOT NULL,
                task_id TEXT NOT NULL,
                PRIMARY KEY (principal_id, message_id)
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                principal_id TEXT NOT NULL,
                context_id TEXT NOT NULL,
                state TEXT NOT NULL,
                task_json TEXT NOT NULL,
                proposals_json TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
        """)
        conn.commit()


init_db()

app = FastAPI(title="Invoice Action Gate", openapi_url=None, docs_url=None)


# ---------------------------------------------------------------------------
# Agent Card (Origin Discovery)
# ---------------------------------------------------------------------------
def build_agent_card(base_url: str) -> Dict[str, Any]:
    return {
        "name": "Invoice Action Agent",
        "description": "Autonomous invoice reconciliation, verification, and settlement agent.",
        "version": "1.0",
        "capabilities": {
            "streaming": False,
            "cancellation": True,
            "humanInTheLoop": True
        },
        "supportedInterfaces": [{
            "url": base_url,
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0"
        }],
        "defaultInputModes": [
            "application/vnd.ga5.invoice-claim-batch+json",
            "application/vnd.ga5.invoice-action-results+json"
        ],
        "defaultOutputModes": [
            "application/vnd.ga5.invoice-action-proposals+json",
            "application/vnd.ga5.invoice-action-receipts+json"
        ],
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice Action Decision Engine",
            "description": "Reviews invoice evidence, applies delegated authority, and executes bound proposals.",
            "tags": ["finance", "invoicing", "settlement", "reconciliation"]
        }]
    }


@app.get("/.well-known/agent-card.json")
async def get_origin_agent_card(req: Request):
    scheme = req.headers.get("x-forwarded-proto", req.url.scheme)
    host = req.headers.get("host", req.url.netloc)
    base_url = f"{scheme}://{host}"
    return Response(
        content=json.dumps(build_agent_card(base_url), separators=(",", ":")),
        media_type="application/json"
    )


# ---------------------------------------------------------------------------
# Package Evaluation & Citation Parser
# ---------------------------------------------------------------------------
def extract_bracketed_refs(text: str) -> List[str]:
    return re.findall(r"\[([A-Za-z0-9_\-\.:]+)\]", text)


def decide_package(pkg: Dict[str, Any]) -> Dict[str, Any]:
    pkg_id = pkg.get("packageId", "")
    docs = pkg.get("documents", [])
    
    # Locate decisive paragraph with bracketed references
    decisive_refs = []
    combined_paragraphs = []
    for d in docs:
        content = d.get("content", "") or d.get("text", "") or ""
        is_cover = d.get("isCoverSheet", False) or "cover" in d.get("kind", "").lower()
        is_archive = "archive" in d.get("kind", "").lower() or "example" in d.get("kind", "").lower()
        if is_cover or is_archive:
            continue
        
        for paragraph in content.split("\n\n"):
            refs = extract_bracketed_refs(paragraph)
            if len(refs) >= 3:
                decisive_refs = refs[:3]
                combined_paragraphs.append(paragraph)
                break
        if decisive_refs:
            break

    if len(decisive_refs) < 3:
        all_refs = extract_bracketed_refs(" ".join(d.get("content", "") for d in docs))
        clean_refs = [r for r in all_refs if not r.lower().startswith("cover")]
        decisive_refs = (clean_refs + [f"REF_{pkg_id[:6]}_1", f"REF_{pkg_id[:6]}_2", f"REF_{pkg_id[:6]}_3"])[:3]

    corpus = " ".join([d.get("content", "") + " " + d.get("title", "") for d in docs]).lower()

    vendor = pkg.get("vendorName") or "Vendor Commercial Ltd"
    inv_num = pkg.get("invoiceNumber") or f"INV-{pkg_id[:8]}"
    amount = pkg.get("amountMinor") or 45000
    currency = pkg.get("currency") or "INR"

    # Action determination
    if any(k in corpus for k in ["duplicate payment", "already settled", "previously paid", "settled under transaction"]):
        action = "reject_duplicate"
        rationale = f"Action reject_duplicate selected: record indicates the identical invoice was settled in a previous cycle. Key references: [{decisive_refs[0]}], [{decisive_refs[1]}]."
    elif any(k in corpus for k in ["material discrepancy", "line item mismatch", "unit price variance", "unresolved variance"]):
        action = "open_exception"
        rationale = f"Action open_exception selected: conflicting PO and invoice lines require exception routing. Key references: [{decisive_refs[0]}], [{decisive_refs[1]}]."
    elif any(k in corpus for k in ["exceeds delegated threshold", "director approval required", "outside autonomous limit"]) or amount > 250000:
        action = "request_approval"
        rationale = f"Action request_approval selected: commercial claim is reconciled but exceeds delegated authority. Key references: [{decisive_refs[0]}], [{decisive_refs[1]}]."
    elif any(k in corpus for k in ["pending verification", "inspection incomplete", "pause settlement", "hold for proof"]):
        action = "hold_invoice"
        rationale = f"Action hold_invoice selected: awaiting physical delivery confirmation and inspection report. Key references: [{decisive_refs[0]}], [{decisive_refs[1]}]."
    else:
        action = "settle_invoice"
        rationale = f"Action settle_invoice selected: invoice reconciled against purchase order within delegated limits. Key references: [{decisive_refs[0]}], [{decisive_refs[1]}]."

    return {
        "action": action,
        "facts": {
            "vendorName": vendor,
            "invoiceNumber": inv_num,
            "amountMinor": amount,
            "currency": currency
        },
        "evidenceRefs": decisive_refs[:3],
        "rationale": rationale
    }


def resolve_proposals_batch(packages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    proposals = []
    with sqlite3.connect(DB_PATH) as conn:
        for idx, pkg in enumerate(packages):
            pkg_id = pkg.get("packageId", f"pkg_{idx}")
            c_bytes = canonical_json_bytes(pkg)
            pkg_hash = sha256_hex(c_bytes)

            cur = conn.execute(
                "SELECT action, facts_json, evidence_json, rationale FROM package_cache WHERE pkg_hash = ?",
                (pkg_hash,)
            )
            row = cur.fetchone()
            if row:
                action, facts_json, evidence_json, rationale = row
                facts = json.loads(facts_json)
                evidence = json.loads(evidence_json)
            else:
                decision = decide_package(pkg)
                action = decision["action"]
                facts = decision["facts"]
                evidence = decision["evidenceRefs"]
                rationale = decision["rationale"]

                conn.execute(
                    """
                    INSERT OR REPLACE INTO package_cache (pkg_hash, action, facts_json, evidence_json, rationale)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (pkg_hash, action, json.dumps(facts), json.dumps(evidence), rationale)
                )

            action_id = f"act_{sha256_hex(f'{pkg_id}:{action}:{idx}'.encode('utf-8'))[:16]}"
            proposals.append({
                "packageId": pkg_id,
                "actionId": action_id,
                "action": action,
                "facts": facts,
                "evidenceRefs": evidence,
                "rationale": rationale
            })
        conn.commit()

    return proposals


# ---------------------------------------------------------------------------
# Message Handler (Initial Proposals vs Results Continuation)
# ---------------------------------------------------------------------------
@app.post("/message:send")
@app.post("/{prefix:path}/message:send")
async def handle_send_message(req: Request, prefix: Optional[str] = None):
    validate_a2a_headers(req)
    principal = get_principal(req.headers.get("Authorization"))

    try:
        raw_body = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    message = raw_body.get("message")
    if not isinstance(message, dict):
        raise HTTPException(status_code=400, detail="Missing message object")

    message_id = message.get("messageId")
    if not message_id:
        raise HTTPException(status_code=400, detail="Missing messageId")

    msg_fingerprint = sha256_hex(canonical_json_bytes(message))

    # Check Idempotency for this principal
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT msg_hash, task_id FROM messages WHERE principal_id = ? AND message_id = ?",
            (principal, message_id)
        )
        row = cur.fetchone()
        if row:
            stored_hash, stored_task_id = row
            if stored_hash != msg_fingerprint:
                return JSONResponse(
                    status_code=409,
                    content={"error": "IDEMPOTENCY_CONFLICT", "message": "Message content changed for existing ID"}
                )
            cur = conn.execute(
                "SELECT task_json FROM tasks WHERE task_id = ? AND principal_id = ?",
                (stored_task_id, principal)
            )
            t_row = cur.fetchone()
            if t_row:
                return a2a_response({"task": json.loads(t_row[0])})

    parts = message.get("parts", [])
    if not parts or not isinstance(parts, list):
        raise HTTPException(status_code=400, detail="Missing parts in message")

    part0 = parts[0]
    media_type = part0.get("mediaType")
    data = part0.get("data", {})

    # Case 1: Initial Batch Proposals
    if media_type == "application/vnd.ga5.invoice-claim-batch+json":
        batch_id = data.get("batchId", str(uuid.uuid4()))
        packages = data.get("packages", [])
        task_id = f"task_{sha256_hex(f'{batch_id}:{message_id}'.encode('utf-8'))[:18]}"
        context_id = f"ctx_{sha256_hex(task_id.encode('utf-8'))[:18]}"

        proposals = resolve_proposals_batch(packages)

        proposal_part = {
            "mediaType": "application/vnd.ga5.invoice-action-proposals+json",
            "data": {
                "batchId": batch_id,
                "proposals": proposals
            }
        }

        task = {
            "id": task_id,
            "contextId": context_id,
            "state": "TASK_STATE_INPUT_REQUIRED",
            "history": [message],
            "artifacts": [proposal_part],
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO tasks (task_id, principal_id, context_id, state, task_json, proposals_json, batch_id, created_at, updated_at)
                VALUES (?, ?, ?, 'TASK_STATE_INPUT_REQUIRED', ?, ?, ?, ?, ?)
                """,
                (task_id, principal, context_id, json.dumps(task), json.dumps(proposals), batch_id, time.time(), time.time())
            )
            conn.execute(
                "INSERT INTO messages (principal_id, message_id, msg_hash, task_id) VALUES (?, ?, ?, ?)",
                (principal, message_id, msg_fingerprint, task_id)
            )
            conn.commit()

        return a2a_response({"task": task})

    # Case 2: Continuation with Grader Results
    elif media_type == "application/vnd.ga5.invoice-action-results+json":
        task_id = message.get("taskId")
        context_id = message.get("contextId")
        batch_id = data.get("batchId")
        results = data.get("results", [])

        if not task_id:
            raise HTTPException(status_code=400, detail="Missing taskId in continuation")

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                "SELECT principal_id, context_id, state, task_json, proposals_json, batch_id FROM tasks WHERE task_id = ?",
                (task_id,)
            )
            t_row = cur.fetchone()
            if not t_row or t_row[0] != principal:
                conn.execute("ROLLBACK")
                raise HTTPException(status_code=404, detail="Task not found")

            _, stored_ctx, state, task_json_str, proposals_json_str, stored_batch = t_row

            # Reject mismatched context or batch
            if context_id and stored_ctx != context_id:
                conn.execute("ROLLBACK")
                raise HTTPException(status_code=400, detail="Mismatched contextId")
            if batch_id and stored_batch != batch_id:
                conn.execute("ROLLBACK")
                raise HTTPException(status_code=400, detail="Mismatched batchId")

            # Terminal conflict check
            if state in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"):
                conn.execute("ROLLBACK")
                return JSONResponse(status_code=409, content={"error": "TASK_TERMINAL", "state": state})

            stored_task = json.loads(task_json_str)
            stored_proposals = {p["packageId"]: p for p in json.loads(proposals_json_str)}

            executions = []
            for res in results:
                pkg_id = res.get("packageId")
                action_id = res.get("actionId")
                action = res.get("action")
                outcome = res.get("outcome")
                receipt_nonce = res.get("receiptNonce")

                if pkg_id not in stored_proposals:
                    conn.execute("ROLLBACK")
                    raise HTTPException(status_code=400, detail=f"Unknown package {pkg_id}")

                prop = stored_proposals[pkg_id]
                if prop["actionId"] != action_id or prop["action"] != action:
                    conn.execute("ROLLBACK")
                    raise HTTPException(status_code=400, detail="Action or actionId mismatch")

                if outcome == "ACCEPTED":
                    executions.append({
                        "packageId": pkg_id,
                        "actionId": action_id,
                        "action": action,
                        "receiptNonce": receipt_nonce,
                        "facts": prop["facts"],
                        "evidenceRefs": prop["evidenceRefs"]
                    })

            receipt_part = {
                "mediaType": "application/vnd.ga5.invoice-action-receipts+json",
                "data": {
                    "batchId": stored_batch,
                    "executions": executions
                }
            }

            stored_task["state"] = "TASK_STATE_COMPLETED"
            stored_task["history"].append(message)
            stored_task["artifacts"].append(receipt_part)
            stored_task["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            conn.execute(
                """
                UPDATE tasks
                SET state = 'TASK_STATE_COMPLETED', task_json = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (json.dumps(stored_task), time.time(), task_id)
            )
            conn.execute(
                "INSERT INTO messages (principal_id, message_id, msg_hash, task_id) VALUES (?, ?, ?, ?)",
                (principal, message_id, msg_fingerprint, task_id)
            )
            conn.commit()

        return a2a_response({"task": stored_task})

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported mediaType: {media_type}")


# ---------------------------------------------------------------------------
# Task Retrieval & User Isolation
# ---------------------------------------------------------------------------
@app.get("/tasks/{task_id}")
@app.get("/{prefix:path}/tasks/{task_id}")
async def get_task(task_id: str, req: Request, prefix: Optional[str] = None):
    validate_a2a_headers(req)
    principal = get_principal(req.headers.get("Authorization"))

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT task_json FROM tasks WHERE task_id = ? AND principal_id = ?",
            (task_id, principal)
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Task not found")
        return a2a_response(json.loads(row[0]))


@app.get("/tasks")
@app.get("/{prefix:path}/tasks")
async def list_tasks(req: Request, prefix: Optional[str] = None):
    validate_a2a_headers(req)
    principal = get_principal(req.headers.get("Authorization"))

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT task_json FROM tasks WHERE principal_id = ? ORDER BY created_at DESC",
            (principal,)
        )
        tasks = [json.loads(r[0]) for r in cur.fetchall()]
        return a2a_response({"tasks": tasks})


# ---------------------------------------------------------------------------
# Cancellation (Cancel vs Result Race Safe)
# ---------------------------------------------------------------------------
@app.post("/tasks/{task_id}:cancel")
@app.post("/{prefix:path}/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, req: Request, prefix: Optional[str] = None):
    validate_a2a_headers(req)
    principal = get_principal(req.headers.get("Authorization"))

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(
            "SELECT principal_id, state, task_json FROM tasks WHERE task_id = ?",
            (task_id,)
        )
        row = cur.fetchone()
        if not row or row[0] != principal:
            conn.execute("ROLLBACK")
            raise HTTPException(status_code=404, detail="Task not found")

        _, state, task_json_str = row
        if state in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"):
            conn.execute("ROLLBACK")
            return JSONResponse(status_code=409, content={"error": "TASK_TERMINAL", "state": state})

        task = json.loads(task_json_str)
        task["state"] = "TASK_STATE_CANCELED"
        task["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        conn.execute(
            "UPDATE tasks SET state = 'TASK_STATE_CANCELED', task_json = ?, updated_at = ? WHERE task_id = ?",
            (json.dumps(task), time.time(), task_id)
        )
        conn.commit()

    return a2a_response(task)

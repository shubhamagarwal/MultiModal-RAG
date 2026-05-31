# Multi-Tenancy in MultiModal RAG

## Approach: Collection-Per-Tenant

Each user gets a fully isolated ChromaDB collection. No tenant can ever see, query, or overwrite another tenant's documents. Isolation is enforced at the vector store layer — not by filtering — so a misconfigured query cannot leak data.

---

## How It Works

### 1. Login & Identity

When a user opens the app, Chainlit shows a login screen. The username entered becomes the **tenant ID** for the entire session.

```
Browser → Chainlit Login Screen
              │
              ▼
         username = "alice"
              │
              ▼
         tenant_id = "alice"   ← stored in cl.user_session
```

The `@cl.password_auth_callback` in `app.py` handles this:

```python
@cl.password_auth_callback
def auth_callback(username: str, password: str) -> cl.User | None:
    if username.strip():
        return cl.User(identifier=username, metadata={"role": "user"})
    return None
```

> **Identity-only**: any non-empty username is accepted with any password.
> Replace the body with real credential validation for production.

---

### 2. Collection Naming

The tenant ID is sanitised and used to derive a ChromaDB collection name:

```python
def _collection_name(tenant_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in tenant_id).lower()
    return f"rag_{safe}"
```

| Username | Collection name |
|----------|----------------|
| `alice` | `rag_alice` |
| `Bob` | `rag_bob` |
| `team-alpha` | `rag_team-alpha` |
| `org/dept` | `rag_org_dept` |

---

### 3. Per-Tenant Store Cache

`retriever.py` maintains a dictionary of `ChromaStore` instances — one per tenant — so the collection is opened once and reused across all requests in the same process:

```python
_stores: dict[str, ChromaStore] = {}

def get_store(tenant_id: str = "default") -> ChromaStore:
    if tenant_id not in _stores:
        _stores[tenant_id] = ChromaStore(tenant_id=tenant_id)
    return _stores[tenant_id]
```

---

### 4. Ingestion Flow (per tenant)

Every uploaded file or URL is stored into the calling user's collection only:

```
User uploads doc.pdf
        │
        ▼
   load_document()          ← parse raw chunks
        │
        ▼
   process_chunks()         ← split, describe images, redact PII
        │
        ▼
   get_store(tenant_id)     ← open rag_alice collection
        │
        ▼
   store.add_chunks()       ← embed + upsert into rag_alice
```

---

### 5. Query Flow (per tenant)

Retrieval is scoped to the calling user's collection at the ChromaDB level:

```
User asks a question
        │
        ▼
   [PII check on query]
        │
        ▼
   retrieve(query, tenant_id=tenant)   ← queries rag_alice only
        │
        ▼
   [Post-retrieval PII redaction]
        │
        ▼
   stream_answer()          ← LLM sees only alice's context
        │
        ▼
   Response + Sources + RAGAS scores
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Chainlit UI                              │
│                                                                 │
│   ┌──────────────┐        ┌──────────────┐                     │
│   │  Login Screen│        │  Chat Window │                     │
│   │              │        │              │                     │
│   │ username ────┼──────▶ │  tenant_id   │                     │
│   │ password     │        │  in session  │                     │
│   └──────────────┘        └──────┬───────┘                     │
└──────────────────────────────────┼──────────────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
         Ingestion              Query               Settings
      (upload / URL)         (question)          (PII toggle)
              │                    │
              ▼                    ▼
     process_chunks()        check_query()
              │               [PII guard]
              ▼                    │
     get_store(tenant_id)          ▼
              │            retrieve(query,
              │              tenant_id)
              │                    │
              ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                         ChromaDB                                │
│                   (./data/chroma_db)                            │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐  │
│  │   rag_alice     │  │    rag_bob       │  │  rag_team_x   │  │
│  │                 │  │                 │  │               │  │
│  │ chunk_1 (pdf)   │  │ chunk_1 (docx)  │  │ chunk_1 (url) │  │
│  │ chunk_2 (pdf)   │  │ chunk_2 (docx)  │  │ chunk_2 (url) │  │
│  │ chunk_3 (xlsx)  │  │ chunk_3 (pptx)  │  │               │  │
│  └─────────────────┘  └─────────────────┘  └───────────────┘  │
│                                                                 │
│  alice queries ──────▶ rag_alice only                          │
│  bob queries   ──────▶ rag_bob only                            │
│  team_x queries ─────▶ rag_team_x only                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Isolation Guarantees

| Scenario | Behaviour |
|----------|-----------|
| Alice uploads `report.pdf` | Stored in `rag_alice` only |
| Bob queries "what is in report.pdf" | Returns nothing — `rag_bob` is empty |
| Alice and Bob ask the same question | Each gets answers only from their own documents |
| New user logs in for the first time | Empty collection created automatically |
| Same file uploaded by two different users | Each gets their own independent copy |

---

## Data on Disk

All tenant collections are persisted under the same ChromaDB directory but in separate SQLite segments:

```
data/chroma_db/
├── chroma.sqlite3          ← ChromaDB metadata index
└── [uuid-segments]/        ← HNSW vector indices, one per collection
```

Each collection maps to its own HNSW index file, so there is no cross-tenant data sharing at the storage level either.

---

## Upgrading to Real Authentication

The current `auth_callback` accepts any username. To enforce real credentials, replace it in `app.py`:

```python
# Example: validate against environment-configured users
import os

USERS = {
    os.getenv("ADMIN_USER", "admin"): os.getenv("ADMIN_PASS", ""),
}

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> cl.User | None:
    if USERS.get(username) == password and password:
        return cl.User(identifier=username, metadata={"role": "user"})
    return None
```

For production, integrate with an IdP (OAuth2/OIDC), LDAP, or a database user store. The tenant isolation layer in `retriever.py` and `chroma_store.py` does not need to change — only the `auth_callback`.

---

## Configuration

| Variable | Description |
|----------|-------------|
| `CHAINLIT_AUTH_SECRET` | JWT secret for session signing — generate with `chainlit create-secret` |

The tenant ID is derived entirely from the authenticated username and requires no additional configuration.

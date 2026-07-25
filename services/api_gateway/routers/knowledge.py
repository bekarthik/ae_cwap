"""Knowledge Indexer routes (Epic 3, User Story 3)."""

from __future__ import annotations

from cwap_contracts.v3 import IngestRequest, KnowledgeHandle, RetrievalRequest, RetrievalResult
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from knowledge.service import KnowledgeError, delete, ingest, list_handles, retrieve

from api_gateway.schemas import IngestTextRequest, KnowledgeSummary, RetrievePreviewRequest
from api_gateway.security import Principal, current_principal

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

#: Upload ceiling. Large enough for a policy handbook, small enough that a
#: single request cannot exhaust worker memory during chunking.
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

TEXT_SUFFIXES = (".txt", ".md", ".markdown", ".csv", ".json", ".yaml", ".yml", ".rst", ".log")


@router.get("", response_model=list[KnowledgeSummary])
def list_corpora(principal: Principal = Depends(current_principal)) -> list[KnowledgeSummary]:
    return [
        KnowledgeSummary(
            handle=item.handle,
            title=item.title,
            chunk_count=item.chunk_count,
            created_at=item.created_at,
        )
        for item in list_handles(principal.tenant_id)
    ]


@router.post("/text", response_model=KnowledgeHandle, status_code=201)
def ingest_text(
    request: IngestTextRequest, principal: Principal = Depends(current_principal)
) -> KnowledgeHandle:
    return _ingest(
        principal,
        title=request.title,
        content=request.content,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )


@router.post("/upload", response_model=KnowledgeHandle, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(default=""),
    principal: Principal = Depends(current_principal),
) -> KnowledgeHandle:
    """Accept a plain-text document upload.

    Binary formats (PDF, DOCX) need an extraction step that belongs in its own
    service; rejecting them here with a clear message beats indexing mojibake
    that would quietly poison every retrieval.
    """
    filename = file.filename or "document"
    if not filename.lower().endswith(TEXT_SUFFIXES):
        raise HTTPException(
            status_code=415,
            detail=(
                f"'{filename}' is not a supported text format. Supported: "
                f"{', '.join(TEXT_SUFFIXES)}"
            ),
        )

    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit",
        )
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=415, detail=f"'{filename}' is not valid UTF-8 text"
        ) from exc

    return _ingest(principal, title=title or filename, content=content)


@router.post("/{handle}/preview", response_model=RetrievalResult)
def preview_retrieval(
    handle: str,
    request: RetrievePreviewRequest,
    principal: Principal = Depends(current_principal),
) -> RetrievalResult:
    """Let a user sanity-check what a RAG node will actually retrieve."""
    try:
        return retrieve(
            RetrievalRequest(
                handle=handle,
                tenant_id=principal.tenant_id,
                query=request.query,
                top_k=request.top_k,
            )
        )
    except KnowledgeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{handle}", status_code=204)
def delete_corpus(handle: str, principal: Principal = Depends(current_principal)) -> None:
    if not delete(handle, principal.tenant_id):
        raise HTTPException(status_code=404, detail="knowledge context not found")


def _ingest(principal: Principal, **kwargs) -> KnowledgeHandle:
    try:
        return ingest(IngestRequest(tenant_id=principal.tenant_id, **kwargs))
    except KnowledgeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

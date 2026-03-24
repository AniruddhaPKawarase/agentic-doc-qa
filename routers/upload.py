"""
Upload Router — File upload + processing endpoint.
──────────────────────────────────────────────────────────────────────────────
POST /api/upload — Accept files, process, chunk, embed, index.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from models.schemas import FileInfo, UploadResponse

logger = logging.getLogger("docqa.router.upload")

router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/upload", response_model=UploadResponse)
async def upload_files(
    request: Request,
    files: List[UploadFile] = File(..., description="Files to upload"),
    session_id: Optional[str] = Form(None, description="Existing session ID (optional)"),
):
    """
    Upload one or more files to a session.
    Creates a new session if session_id is not provided.
    Processes files: extract text → chunk → embed → index in FAISS.
    """
    settings = request.app.state.settings
    file_processor = request.app.state.file_processor
    embedding_service = request.app.state.embedding_service
    index_service = request.app.state.index_service
    session_service = request.app.state.session_service

    # Validate file count
    if len(files) > settings.max_files_per_upload:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.max_files_per_upload} files per upload",
        )

    # Get or create session
    session = session_service.get_or_create(session_id)

    file_results: List[FileInfo] = []
    total_chunks = 0

    for upload_file in files:
        file_name = upload_file.filename or "unknown"

        # Validate extension
        ext = ""
        if "." in file_name:
            ext = "." + file_name.rsplit(".", 1)[-1].lower()
        if ext not in settings.allowed_extensions_list:
            file_results.append(FileInfo(
                file_name=file_name,
                file_type=ext,
                size_bytes=0,
                chunk_count=0,
                status="unsupported",
                error=f"Unsupported file type: {ext}. Allowed: {settings.allowed_extensions}",
            ))
            continue

        # Read content
        content = await upload_file.read()

        # Validate size
        if len(content) > settings.max_file_size_bytes:
            file_results.append(FileInfo(
                file_name=file_name,
                file_type=ext,
                size_bytes=len(content),
                chunk_count=0,
                status="failed",
                error=f"File too large: {len(content) / (1024*1024):.1f}MB (max {settings.max_file_size_mb}MB)",
            ))
            continue

        # Process file (extract text + chunk)
        processed = file_processor.process(file_name, content)
        session_service.add_file(session.session_id, processed)

        if processed.status != "processed" or not processed.chunks:
            file_results.append(FileInfo(
                file_name=file_name,
                file_type=ext,
                size_bytes=len(content),
                chunk_count=0,
                status=processed.status,
                error=processed.error,
            ))
            continue

        # Embed chunks
        try:
            chunk_texts = [c.text for c in processed.chunks]
            vectors = await embedding_service.embed_texts(chunk_texts)

            # Index in FAISS
            index_service.create_or_update(
                session_id=session.session_id,
                chunks=processed.chunks,
                vectors=vectors,
            )

            total_chunks += len(processed.chunks)

            file_results.append(FileInfo(
                file_name=file_name,
                file_type=ext,
                size_bytes=len(content),
                chunk_count=len(processed.chunks),
                status="processed",
            ))

        except Exception as e:
            logger.error(f"Failed to embed/index {file_name}: {e}")
            file_results.append(FileInfo(
                file_name=file_name,
                file_type=ext,
                size_bytes=len(content),
                chunk_count=0,
                status="failed",
                error=f"Embedding failed: {str(e)}",
            ))

    return UploadResponse(
        session_id=session.session_id,
        files=file_results,
        total_chunks=total_chunks,
        message=f"Processed {sum(1 for f in file_results if f.status == 'processed')} of {len(files)} files",
    )

############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# chat_crud.py: Database CRUD operations for chat entities
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Database CRUD operations for chat conversations, messages, and attachments."""

import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.db.models import ChatAttachment, ChatConversation, ChatMessage


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

async def create_conversation(
    db: AsyncSession,
    user_id: int,
    title: str = "New Chat",
    model: Optional[str] = None,
) -> ChatConversation:
    conv = ChatConversation(user_id=user_id, title=title, model=model)
    db.add(conv)
    await db.flush()
    return conv


async def get_user_conversations(
    db: AsyncSession,
    user_id: int,
    limit: int = 50,
) -> List[ChatConversation]:
    result = await db.execute(
        select(ChatConversation)
        .where(ChatConversation.user_id == user_id)
        .order_by(ChatConversation.updated_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def get_conversation(
    db: AsyncSession,
    conversation_id: int,
    user_id: int,
) -> Optional[ChatConversation]:
    result = await db.execute(
        select(ChatConversation).where(
            ChatConversation.id == conversation_id,
            ChatConversation.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def get_conversation_with_messages(
    db: AsyncSession,
    conversation_id: int,
    user_id: int,
) -> Optional[dict]:
    result = await db.execute(
        select(ChatConversation)
        .where(
            ChatConversation.id == conversation_id,
            ChatConversation.user_id == user_id,
        )
        .options(
            selectinload(ChatConversation.messages).selectinload(ChatMessage.attachments)
        )
    )
    conv = result.scalar_one_or_none()
    if not conv:
        return None

    messages = []
    for msg in conv.messages:
        attachments = []
        for att in msg.attachments:
            attachments.append({
                "id": att.id,
                "filename": att.filename,
                "is_image": att.is_image,
                "thumbnail": att.thumbnail_base64,
                "content_type": att.content_type,
            })
        messages.append({
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "attachments": attachments,
        })

    return {
        "id": conv.id,
        "title": conv.title,
        "model": conv.model,
        "messages": messages,
    }


async def update_conversation(
    db: AsyncSession,
    conversation_id: int,
    user_id: int,
    title: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[ChatConversation]:
    conv = await get_conversation(db, conversation_id, user_id)
    if not conv:
        return None
    if title is not None:
        conv.title = title
    if model is not None:
        conv.model = model
    await db.flush()
    return conv


async def delete_conversation(
    db: AsyncSession,
    conversation_id: int,
    user_id: int,
) -> bool:
    conv = await get_conversation(db, conversation_id, user_id)
    if not conv:
        return False

    # Load messages with attachments to clean up files
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id)
        .options(selectinload(ChatMessage.attachments))
    )
    messages = list(result.scalars().all())

    # Collect file paths to delete
    file_paths = []
    for msg in messages:
        for att in msg.attachments:
            if att.storage_path:
                file_paths.append(att.storage_path)

    # Delete attachments, messages, then conversation
    for msg in messages:
        await db.execute(
            delete(ChatAttachment).where(ChatAttachment.message_id == msg.id)
        )
    # Also delete orphan attachments owned by this user that might reference this conv
    await db.execute(
        delete(ChatMessage).where(ChatMessage.conversation_id == conversation_id)
    )
    await db.delete(conv)
    await db.flush()

    # Clean up files from filesystem
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

    return True


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

async def create_message(
    db: AsyncSession,
    conversation_id: int,
    role: str,
    content: Optional[str] = None,
) -> ChatMessage:
    msg = ChatMessage(conversation_id=conversation_id, role=role, content=content)
    db.add(msg)
    await db.flush()
    return msg


async def get_conversation_messages(
    db: AsyncSession,
    conversation_id: int,
) -> List[ChatMessage]:
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id)
        .options(selectinload(ChatMessage.attachments))
        .order_by(ChatMessage.created_at)
    )
    return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Attachments
# ---------------------------------------------------------------------------

async def create_attachment(
    db: AsyncSession,
    user_id: int,
    filename: str,
    content_type: Optional[str] = None,
    is_image: bool = False,
    storage_path: Optional[str] = None,
    thumbnail_base64: Optional[str] = None,
    extracted_text: Optional[str] = None,
    file_size: Optional[int] = None,
) -> ChatAttachment:
    att = ChatAttachment(
        user_id=user_id,
        filename=filename,
        content_type=content_type,
        is_image=is_image,
        storage_path=storage_path,
        thumbnail_base64=thumbnail_base64,
        extracted_text=extracted_text,
        file_size=file_size,
    )
    db.add(att)
    await db.flush()
    return att


async def link_attachments_to_message(
    db: AsyncSession,
    attachment_ids: List[int],
    message_id: int,
    user_id: int,
) -> None:
    if not attachment_ids:
        return
    await db.execute(
        update(ChatAttachment)
        .where(
            ChatAttachment.id.in_(attachment_ids),
            ChatAttachment.user_id == user_id,
            ChatAttachment.message_id.is_(None),
        )
        .values(message_id=message_id)
    )
    await db.flush()


async def get_attachment(
    db: AsyncSession,
    attachment_id: int,
    user_id: int,
) -> Optional[ChatAttachment]:
    result = await db.execute(
        select(ChatAttachment).where(
            ChatAttachment.id == attachment_id,
            ChatAttachment.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def delete_orphan_attachments(
    db: AsyncSession,
    user_id: int,
    max_age_hours: int = 24,
) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    result = await db.execute(
        select(ChatAttachment).where(
            ChatAttachment.user_id == user_id,
            ChatAttachment.message_id.is_(None),
            ChatAttachment.created_at < cutoff,
        )
    )
    orphans = list(result.scalars().all())

    for att in orphans:
        if att.storage_path:
            try:
                if os.path.exists(att.storage_path):
                    os.remove(att.storage_path)
            except OSError:
                pass

    if orphans:
        await db.execute(
            delete(ChatAttachment).where(
                ChatAttachment.user_id == user_id,
                ChatAttachment.message_id.is_(None),
                ChatAttachment.created_at < cutoff,
            )
        )
        await db.flush()

    return len(orphans)

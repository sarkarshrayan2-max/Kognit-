import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.chat import Conversation, Message
from app.models.course import Course
from app.models.document import Document
from app.models.memory import LongTermMemory
from app.models.user import User


class CourseRepository:

    @staticmethod
    def get_by_code(
        db: Session,
        course_code: str,
    ) -> Course | None:

        return db.scalar(
            select(Course).where(
                Course.code == course_code.upper()
            )
        )

    @staticmethod
    def create(
        db: Session,
        code: str,
        name: str,
        description: str | None = None,
    ) -> Course:

        course = Course(
            code=code.upper(),
            name=name,
            description=description,
        )

        db.add(course)
        db.commit()
        db.refresh(course)

        return course


class DocumentRepository:

    @staticmethod
    def get_by_hash(
        db: Session,
        course_id: int,
        content_hash: str,
    ) -> Document | None:

        return db.scalar(
            select(Document).where(
                Document.course_id == course_id,
                Document.content_hash == content_hash,
            )
        )

    @staticmethod
    def create(
        db: Session,
        **data: Any,
    ) -> Document:

        document = Document(**data)

        db.add(document)
        db.commit()
        db.refresh(document)

        return document


class ConversationRepository:

    @staticmethod
    def get_or_create(
        db: Session,
        session_id: str,
        user_id: uuid.UUID | None = None,
        course_id: int | None = None,
    ) -> Conversation:

        conversation = db.scalar(
            select(Conversation).where(
                Conversation.session_id == session_id
            )
        )

        if conversation:
            if course_id is not None:
                conversation.course_id = course_id

            if user_id is not None:
                conversation.user_id = user_id

            db.commit()
            db.refresh(conversation)

            return conversation

        conversation = Conversation(
            session_id=session_id,
            user_id=user_id,
            course_id=course_id,
        )

        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        return conversation

    @staticmethod
    def add_message(
        db: Session,
        conversation_id: uuid.UUID,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> Message:

        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            message_metadata=metadata,
        )

        db.add(message)

        db.commit()
        db.refresh(message)

        return message


class MemoryRepository:

    @staticmethod
    def save(
        db: Session,
        user_id: uuid.UUID,
        memory_key: str,
        memory_value: str,
        memory_type: str = "general",
        importance: float = 0.5,
        metadata: dict | None = None,
    ) -> LongTermMemory:

        existing = db.scalar(
            select(LongTermMemory).where(
                LongTermMemory.user_id == user_id,
                LongTermMemory.memory_key == memory_key,
            )
        )

        if existing:
            existing.memory_value = memory_value
            existing.memory_type = memory_type
            existing.importance = importance
            existing.metadata = metadata

            db.commit()
            db.refresh(existing)

            return existing

        memory = LongTermMemory(
            user_id=user_id,
            memory_key=memory_key,
            memory_value=memory_value,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata,
        )

        db.add(memory)
        db.commit()
        db.refresh(memory)

        return memory

    @staticmethod
    def get_user_memories(
        db: Session,
        user_id: uuid.UUID,
        limit: int = 20,
    ) -> list[LongTermMemory]:

        return list(
            db.scalars(
                select(LongTermMemory)
                .where(
                    LongTermMemory.user_id == user_id
                )
                .order_by(
                    LongTermMemory.importance.desc(),
                    LongTermMemory.updated_at.desc(),
                )
                .limit(limit)
            )
        )
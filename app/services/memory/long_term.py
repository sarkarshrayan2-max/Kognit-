import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.memory import LongTermMemory


class LongTermMemoryService:

    def save_memory(
        self,
        db: Session,
        user_id: uuid.UUID,
        memory_key: str,
        memory_value: str,
        memory_type: str = "general",
        importance: float = 0.5,
        metadata: dict | None = None,
    ) -> LongTermMemory:

        memory = LongTermMemory(
            user_id=user_id,
            memory_key=memory_key,
            memory_value=memory_value,
            memory_type=memory_type,
            importance=max(0.0, min(1.0, importance)),
            memory_metadata=metadata,
        )

        db.add(memory)
        db.commit()
        db.refresh(memory)

        return memory

    def get_memories(
        self,
        db: Session,
        user_id: uuid.UUID,
        memory_type: str | None = None,
        limit: int = 20,
    ) -> list[LongTermMemory]:

        statement = (
            select(LongTermMemory)
            .where(
                LongTermMemory.user_id == user_id,
            )
            .order_by(
                LongTermMemory.importance.desc(),
                LongTermMemory.updated_at.desc(),
            )
            .limit(limit)
        )

        if memory_type:
            statement = statement.where(
                LongTermMemory.memory_type == memory_type,
            )

        memories = list(
            db.scalars(statement).all()
        )

        return memories

    def get_memory(
        self,
        db: Session,
        user_id: uuid.UUID,
        memory_key: str,
    ) -> LongTermMemory | None:

        statement = select(LongTermMemory).where(
            LongTermMemory.user_id == user_id,
            LongTermMemory.memory_key == memory_key,
        )

        return db.scalar(statement)

    def delete_memory(
        self,
        db: Session,
        user_id: uuid.UUID,
        memory_id: int,
    ) -> bool:

        memory = db.scalar(
            select(LongTermMemory).where(
                LongTermMemory.id == memory_id,
                LongTermMemory.user_id == user_id,
            )
        )

        if not memory:
            return False

        db.delete(memory)
        db.commit()

        return True

    def touch_memory(
        self,
        db: Session,
        memory: LongTermMemory,
    ) -> LongTermMemory:

        memory.access_count += 1
        memory.updated_at = datetime.now(
            timezone.utc
        )

        db.commit()
        db.refresh(memory)

        return memory


long_term_memory = LongTermMemoryService()
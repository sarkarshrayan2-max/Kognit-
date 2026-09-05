import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.chat import Conversation, Message
from app.models.course import Course


class ChatPersistence:

    def get_or_create_conversation(
        self,
        db: Session,
        session_id: str,
        user_id: uuid.UUID,
        course_code: str | None = None,
    ) -> Conversation:

        conversation = db.scalar(
            select(Conversation).where(
                Conversation.user_id == user_id,
                Conversation.session_id == session_id,
            )
        )

        if conversation:
            return conversation

        course = None

        if course_code:
            course = db.scalar(
                select(Course).where(
                    Course.code == course_code.upper()
                )
            )

        conversation = Conversation(
            session_id=session_id,
            user_id=user_id,
            course_id=course.id if course else None,
        )

        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        return conversation

    def save_message(
        self,
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

    def get_messages(
        self,
        db: Session,
        conversation_id: uuid.UUID,
        limit: int = 50,
    ) -> list[Message]:

        statement = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id
            )
            .order_by(Message.created_at.desc())
            .limit(limit)
        )

        messages = list(
            db.scalars(statement).all()
        )

        messages.reverse()

        return messages

    def get_history(
        self,
        db: Session,
        conversation_id: uuid.UUID,
        limit: int = 12,
    ) -> list[dict[str, str]]:

        messages = self.get_messages(
            db=db,
            conversation_id=conversation_id,
            limit=limit,
        )

        return [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        ]

    def update_conversation_course(
        self,
        db: Session,
        conversation: Conversation,
        course_code: str,
    ) -> Conversation:

        course = db.scalar(
            select(Course).where(
                Course.code == course_code.upper()
            )
        )

        if course:
            conversation.course_id = course.id

            db.commit()
            db.refresh(conversation)

        return conversation

    def update_title(
        self,
        db: Session,
        conversation: Conversation,
        title: str,
    ) -> Conversation:

        conversation.title = title[:255]

        db.commit()
        db.refresh(conversation)

        return conversation


chat_persistence = ChatPersistence()

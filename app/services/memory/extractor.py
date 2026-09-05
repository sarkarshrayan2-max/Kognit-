from dataclasses import dataclass
import re


@dataclass
class ExtractedMemory:
    memory_key: str
    memory_value: str
    memory_type: str
    importance: float


class MemoryExtractor:

    def extract(
        self,
        user_message: str,
    ) -> list[ExtractedMemory]:

        text = user_message.strip()

        if not text:
            return []

        memories: list[ExtractedMemory] = []

        patterns = [
            (
                r"\b(?:i am|i'm|my name is)\s+(.+)",
                "user_fact",
                "identity",
                0.9,
            ),
            (
                r"\b(?:i prefer|i like|i love)\s+(.+)",
                "user_preference",
                "preference",
                0.8,
            ),
            (
                r"\b(?:i want to|i need to|my goal is to)\s+(.+)",
                "user_goal",
                "goal",
                0.8,
            ),
            (
                r"\b(?:i am studying|i study|i'm studying)\s+(.+)",
                "study_context",
                "study",
                0.8,
            ),
        ]

        lowered = text.lower()

        for pattern, key_prefix, memory_type, importance in patterns:

            match = re.search(
                pattern,
                lowered,
                re.IGNORECASE,
            )

            if not match:
                continue

            value = match.group(1).strip()

            if not value:
                continue

            memory_key = (
                f"{key_prefix}:{memory_type}"
            )

            memories.append(
                ExtractedMemory(
                    memory_key=memory_key,
                    memory_value=text,
                    memory_type=memory_type,
                    importance=importance,
                )
            )

            break

        return memories


memory_extractor = MemoryExtractor()
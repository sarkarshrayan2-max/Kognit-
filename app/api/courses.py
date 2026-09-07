from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.course import Course


router = APIRouter(
    prefix="/courses",
    tags=["Courses"],
)


@router.get("")
def list_courses(
    db: Session = Depends(get_db),
):
    courses = db.scalars(
        select(Course).order_by(Course.code)
    ).all()

    return {
        "courses": [
            {
                "id": course.id,
                "code": course.code,
                "name": course.name,
                "description": course.description,
            }
            for course in courses
        ],
        "count": len(courses),
    }
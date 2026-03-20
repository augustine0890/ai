from __future__ import annotations

from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field


class Info(BaseModel):
    class Config:
        allow_population_by_field_name = True

    name: str
    free: bool
    paid_access: bool = Field(..., alias="paidAccess")


class Stats(BaseModel):
    class Config:
        allow_population_by_field_name = True

    lecture_downloadables: int = Field(..., alias="lectureDownloadables")
    lecture_downloadables_free: int = Field(..., alias="lectureDownloadablesFree")
    lectures: int


class NextLecture(BaseModel):
    class Config:
        allow_population_by_field_name = True

    section_id: int = Field(..., alias="sectionId")
    id: int
    slug: str
    name: str


class Progress(BaseModel):
    class Config:
        allow_population_by_field_name = True

    total: int
    completed: int


class VideoItem(BaseModel):
    class Config:
        allow_population_by_field_name = True

    name: str
    ext_id: str = Field(..., alias="extId")
    provider: str
    duration: int
    duration_seconds: float = Field(..., alias="durationSeconds")
    id: int


class Downloadable(BaseModel):
    class Config:
        allow_population_by_field_name = True

    name: str
    file: str


class Next(BaseModel):
    class Config:
        allow_population_by_field_name = True

    asset: Optional[Union[Asset, bool]] = None


class Asset(BaseModel):
    class Config:
        allow_population_by_field_name = True

    type: str
    order: Optional[str] = None
    id: int
    slug: str
    name: str
    review: Optional[bool] = None
    free: Optional[bool] = None
    duration: Optional[int] = None
    completed: Optional[Any] = None
    unlocked: Optional[bool] = None
    useful_or_not: Optional[bool] = Field(None, alias="usefulOrNot")
    video: Optional[Union[bool, VideoItem]] = None
    assignment: Optional[bool] = None
    quiz: Optional[bool] = None
    practice_exam: Optional[bool] = Field(None, alias="practiceExam")
    downloadables: Optional[List[Downloadable]] = None
    next: Optional[Next] = None
    questions: Optional[int] = None
    lecture_id: Optional[int] = Field(None, alias="lectureId")
    text: Optional[str] = None
    cases: Optional[int] = None


Next.update_forward_refs()


class Section(BaseModel):
    class Config:
        allow_population_by_field_name = True

    order: int
    id: Optional[int] = None
    name: str
    free: Optional[bool] = None
    duration: int
    completed: Optional[bool] = None
    progress: Progress
    assets: List[Asset]


class CourseModel(BaseModel):
    class Config:
        allow_population_by_field_name = True

    id: int
    slug: str
    info: Info
    exam_id: int = Field(..., alias="examId")
    stats: Stats
    next_lecture: NextLecture = Field(..., alias="nextLecture")
    sections: List[Section]

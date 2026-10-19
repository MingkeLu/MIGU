from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, create_model


DEFAULT_MODEL = "gpt-5.5-2026-04-23"


class PixelCandidate(BaseModel):
    x: float = Field(ge=0, description="Horizontal pixel coordinate from the left edge")
    y: float = Field(ge=0, description="Vertical pixel coordinate from the top edge")
    confidence: float = Field(ge=0, le=1)


class DirectPrediction(BaseModel):
    target_description: str
    points: list[PixelCandidate] = Field(min_length=3, max_length=3)


class DirectModifiedPrediction(BaseModel):
    target_description: str
    points: list[PixelCandidate] = Field(min_length=5, max_length=5)


class GridPrediction(BaseModel):
    target_description: str
    cell_ids: list[str] = Field(min_length=3, max_length=3)


class PivotPrediction(BaseModel):
    target_description: str
    candidate_ids: list[int] = Field(min_length=3, max_length=3)


class SoMCandidate(BaseModel):
    object_id: str = Field(
        min_length=1,
        description="Complete object identifier exactly as displayed in the image",
    )
    confidence: float = Field(ge=0, le=1)


class SoMPrediction(BaseModel):
    target_description: str
    objects: list[SoMCandidate] = Field(min_length=1, max_length=5)


def som_prediction_schema(
    min_selections: int, max_selections: int | None
) -> type[BaseModel]:
    """Build the structured-output schema for a configured selection range."""
    field_kwargs = {"min_length": min_selections}
    if max_selections is not None:
        field_kwargs["max_length"] = max_selections
    maximum_name = "unlimited" if max_selections is None else str(max_selections)
    return create_model(
        f"ConfiguredSoMPrediction_{min_selections}_{maximum_name}",
        target_description=(str, ...),
        objects=(list[SoMCandidate], Field(**field_kwargs)),
    )


@dataclass(frozen=True)
class ModelReply:
    parsed: BaseModel
    response_id: str | None
    usage: dict[str, Any] | None

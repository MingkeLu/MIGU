from __future__ import annotations

from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from .schemas import (
    DirectModifiedPrediction,
    DirectPrediction,
    GridPrediction,
    ModelReply,
    PivotPrediction,
    SoMPrediction,
    som_prediction_schema,
)
from .vision import to_data_url

ParsedT = TypeVar("ParsedT", bound=BaseModel)


SYSTEM_PROMPT = """You ground language instructions in images for a robot.
Use all spatial evidence, especially pointing hands, finger direction, gaze, and scene geometry.
When a deictic phrase such as 'this' or 'there' is used, infer its referent from that evidence.
Select the physical object or location, not the pointing hand itself, unless the query explicitly asks for the hand.
For a pointing gesture, estimate the ray from the finger base through the fingertip and continue beyond the
fingertip into the scene. Do not select the hand or a nearby marker merely because it is visually salient.
For a collision-free placement request, select a visibly open supporting surface near the indicated location
with clearance from hands, existing objects, and scene edges; do not select an obstacle as the destination.
Return only the requested structured result. Coordinates use (x, y), with (0, 0) at the image's top-left."""


class OpenAIVLM:
    def __init__(
        self,
        model: str,
        client: OpenAI | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        if client is not None:
            self.client = client
        else:
            kwargs: dict[str, str] = {}
            if base_url is not None:
                kwargs["base_url"] = base_url
            if api_key is not None:
                kwargs["api_key"] = api_key
            self.client = OpenAI(**kwargs)

    def _parse(self, images, prompt: str, schema: type[ParsedT]) -> ModelReply:
        if not isinstance(images, (list, tuple)):
            images = [images]
        content = [{"type": "input_text", "text": prompt}]
        content.extend(
            {
                "type": "input_image",
                "image_url": to_data_url(image),
                "detail": "high",
            }
            for image in images
        )
        request_kwargs = {}
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            request_kwargs["reasoning"] = {"effort": self.reasoning_effort}
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": content,
                },
            ],
            text_format=schema,
            # Reasoning tokens count against this budget. GPT-5.5 can consume
            # several hundred before emitting the small structured payload.
            max_output_tokens=4096,
            **request_kwargs,
        )
        if response.output_parsed is None:
            raise RuntimeError(
                f"Model response {response.id} did not contain a parsed result "
                f"(status={response.status}, incomplete={response.incomplete_details})"
            )
        usage = (
            response.usage.model_dump(mode="json")
            if response.usage is not None
            else None
        )
        return ModelReply(
            parsed=response.output_parsed, response_id=response.id, usage=usage
        )

    def direct(self, image, instruction: str, query: str | None, trial: int) -> ModelReply:
        width, height = image.size
        query_block = f"\nReferent query: {query!r}" if query else ""
        prompt = f"""Ground one referent in this image.
Instruction: {instruction!r}{query_block}
Displayed image size: width={width}, height={height} pixels.

Return exactly three distinct best pixel hypotheses, ranked best-first. Each point should lie on the visible
surface of the queried object or at the queried destination. This is independent trial {trial}; inspect the
image afresh rather than copying a stereotyped coordinate. Keep x in [0, {width - 1}] and y in [0, {height - 1}]."""
        return self._parse(image, prompt, DirectPrediction)

    def direct_modified(
        self, image, instruction: str, query: str, trial: int
    ) -> ModelReply:
        width, height = image.size
        prompt = f"""Ground one referent in this image by considering all plausible pixel hypotheses.
Instruction: {instruction!r}
Referent query: {query!r}
Displayed image size: width={width}, height={height} pixels.

Treat the apparent hand or finger direction as uncertain rather than decisive: RGB appearance can make the
pointing ray ambiguous. First identify every plausible object or destination using the instruction, object
semantics, gaze, relative layout, supporting surfaces, and scene geometry as well as the gesture. Then return
exactly five distinct best pixel hypotheses, ranked best-first, that cover the strongest genuinely different
interpretations when ambiguity remains. Prefer points on different plausible objects or separated destination
regions over five small variations around one guessed point. Each point must lie on the visible surface of its
corresponding queried object or at its corresponding queried destination, never on the hand itself. This is
independent trial {trial}; inspect the image afresh and reconsider the full hypothesis space. Keep x in
[0, {width - 1}] and y in [0, {height - 1}]."""
        return self._parse(image, prompt, DirectModifiedPrediction)

    def grid(
        self,
        clean_image,
        grid_image,
        instruction: str,
        query: str | None,
        rows: int,
        cols: int,
        trial: int,
    ) -> ModelReply:
        query_block = f"\nReferent query: {query!r}" if query else ""
        prompt = f"""Ground one referent using two views of the same scene. Image 1 is clean: use it to identify
the referent from the pointing gesture. Image 2 has the coordinate grid: use it only to read the final cells.
Instruction: {instruction!r}{query_block}
The grid has {rows} rows and {cols} columns. Cell IDs have the form R01C01; rows increase top-to-bottom
and columns left-to-right.

Return exactly three distinct valid cell IDs, ranked best-first. Select cells whose centers best represent
the queried object or destination. This is independent trial {trial}; inspect the image afresh."""
        return self._parse([clean_image, grid_image], prompt, GridPrediction)

    def pivot(
        self,
        clean_image,
        proposal_image,
        instruction: str,
        query: str | None,
        candidate_count: int,
        iteration: int,
    ) -> ModelReply:
        query_block = f"\nReferent query: {query!r}" if query else ""
        prompt = f"""Ground one referent using two views of the same scene. Image 1 is clean: first identify the
referent from the pointing gesture. Image 2 contains numbered proposals: then select the closest valid proposals.
Instruction: {instruction!r}{query_block}
There are {candidate_count} proposals, numbered 0 through {candidate_count - 1}.

Return exactly three distinct valid candidate IDs, ranked best-first. Choose the points that best lie on
the queried object or destination. This is PIVOT refinement iteration {iteration}."""
        return self._parse([clean_image, proposal_image], prompt, PivotPrediction)

    def som(
        self,
        annotated_image,
        instruction: str,
        query: str | None,
        trial: int,
        *,
        min_selections: int = 1,
        max_selections: int | None = 5,
        object_catalog: list[tuple[str, str]] | None = None,
    ) -> ModelReply:
        if max_selections is None:
            selection_instruction = "Return every genuinely plausible distinct visible object ID; there is no selection-count limit."
        elif min_selections == max_selections:
            selection_instruction = f"Return exactly {min_selections} distinct visible object IDs."
        else:
            selection_instruction = (
                f"Return between {min_selections} and {max_selections} distinct visible object IDs."
            )
        if object_catalog:
            catalog_lines = "\n".join(
                f"- ID {object_id!r}: category {category!r}"
                for object_id, category in object_catalog
            )
            catalog_block = f"""

The segmentation model also supplied this ID-to-category catalog:
{catalog_lines}
Use these categories as semantic evidence while still checking them against the image. Return the ID, not
the category name. Each catalog entry corresponds to the identically marked object in the image."""
        else:
            catalog_block = ""
        query_block = f"\nReferent query: {query!r}" if query else ""
        prompt = f"""Ground the single referent requested by the instruction in this pre-annotated Set-of-Marks image. The input already contains
bounding boxes and visible object IDs; read those marks directly and do not propose new boxes. IDs may use any
textual format. Copy each complete ID exactly as displayed, preserving all letters, numbers, and separators;
never shorten an ID to only its numeric part.
Instruction: {instruction!r}{query_block}{catalog_block}

Use spatial evidence only to form a broad candidate set, not to create large confidence differences within
that set. Pointing direction, gaze, relative layout, support relationships, and scene geometry indicate an
approximate region or direction and should recall every visibly plausible marked referent there. Because an
RGB image provides incomplete and uncertain evidence about the true 3D pointing ray, do not give one candidate
substantially higher confidence merely because its 2D alignment with the apparent gesture looks slightly
better than that of other spatially plausible candidates.

Use semantic evidence to determine confidence among the spatial candidates. Interpret the instruction and
identify the queried referent's semantic role and required affordances. Treat any Robot State as important
semantic evidence: use it to resolve references such as "it" and infer what the current action requires. In
particular, judge destinations and supporting surfaces by their physical and functional compatibility with
the object being held (for example, a large floor-standing object favors the floor over a tabletop). Use this
together with object identity and action context to distinguish manipulated objects, destinations, supporting
surfaces, tools, obstacles, and contextual objects. Strong semantic evidence may raise, lower, or eliminate a
candidate. If no semantic evidence meaningfully distinguishes several spatially plausible candidates, assign
them similar confidences rather than allowing uncertain spatial precision to separate their scores.
Calibrate confidence conservatively and do not be overconfident when the visual or semantic evidence is
ambiguous, incomplete, or weak.

{selection_instruction} Rank them best-first, covering
genuinely different plausible referents when ambiguity remains instead of concentrating on near-identical
interpretations. For each ID, provide a confidence in [0, 1] governed primarily by semantic evidence after
spatial evidence establishes plausibility; confidences need not sum to one. Return marked physical objects,
not pixels or the pointing hand unless explicitly requested. This is independent trial {trial}; inspect the
complete image afresh and reconsider the full hypothesis space."""
        schema = som_prediction_schema(min_selections, max_selections)
        return self._parse(annotated_image, prompt, schema)

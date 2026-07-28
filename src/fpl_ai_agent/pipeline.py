"""Pipeline orchestration primitives."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PipelinePhase:
    """Represents a high-level implementation phase."""

    name: str
    description: str
    is_complete: bool = False


def mark_phase_complete(phase: PipelinePhase) -> PipelinePhase:
    """Return an updated phase marked as complete."""
    return PipelinePhase(name=phase.name, description=phase.description, is_complete=True)

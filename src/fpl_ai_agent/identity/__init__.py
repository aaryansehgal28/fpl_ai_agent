"""Canonical identity resolution for teams and players."""

from fpl_ai_agent.identity.point_in_time import point_in_time_join
from fpl_ai_agent.identity.resolution import (
	IdentityMatch,
	RapidFuzzResolutionConfig,
	build_canonical_mapping_table,
	resolve_to_canonical,
)

__all__ = [
	"IdentityMatch",
	"RapidFuzzResolutionConfig",
	"build_canonical_mapping_table",
	"resolve_to_canonical",
	"point_in_time_join",
]

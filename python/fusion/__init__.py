from .shared_space import (
    NEVER_SPIKED,
    ProjectionParams,
    QgnnTriggerResult,
    fuse_to_4d,
    make_projection,
    maybe_trigger_qgnn,
    project_to_3d,
    time_since_last_spike,
)

__all__ = [
    "NEVER_SPIKED",
    "ProjectionParams",
    "QgnnTriggerResult",
    "fuse_to_4d",
    "make_projection",
    "maybe_trigger_qgnn",
    "project_to_3d",
    "time_since_last_spike",
]

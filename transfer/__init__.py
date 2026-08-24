"""Cross-network transfer support for the HyperLight hypernetwork agents.

This package holds everything that is specific to *moving a trained
hypernetwork from one road network to another*.  The agent itself
(``agent/hyperlight_ppo.py``) only gains thin hook points; all of the
transfer-specific logic lives here so it can be read, tested and reverted
as one unit.

See ``transfer/TRANSFER.md`` for the design rationale, the blockers this
package removes, and the ones it deliberately does not.
"""

from .structural import (
    FEATURE_NAMES,
    FEATURE_DIM,
    SPEC_VERSION,
    build_structural_features,
    resolve_features,
    spec_id,
    summarize_raw_features,
)
from .checkpoint import (
    TransferError,
    NODE_DEPENDENT_KEYS,
    validate_transfer_architecture,
    load_for_transfer,
    format_report,
)

__all__ = [
    'FEATURE_NAMES',
    'FEATURE_DIM',
    'SPEC_VERSION',
    'build_structural_features',
    'resolve_features',
    'spec_id',
    'summarize_raw_features',
    'TransferError',
    'NODE_DEPENDENT_KEYS',
    'validate_transfer_architecture',
    'load_for_transfer',
    'format_report',
]

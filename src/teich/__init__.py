from __future__ import annotations

__version__ = "0.2.4"

from .audit import SFTAuditReport, audit_sft_dataset
from .config import Config, load_config
from .converter import (
    TrainingExample,
    convert_trace_to_training_example,
    convert_traces_to_training_data,
    detect_trace_type,
)
from .formatter import PrepareReport, RowContextFit, mask_data, preview_sft_example, row_fits_context
from .loader import load_traces, trace_is_complete
from .prepare import prepare_data
from .swift import convert_to_ms_swift, to_ms_swift_messages, to_ms_swift_row
from .tool_schema import ToolCallValidationReport, validate_tool_calls

__all__ = [
    "PrepareReport",
    "RowContextFit",
    "SFTAuditReport",
    "ToolCallValidationReport",
    "Config",
    "TrainingExample",
    "audit_sft_dataset",
    "convert_to_ms_swift",
    "convert_trace_to_training_example",
    "convert_traces_to_training_data",
    "detect_trace_type",
    "to_ms_swift_messages",
    "to_ms_swift_row",
    "load_traces",
    "load_config",
    "mask_data",
    "prepare_data",
    "preview_sft_example",
    "row_fits_context",
    "trace_is_complete",
    "validate_tool_calls",
]

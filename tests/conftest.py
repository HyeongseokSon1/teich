"""Pytest configuration."""

import sys
from pathlib import Path

# Add src directory to Python path for imports
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

"""Layerwise Debate Tree (LDT).

LDT builds a shared reasoning tree one layer at a time. Each frontier node asks
all agents for one next hop, merges equivalent proposals, detects local
conflicts, debates only conflict groups, and keeps a bounded beam of reliable
children.
"""

from src.ldt.algorithm import run

__all__ = ["run"]

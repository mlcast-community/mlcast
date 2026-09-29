"""Shared rich console for user-facing CLI output.

Writes to stderr so stdout stays clean for piping. Both the rich progress
display and the loguru logs target stderr, and no logs are emitted during
the progress live region, so they don't clobber each other.
"""

from __future__ import annotations

from rich.console import Console

console = Console(stderr=True)

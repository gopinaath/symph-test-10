"""Path safety utilities for workspace management."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_MAX_SEGMENT_LEN = 255


@dataclass
class SafeResolveError:
    """Describes why a path failed safe resolution."""

    message: str


def safe_resolve(path: str | Path, root: str | Path) -> Path | SafeResolveError:
    """Resolve *path* segment-by-segment ensuring the result stays under *root*.

    Symlinks are resolved at each segment so that a symlink pointing outside
    the root is caught immediately rather than only after full resolution.

    Returns the resolved :class:`Path` on success or a :class:`SafeResolveError`
    on failure.
    """
    root = Path(root).resolve()

    if not root.is_dir():
        return SafeResolveError(f"root is not a directory: {root}")

    target = Path(path)

    # If *path* is relative, anchor it under root.
    if not target.is_absolute():
        target = root / target

    # Resolve the target path (resolving ..) to see where it actually points,
    # but without following symlinks yet (we do that segment-by-segment below).
    # For existing paths we use resolve(); for non-existing we use absolute + normpath.
    import os

    normalized = Path(os.path.normpath(str(target)))

    # Determine the segments *below* root that we actually need to check.
    try:
        suffix_parts = normalized.relative_to(root).parts
    except ValueError:
        return SafeResolveError(f"path escapes root: {normalized} is not under {root}")

    current = root
    for segment in suffix_parts:
        if len(segment) > _MAX_SEGMENT_LEN:
            return SafeResolveError(f"path segment exceeds {_MAX_SEGMENT_LEN} chars: {segment!r}")

        candidate = current / segment

        # Resolve the candidate so far (resolves symlinks on this segment).
        try:
            resolved = candidate.resolve()
        except OSError as exc:
            return SafeResolveError(str(exc))

        # If the candidate exists and is a symlink, the resolved path must
        # still be under root.  For non-existing trailing segments we just
        # keep going (the parent must be under root though).
        if (resolved.exists() or candidate.is_symlink()) and not _is_under(resolved, root):
            return SafeResolveError(f"path escapes root: {resolved} is not under {root}")

        current = resolved

    # Final check on the fully resolved path.
    if not _is_under(current, root):
        return SafeResolveError(f"path escapes root: {current} is not under {root}")

    return current


def _is_under(child: Path, root: Path) -> bool:
    """Return True if *child* is equal to or a descendant of *root*."""
    try:
        child.relative_to(root)
        return True
    except ValueError:
        return False

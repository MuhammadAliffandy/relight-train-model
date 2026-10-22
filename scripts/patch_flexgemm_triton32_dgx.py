#!/usr/bin/env python3
"""Remove FlexGEMM JIT annotations that Triton 3.2 miscompiles."""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import shutil
import tempfile
from pathlib import Path


UNSUPPORTED_ANNOTATION = re.compile(
    r"(?:tl\.pointer_type(?:\s*\|\s*None)?|tl\.(?:tensor|const)\s*\|\s*None)\Z"
)
BACKUP_SUFFIX = ".moge_triton32.bak"


def remove_unsupported_annotations(source: bytes, path: Path) -> tuple[bytes, int]:
    tree = ast.parse(source, filename=str(path))
    text = source.decode("utf-8")
    lines = source.splitlines(keepends=True)
    line_starts = [0]
    for line in lines:
        line_starts.append(line_starts[-1] + len(line))

    edits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.arg) or node.annotation is None:
            continue
        annotation = ast.get_source_segment(text, node.annotation)
        if annotation is None or not UNSUPPORTED_ANNOTATION.fullmatch(annotation.strip()):
            continue
        start = line_starts[node.lineno - 1] + node.col_offset + len(node.arg.encode("utf-8"))
        end = line_starts[node.annotation.end_lineno - 1] + node.annotation.end_col_offset
        declaration = source[start:end].decode("utf-8")
        if not declaration.lstrip().startswith(":") or not UNSUPPORTED_ANNOTATION.fullmatch(
            declaration.split(":", 1)[1].strip()
        ):
            raise RuntimeError(f"Unexpected Triton annotation in {path}:{node.lineno}")
        edits.append((start, end))

    patched = source
    for start, end in sorted(edits, reverse=True):
        patched = patched[:start] + patched[end:]
    if edits:
        ast.parse(patched, filename=str(path))
    return patched, len(edits)


def main() -> None:
    import triton

    if not triton.__version__.startswith("3.2."):
        print(f"FlexGEMM pointer patch skipped for Triton {triton.__version__}", flush=True)
        return

    spec = importlib.util.find_spec("flex_gemm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("flex_gemm is not installed in this Python environment")
    kernel_root = Path(next(iter(spec.submodule_search_locations))) / "kernels" / "triton"
    if not kernel_root.is_dir():
        raise RuntimeError(f"FlexGEMM Triton kernels not found: {kernel_root}")

    changed_files = total_annotations = 0
    for path in sorted(kernel_root.rglob("*.py")):
        original = path.read_bytes()
        patched, count = remove_unsupported_annotations(original, path)
        if not count:
            continue
        backup = path.with_name(path.name + BACKUP_SUFFIX)
        if not backup.exists():
            shutil.copy2(path, backup)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as temporary:
            temporary.write(patched)
            temporary_path = Path(temporary.name)
        try:
            shutil.copymode(path, temporary_path)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        changed_files += 1
        total_annotations += count

    print(
        f"FlexGEMM/Triton {triton.__version__}: removed {total_annotations} "
        f"unsupported annotations in {changed_files} files; originals saved as *{BACKUP_SUFFIX}",
        flush=True,
    )


if __name__ == "__main__":
    main()

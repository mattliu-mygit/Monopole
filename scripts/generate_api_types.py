"""Generate or verify committed OpenAPI and TypeScript transport contracts."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from weave_agent_signals.api import create_app

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "frontend" / "src" / "generated"


def _generate(directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    schema_path = directory / "openapi.json"
    types_path = directory / "api.ts"
    schema = create_app(frontend_dist=Path("__api_generation_no_frontend__")).openapi()
    schema_path.write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            str(ROOT / "frontend" / "node_modules" / ".bin" / "openapi-typescript"),
            str(schema_path),
            "-o",
            str(types_path),
        ],
        check=True,
        cwd=ROOT / "frontend",
    )
    return schema_path, types_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        _generate(GENERATED)
        return
    with tempfile.TemporaryDirectory(prefix="monopole-api-") as temporary:
        generated = _generate(Path(temporary))
        stale = [
            path.name
            for path in generated
            if not (GENERATED / path.name).exists()
            or path.read_bytes() != (GENERATED / path.name).read_bytes()
        ]
    if stale:
        raise SystemExit(f"generated API contract is stale: {', '.join(stale)}")


if __name__ == "__main__":
    main()

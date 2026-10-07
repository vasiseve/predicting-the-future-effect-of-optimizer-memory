from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import GridConfig
from .runner import run_grid, summarize_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RTTP full-grid experiment.")
    parser.add_argument("--config", type=str, default="", help="Optional JSON config path.")
    parser.add_argument("--summary-only", action="store_true", help="Only rebuild summaries from an existing output root.")
    parser.add_argument("--root", type=str, default="", help="Existing output root for --summary-only.")
    parser.add_argument("--paper-material-dir", type=str, default="", help="Optional directory to receive paper-facing export files.")
    parser.add_argument("--write-default-config", type=str, default="", help="Write the default config JSON and exit.")
    args = parser.parse_args()

    if args.write_default_config:
        GridConfig.from_env().to_json(args.write_default_config)
        print(f"wrote {args.write_default_config}")
        return

    if args.summary_only:
        if not args.root:
            raise SystemExit("--summary-only requires --root")
        summarize_outputs(Path(args.root))
        return

    cfg = GridConfig.from_json(args.config) if args.config else GridConfig.from_env()
    if args.paper_material_dir:
        cfg = GridConfig(**{**cfg.__dict__, "paper_material_dir": args.paper_material_dir})
    root = run_grid(cfg)
    print(json.dumps({"output_root": str(root)}, indent=2))


if __name__ == "__main__":
    main()

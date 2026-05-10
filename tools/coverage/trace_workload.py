#!/usr/bin/env python3
"""Trace a native CUDA workload with Nsight Systems and join API use to coverage."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import api_coverage  # noqa: E402


API_NAME_RE = re.compile(r"^(?:cu|cuda|__cuda|nccl|nvml|nvrtc|cublas|cublasLt|cudnn)[A-Za-z0-9_]*$")
IGNORE_MARKERS = {"cuBLAS", "cuDNN", "NCCL", "CUDA"}


@dataclass
class ObservedApi:
    raw_names: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    calls: int = 0
    total_ns: int = 0
    nonzero_returns: int = 0


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "workload"


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute(
        "select 1 from sqlite_master where type='table' and name=?",
        (table,),
    ).fetchone()
    return row is not None


def add_observation(
    observed: dict[str, ObservedApi],
    raw_name: str,
    source: str,
    calls: int = 1,
    total_ns: int = 0,
    nonzero_returns: int = 0,
) -> None:
    normalized = api_coverage.normalize_api_name(raw_name)
    if normalized in IGNORE_MARKERS:
        return
    if not API_NAME_RE.match(normalized):
        return
    item = observed[normalized]
    item.raw_names.add(raw_name)
    item.sources.add(source)
    item.calls += calls
    item.total_ns += total_ns
    item.nonzero_returns += nonzero_returns


def extract_runtime_apis(con: sqlite3.Connection, observed: dict[str, ObservedApi]) -> None:
    if not (table_exists(con, "CUPTI_ACTIVITY_KIND_RUNTIME") and table_exists(con, "StringIds")):
        return
    query = """
        select s.value,
               count(*) as calls,
               coalesce(sum(r.end - r.start), 0) as total_ns,
               sum(case when r.returnValue != 0 then 1 else 0 end) as nonzero_returns
          from CUPTI_ACTIVITY_KIND_RUNTIME r
          join StringIds s on s.id = r.nameId
         group by s.value
    """
    for raw_name, calls, total_ns, nonzero_returns in con.execute(query):
        add_observation(
            observed,
            raw_name,
            "cupti-runtime",
            calls=int(calls or 0),
            total_ns=int(total_ns or 0),
            nonzero_returns=int(nonzero_returns or 0),
        )


def extract_nvtx_api_markers(con: sqlite3.Connection, observed: dict[str, ObservedApi]) -> None:
    if not table_exists(con, "NVTX_EVENTS"):
        return

    string_ids = {}
    if table_exists(con, "StringIds"):
        string_ids = dict(con.execute("select id, value from StringIds"))

    for text, text_id in con.execute("select text, textId from NVTX_EVENTS"):
        for candidate in (text, string_ids.get(text_id)):
            if candidate:
                name = candidate.split("(", 1)[0].strip()
                add_observation(observed, name, "nvtx", calls=1)


def extract_observed_apis(sqlite_path: Path) -> dict[str, ObservedApi]:
    observed: dict[str, ObservedApi] = defaultdict(ObservedApi)
    with sqlite3.connect(sqlite_path) as con:
        extract_runtime_apis(con, observed)
        extract_nvtx_api_markers(con, observed)
    return dict(observed)


def run_nsys(args: argparse.Namespace, trace_dir: Path) -> tuple[Path, Path]:
    nsys = shutil.which("nsys")
    if nsys is None:
        raise SystemExit("nsys not found in PATH")

    trace_dir.mkdir(parents=True, exist_ok=True)
    output_base = trace_dir / sanitize_name(args.name)
    cmd = [
        nsys,
        "profile",
        "--force-overwrite=true",
        f"--trace={args.trace}",
        f"--cuda-trace-all-apis={'true' if args.cuda_trace_all_apis else 'false'}",
        "--export=sqlite",
        "--stats=false",
        "-o",
        str(output_base),
        *args.command,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as err:
        raise SystemExit(f"nsys failed with exit code {err.returncode}") from None
    sqlite_path = output_base.with_suffix(".sqlite")
    report_path = output_base.with_suffix(".nsys-rep")
    if not sqlite_path.exists():
        raise SystemExit(f"Nsight Systems did not create expected SQLite export: {sqlite_path}")
    return sqlite_path, report_path


def build_report(
    repo_root: Path,
    sqlite_path: Path,
    report_path: Path | None,
    command: list[str],
) -> dict[str, object]:
    coverage = api_coverage.load_coverage(repo_root)
    observed = extract_observed_apis(sqlite_path)

    apis = []
    summary = defaultdict(int)
    by_family = defaultdict(lambda: defaultdict(int))
    for name in sorted(observed):
        item = observed[name]
        status = api_coverage.lookup_api(coverage, name)
        if status is None:
            family = "unknown"
            category = "unknown"
            coverage_status = "untracked"
        else:
            family = status.family
            category = status.category
            coverage_status = status.status
        summary[coverage_status] += 1
        by_family[family][coverage_status] += 1
        apis.append(
            {
                "family": family,
                "name": name,
                "category": category,
                "coverage_status": coverage_status,
                "calls": item.calls,
                "total_ns": item.total_ns,
                "nonzero_returns": item.nonzero_returns,
                "sources": sorted(item.sources),
                "raw_names": sorted(item.raw_names),
            }
        )

    return {
        "metadata": {
            "command": command,
            "sqlite": str(sqlite_path),
            "nsys_report": str(report_path) if report_path else None,
        },
        "summary": {
            "observed_apis": len(apis),
            "by_status": dict(sorted(summary.items())),
            "by_family": {family: dict(sorted(counts.items())) for family, counts in sorted(by_family.items())},
        },
        "apis": apis,
    }


def render_markdown(report: dict[str, object]) -> str:
    metadata = report["metadata"]
    summary = report["summary"]
    apis = report["apis"]
    assert isinstance(metadata, dict)
    assert isinstance(summary, dict)
    assert isinstance(apis, list)

    lines = [
        "# Workload API Trace",
        "",
        f"- Command: `{' '.join(metadata.get('command') or [])}`",
        f"- SQLite: `{metadata.get('sqlite')}`",
        f"- Nsight report: `{metadata.get('nsys_report')}`",
        f"- Observed APIs: {summary.get('observed_apis')}",
        "",
        "Nsight traces include CUDA calls made inside CUDA libraries. For libraries that GPU-Remoting remotes at the library boundary, internal driver/runtime calls are prioritization signals, not automatically client-surface requirements.",
        "",
        "## Status",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    by_status = summary.get("by_status") or {}
    assert isinstance(by_status, dict)
    for status, count in by_status.items():
        lines.append(f"| {status} | {count} |")

    lines.extend(["", "## APIs", "", "| Family | API | Status | Calls | Sources |", "|---|---|---|---:|---|"])
    for api in apis:
        assert isinstance(api, dict)
        lines.append(
            f"| {api['family']} | `{api['name']}` | {api['coverage_status']} | "
            f"{api['calls']} | {', '.join(api['sources'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--name", default="workload")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--trace", default="cuda,nvtx,nccl,cublas,cudnn")
    parser.add_argument("--no-cuda-trace-all-apis", dest="cuda_trace_all_apis", action="store_false")
    parser.add_argument("--sqlite", type=Path, help="Parse an existing Nsight Systems SQLite export instead of profiling")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Workload command after --")
    parser.set_defaults(cuda_trace_all_apis=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir or repo_root / "target" / "coverage" / "traces"
    trace_dir = output_dir / sanitize_name(args.name)

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    args.command = command

    if args.sqlite:
        sqlite_path = args.sqlite.resolve()
        inferred_report = sqlite_path.with_suffix(".nsys-rep")
        report_path = inferred_report if inferred_report.exists() else None
        command = []
    else:
        if not command:
            raise SystemExit("expected workload command after --, or pass --sqlite")
        sqlite_path, report_path = run_nsys(args, trace_dir)

    report = build_report(repo_root, sqlite_path, report_path, command)
    write_json(trace_dir / "api-trace.json", report)
    write_markdown(trace_dir / "api-trace.md", report)
    print(trace_dir / "api-trace.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

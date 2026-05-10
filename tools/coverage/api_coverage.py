#!/usr/bin/env python3
"""Report CUDA-family API coverage from generated bindings and hook lists."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FAMILIES = ("cuda", "cudart", "nccl", "nvml", "nvrtc", "cublas", "cublasLt", "cudnn")

BINDING_RE = re.compile(r"\bpub\s+fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
HOOK_RE = re.compile(r"^\s*fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
UNIMPLEMENT_RE = re.compile(r'extern\s+"C"\s+fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(')
RUNTIME_ABI_SUFFIX_RE = re.compile(r"_v\d{3,}$")

ALIASES = {
    "cuDeviceGetUuid": "cuDeviceGetUuid_v2",
    "cuDevicePrimaryCtxRelease": "cuDevicePrimaryCtxRelease_v2",
}


@dataclass(frozen=True)
class ApiStatus:
    family: str
    name: str
    status: str
    category: str
    in_bindings: bool
    in_hooks: bool
    in_placeholders: bool


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_names(path: Path, pattern: re.Pattern[str]) -> set[str]:
    if not path.exists():
        return set()
    return set(pattern.findall(path.read_text(encoding="utf-8")))


def normalize_api_name(name: str) -> str:
    """Normalize profiler ABI names to public API names when possible."""
    name = RUNTIME_ABI_SUFFIX_RE.sub("", name)
    return ALIASES.get(name, name)


def family_for_api(name: str) -> str | None:
    normalized = normalize_api_name(name)
    checks = (
        ("cublasLt", ("cublasLt",)),
        ("cublas", ("cublas",)),
        ("cudnn", ("cudnn",)),
        ("nccl", ("nccl",)),
        ("nvml", ("nvml",)),
        ("nvrtc", ("nvrtc",)),
        ("cudart", ("cuda", "__cuda")),
        ("cuda", ("cu",)),
    )
    for family, prefixes in checks:
        if normalized.startswith(prefixes):
            return family
    return None


def category_for_api(name: str, family: str) -> str:
    if family == "nccl":
        if "Comm" in name:
            return "communicator"
        if any(term in name for term in ("All", "Reduce", "Gather", "Broadcast", "Bcast", "Send", "Recv")):
            return "collective"
        return "nccl"
    if family == "nvrtc":
        return "runtime-compilation"
    if family == "nvml":
        return "management"
    if family in ("cublas", "cublasLt"):
        return "blas"
    if family == "cudnn":
        return "dnn"

    rules = (
        ("graphs", ("Graph", "UserObject")),
        ("modules/kernels", ("Module", "Library", "Kernel", "Launch", "Func", "Occupancy", "FatBinary", "RegisterFunction", "GetSymbol")),
        ("memory", ("Mem", "Malloc", "Free", "Memcpy", "Memset", "HostAlloc", "HostRegister", "Pointer", "Array", "Managed", "ExternalMemory")),
        ("streams/events", ("Stream", "Event", "Synchronize", "Wait", "LaunchHostFunc")),
        ("device/context", ("Device", "Ctx", "PrimaryCtx", "Init", "SetDevice", "GetDevice")),
        ("peer/ipc", ("Peer", "Ipc", "P2P")),
        ("textures/surfaces", ("Texture", "Tex", "Surface", "Surf")),
        ("errors/version", ("Error", "Version", "GetProcAddress", "DriverEntryPoint")),
    )
    for category, terms in rules:
        if any(term in name for term in terms):
            return category
    return "other"


def load_coverage(repo_root: Path) -> dict[str, dict[str, object]]:
    coverage: dict[str, dict[str, object]] = {}
    for family in FAMILIES:
        bindings = parse_names(repo_root / "cudasys" / "src" / "bindings" / "funcs" / f"{family}.rs", BINDING_RE)
        hooks = parse_names(repo_root / "cudasys" / "src" / "hooks" / f"{family}.rs", HOOK_RE)
        placeholders = parse_names(repo_root / "client" / "src" / "hijack" / f"{family}_unimplement.rs", UNIMPLEMENT_RE)

        entries: dict[str, ApiStatus] = {}
        for name in sorted(bindings | hooks | placeholders):
            in_bindings = name in bindings
            in_hooks = name in hooks
            in_placeholders = name in placeholders
            if in_bindings and in_hooks:
                status = "implemented"
            elif in_bindings and in_placeholders:
                status = "placeholder"
            elif in_bindings:
                status = "unknown"
            elif in_hooks:
                status = "extra-hook"
            else:
                status = "extra-placeholder"
            entries[name] = ApiStatus(
                family=family,
                name=name,
                status=status,
                category=category_for_api(name, family),
                in_bindings=in_bindings,
                in_hooks=in_hooks,
                in_placeholders=in_placeholders,
            )

        coverage[family] = {
            "bindings": bindings,
            "hooks": hooks,
            "placeholders": placeholders,
            "entries": entries,
        }
    return coverage


def lookup_api(coverage: dict[str, dict[str, object]], api_name: str) -> ApiStatus | None:
    name = normalize_api_name(api_name)
    family = family_for_api(name)
    if family is None:
        return None
    entries = coverage[family]["entries"]
    assert isinstance(entries, dict)
    status = entries.get(name)
    if status is not None:
        return status
    return ApiStatus(
        family=family,
        name=name,
        status="untracked",
        category=category_for_api(name, family),
        in_bindings=False,
        in_hooks=False,
        in_placeholders=False,
    )


def iter_statuses(coverage: dict[str, dict[str, object]]) -> Iterable[ApiStatus]:
    for family in FAMILIES:
        entries = coverage[family]["entries"]
        assert isinstance(entries, dict)
        yield from entries.values()


def family_summary(coverage: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for family in FAMILIES:
        bindings = coverage[family]["bindings"]
        hooks = coverage[family]["hooks"]
        placeholders = coverage[family]["placeholders"]
        assert isinstance(bindings, set)
        assert isinstance(hooks, set)
        assert isinstance(placeholders, set)
        implemented = len(bindings & hooks)
        placeholder = len(bindings & placeholders)
        unknown = len(bindings - hooks - placeholders)
        rows.append(
            {
                "family": family,
                "bindings": len(bindings),
                "implemented": implemented,
                "placeholders": placeholder,
                "unknown": unknown,
                "extra_hooks": len(hooks - bindings),
                "extra_placeholders": len(placeholders - bindings),
                "coverage_percent": round(100.0 * implemented / len(bindings), 1) if bindings else 0.0,
            }
        )
    return rows


def category_summary(coverage: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    totals: dict[tuple[str, str], dict[str, int]] = {}
    for status in iter_statuses(coverage):
        if not status.in_bindings:
            continue
        key = (status.family, status.category)
        bucket = totals.setdefault(key, {"bindings": 0, "implemented": 0, "placeholders": 0, "unknown": 0})
        bucket["bindings"] += 1
        if status.status == "implemented":
            bucket["implemented"] += 1
        elif status.status == "placeholder":
            bucket["placeholders"] += 1
        else:
            bucket["unknown"] += 1
    rows = []
    for (family, category), bucket in sorted(totals.items()):
        bindings = bucket["bindings"]
        rows.append(
            {
                "family": family,
                "category": category,
                **bucket,
                "coverage_percent": round(100.0 * bucket["implemented"] / bindings, 1) if bindings else 0.0,
            }
        )
    return rows


def to_json(coverage: dict[str, dict[str, object]], include_api_list: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "families": family_summary(coverage),
        "categories": category_summary(coverage),
    }
    if include_api_list:
        payload["apis"] = [
            {
                "family": status.family,
                "name": status.name,
                "status": status.status,
                "category": status.category,
                "in_bindings": status.in_bindings,
                "in_hooks": status.in_hooks,
                "in_placeholders": status.in_placeholders,
            }
            for status in sorted(iter_statuses(coverage), key=lambda s: (s.family, s.name))
        ]
    return payload


def render_markdown(coverage: dict[str, dict[str, object]], include_api_list: bool) -> str:
    lines = [
        "# CUDA API Coverage",
        "",
        "Generated from `cudasys/src/bindings/funcs/*.rs`, `cudasys/src/hooks/*.rs`, and generated unimplemented placeholders.",
        "",
        "## Families",
        "",
        "| Family | Bindings | Implemented | Placeholders | Unknown | Extra hooks | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in family_summary(coverage):
        lines.append(
            f"| {row['family']} | {row['bindings']} | {row['implemented']} | {row['placeholders']} | "
            f"{row['unknown']} | {row['extra_hooks']} | {row['coverage_percent']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Categories",
            "",
            "| Family | Category | Bindings | Implemented | Placeholders | Unknown | Coverage |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in category_summary(coverage):
        lines.append(
            f"| {row['family']} | {row['category']} | {row['bindings']} | {row['implemented']} | "
            f"{row['placeholders']} | {row['unknown']} | {row['coverage_percent']:.1f}% |"
        )

    if include_api_list:
        lines.extend(["", "## APIs", "", "| Family | API | Category | Status |", "|---|---|---|---|"])
        for status in sorted(iter_statuses(coverage), key=lambda s: (s.family, s.name)):
            lines.append(f"| {status.family} | `{status.name}` | {status.category} | {status.status} |")

    lines.append("")
    return "\n".join(lines)


def render_csv(coverage: dict[str, dict[str, object]]) -> str:
    import io

    out = io.StringIO()
    writer = csv.DictWriter(
        out,
        fieldnames=("family", "name", "category", "status", "in_bindings", "in_hooks", "in_placeholders"),
    )
    writer.writeheader()
    for status in sorted(iter_statuses(coverage), key=lambda s: (s.family, s.name)):
        writer.writerow(
            {
                "family": status.family,
                "name": status.name,
                "category": status.category,
                "status": status.status,
                "in_bindings": status.in_bindings,
                "in_hooks": status.in_hooks,
                "in_placeholders": status.in_placeholders,
            }
        )
    return out.getvalue()


def write_output(path: str, content: str) -> None:
    if path == "-":
        print(content, end="")
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--format", choices=("markdown", "json", "csv"), default="markdown")
    parser.add_argument("--output", default="-")
    parser.add_argument("--include-api-list", action="store_true")
    args = parser.parse_args()

    coverage = load_coverage(args.repo_root.resolve())
    if args.format == "markdown":
        content = render_markdown(coverage, args.include_api_list)
    elif args.format == "json":
        content = json.dumps(to_json(coverage, args.include_api_list), indent=2, sort_keys=True) + "\n"
    else:
        content = render_csv(coverage)
    write_output(args.output, content)
    return 0


if __name__ == "__main__":
    sys.exit(main())

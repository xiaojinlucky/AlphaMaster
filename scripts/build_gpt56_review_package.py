"""生成不含 Git 历史、数据、模型与凭证的 GPT-5.6 源码审查包。"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_BASE = "093e3172a24f1e4f7621e590b2641c4c5c61e83d"
DEFAULT_GITLEAKS = Path(
    r"C:\Users\Administrator\.codex-shared\tools\gitleaks\v8.30.1\gitleaks.exe"
)
ALLOWLIST_VERSION = "alphamaster_gpt56_review_allowlist_v2"
MAX_FILE_BYTES = 1024 * 1024

SOURCE_DIRECTORIES = {
    "backtest_viz",
    "data_pipeline",
    "execution",
    "lord",
    "model_core",
    "scripts",
    "strategy_manager",
    "tests",
    "utils",
    "web",
}
EXPLICIT_FILES = {
    ".gitattributes",
    ".gitignore",
    ".env.example",
    ".codex/config.toml",
    "LICENSE",
    "README.md",
    "CONTEXT.md",
    "lessons.md",
    "web_settings.example.json",
    "docs/GPT56_PRO_EXTENDED_HANDOFF.md",
    "docs/GPT56_PRO_EXTENDED_PROMPT.md",
    "docs/GPT56_REVIEW_ALLOWLIST.txt",
    "docs/AlphaMaster新手使用指南.md",
    "docs/slurm-deployment-design.md",
    "strategies/README.md",
}
ALLOWED_SUFFIXES = {
    ".py",
    ".js",
    ".css",
    ".html",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".in",
    ".lock",
    ".bat",
    ".ps1",
}
GENERATED_ROOT_FILES = {
    "CONTEXT_GPT56.md",
    "PROMPT_GPT56.md",
    "SELECTED_WORKTREE_STATUS.txt",
    "EXCLUDED_WORKTREE_FILES.txt",
    "TRACKED_DIFF.patch",
    "SELECTED_UNTRACKED_FILES.tsv",
    "PACKAGE_METADATA.json",
    "SOURCE_MANIFEST.tsv",
}
FORBIDDEN_NAMES = {
    ".env",
    "web_settings.json",
    "ai_analysis_history.json",
    "portfolio_state.json",
    "verification_results.json",
    "train_elapsed.json",
    "stop_signal",
    "train.pid",
}
FORBIDDEN_PARTS = {
    ".git",
    ".github",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".hypothesis",
    "data",
    "local_data",
    "local_runs",
    "runs",
    "published_training",
    "checkpoints",
    "logs",
    "scratch",
    "tmp",
    "tmp_downloads",
    "backtest_output",
}
FORBIDDEN_SUFFIXES = {
    ".zip",
    ".7z",
    ".rar",
    ".tar",
    ".tgz",
    ".gz",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".parquet",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".duckdb",
    ".log",
    ".pem",
    ".key",
    ".crt",
    ".cer",
    ".pfx",
    ".p12",
    ".exe",
    ".dll",
    ".pyd",
    ".so",
    ".dylib",
}


def _run(
    command: list[str],
    *,
    cwd: Path = PROJECT_ROOT,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
        capture_output=True,
        text=text,
        encoding="utf-8" if text else None,
    )


def _git_text(*args: str) -> str:
    return _run(["git", *args]).stdout


def _git_nul_paths(*args: str) -> set[str]:
    raw = _run(["git", *args], text=False).stdout
    return {
        item.decode("utf-8")
        for item in raw.split(b"\0")
        if item
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _is_forbidden(relative: str) -> bool:
    pure = PurePosixPath(relative)
    parts = {part.casefold() for part in pure.parts}
    name = pure.name.casefold()
    runtime_json = pure.suffix.casefold() == ".json" and (
        name.startswith("training_history") or name.startswith("training_time")
    )
    return (
        bool(parts & FORBIDDEN_PARTS)
        or name in FORBIDDEN_NAMES
        or runtime_json
        or pure.suffix.casefold() in FORBIDDEN_SUFFIXES
    )


def _matches_allowlist(relative: str) -> bool:
    if relative in EXPLICIT_FILES:
        return True
    pure = PurePosixPath(relative)
    if len(pure.parts) == 1:
        name = pure.name
        return (
            (name.endswith(".py") or name.endswith(".bat"))
            or fnmatch.fnmatchcase(name, "requirements*.txt")
            or fnmatch.fnmatchcase(name, "requirements*.in")
            or fnmatch.fnmatchcase(name, "requirements*.lock")
        )
    return pure.parts[0] in SOURCE_DIRECTORIES and pure.suffix.casefold() in ALLOWED_SUFFIXES


def _git_visible_paths() -> set[str]:
    return _git_nul_paths("ls-files", "-z", "--cached", "--others", "--exclude-standard")


def _selected_sources() -> list[tuple[str, Path]]:
    visible = _git_visible_paths()
    missing = sorted(relative for relative in EXPLICIT_FILES if relative not in visible)
    if missing:
        raise RuntimeError(f"交接白名单文件缺失或被忽略: {missing[0]}")

    selected: list[tuple[str, Path]] = []
    for relative in sorted(path for path in visible if _matches_allowlist(path)):
        if _is_forbidden(relative):
            raise RuntimeError(f"白名单候选命中禁止路径: {relative}")
        path = PROJECT_ROOT.joinpath(*PurePosixPath(relative).parts)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"白名单候选不是普通文件: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise RuntimeError(f"白名单候选超过 1 MiB: {relative}")
        selected.append((relative, path))
    return selected


def _pathspec_args(selected: list[tuple[str, Path]]) -> list[str]:
    return [relative for relative, _path in selected]


def _source_fingerprint(selected: list[tuple[str, Path]]) -> dict[str, dict[str, int | str]]:
    return {
        relative: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for relative, path in selected
    }


def _snapshot(
    selected: list[tuple[str, Path]],
    review_base: str,
) -> dict[str, object]:
    pathspec = _pathspec_args(selected)
    full_status = _git_text("status", "--short", "--untracked-files=all")
    selected_status = _git_text(
        "status",
        "--short",
        "--untracked-files=all",
        "--",
        *pathspec,
    )
    tracked_diff = _git_text(
        "diff",
        "--no-ext-diff",
        "--src-prefix=a/",
        "--dst-prefix=b/",
        review_base,
        "--",
        *pathspec,
    )
    untracked = _git_nul_paths("ls-files", "-z", "--others", "--exclude-standard")
    selected_untracked = sorted(untracked & set(pathspec))
    return {
        "head": _git_text("rev-parse", "HEAD").strip(),
        "branch": _git_text("branch", "--show-current").strip(),
        "full_status": full_status,
        "selected_status": selected_status,
        "tracked_diff": tracked_diff,
        "selected_untracked": selected_untracked,
        "sources": _source_fingerprint(selected),
    }


def _snapshot_digest(snapshot: dict[str, object]) -> str:
    canonical = json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return _sha256_bytes(canonical)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _excluded_status(full_status: str, selected_status: str) -> str:
    selected_lines = set(selected_status.splitlines())
    excluded = [line for line in full_status.splitlines() if line not in selected_lines]
    header = (
        "# 以下工作区状态不属于 AlphaMaster GPT-5.6 白名单，未复制内容。\n"
        "# 其中可能包含其他线程文件；这里只保留文件名和排除事实，避免证据悬空。\n"
    )
    return header + ("\n".join(excluded) + "\n" if excluded else "(无)\n")


def _selected_untracked_tsv(
    selected_untracked: list[str],
    sources: dict[str, dict[str, int | str]],
) -> str:
    rows = ["sha256\tbytes\tpath"]
    for relative in selected_untracked:
        row = sources[relative]
        rows.append(f"{row['sha256']}\t{row['bytes']}\t{relative}")
    return "\n".join(rows) + "\n"


def _validate_review_base(review_base: str, head: str) -> None:
    if len(review_base) != 40 or any(char not in "0123456789abcdef" for char in review_base):
        raise RuntimeError("review base 必须是 40 位小写 commit SHA")
    _run(["git", "cat-file", "-e", f"{review_base}^{{commit}}"])
    ancestor = _run(
        ["git", "merge-base", "--is-ancestor", review_base, head],
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("review base 不是当前 HEAD 的祖先")


def _run_gitleaks(gitleaks: Path, package_dir: Path) -> str:
    if not gitleaks.is_file():
        raise RuntimeError(f"gitleaks 不可用: {gitleaks}")
    version = _run([str(gitleaks), "version"]).stdout.strip()
    scan = _run(
        [str(gitleaks), "dir", str(package_dir), "--no-banner", "--redact"],
        check=False,
    )
    if scan.returncode != 0:
        detail = (scan.stdout + scan.stderr).strip()
        raise RuntimeError(f"gitleaks 扫描失败或命中: {detail}")
    return version


def _manifest_rows(package_dir: Path) -> list[tuple[str, int, str]]:
    rows: list[tuple[str, int, str]] = []
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file() or path.name == "SOURCE_MANIFEST.tsv":
            continue
        relative = path.relative_to(package_dir).as_posix()
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise RuntimeError(f"包成员超过 1 MiB: {relative}")
        rows.append((_sha256(path), size, relative))
    return rows


def _verify_manifest(package_dir: Path) -> None:
    manifest_path = package_dir / "SOURCE_MANIFEST.tsv"
    declared: dict[str, tuple[str, int]] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, raw_size, relative = line.split("\t", 2)
        if relative in declared:
            raise RuntimeError(f"manifest 重复路径: {relative}")
        declared[relative] = (digest, int(raw_size))
    actual = {
        path.relative_to(package_dir).as_posix(): (_sha256(path), path.stat().st_size)
        for path in package_dir.rglob("*")
        if path.is_file() and path.name != "SOURCE_MANIFEST.tsv"
    }
    if declared != actual:
        raise RuntimeError("manifest 与实际包成员不一致")


def _verify_zip(zip_path: Path, package_dir: Path, package_name: str) -> None:
    expected = {
        f"{package_name}/{path.relative_to(package_dir).as_posix()}": (
            _sha256(path),
            path.stat().st_size,
        )
        for path in package_dir.rglob("*")
        if path.is_file()
    }
    with zipfile.ZipFile(zip_path, "r") as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC 校验失败")
        infos = [info for info in archive.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or len(names) != len({name.casefold() for name in names}):
            raise RuntimeError("ZIP 含重复或大小写冲突成员")
        if set(names) != set(expected):
            raise RuntimeError("ZIP 成员集合与构建目录不一致")
        for info in infos:
            pure = PurePosixPath(info.filename)
            if pure.is_absolute() or ".." in pure.parts or pure.parts[0] != package_name:
                raise RuntimeError(f"ZIP 路径非法: {info.filename}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise RuntimeError(f"ZIP 含链接成员: {info.filename}")
            inner = PurePosixPath(*pure.parts[1:])
            if inner.parts[0] == "source":
                source_relative = PurePosixPath(*inner.parts[1:]).as_posix()
                if _is_forbidden(source_relative):
                    raise RuntimeError(f"ZIP 命中禁止源码路径: {source_relative}")
            elif inner.as_posix() not in GENERATED_ROOT_FILES:
                raise RuntimeError(f"ZIP 含未声明生成文件: {inner.as_posix()}")
            content = archive.read(info)
            expected_hash, expected_size = expected[info.filename]
            if len(content) != expected_size or _sha256_bytes(content) != expected_hash:
                raise RuntimeError(f"ZIP 成员内容复验失败: {info.filename}")


def _build(
    output_root: Path,
    review_base: str,
    gitleaks: Path,
) -> tuple[Path, Path]:
    selected = _selected_sources()
    initial = _snapshot(selected, review_base)
    head = str(initial["head"])
    branch = str(initial["branch"])
    if not branch:
        raise RuntimeError("当前处于 detached HEAD")
    _validate_review_base(review_base, head)

    generated_at = datetime.now(timezone.utc)
    timestamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    package_name = f"AlphaMaster_GPT56_REVIEW_{head[:8]}_worktree_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    zip_path = output_root / f"{package_name}.zip"
    hash_path = output_root / f"{package_name}.zip.sha256"
    if zip_path.exists() or hash_path.exists():
        raise RuntimeError("最终包或哈希文件已存在，拒绝覆盖")

    temp_root = Path(tempfile.mkdtemp(prefix=f".{package_name}.", dir=output_root))
    package_dir = temp_root / package_name
    temp_zip = temp_root / f"{package_name}.zip"
    temp_hash = temp_root / f"{package_name}.zip.sha256"
    try:
        source_root = package_dir / "source"
        package_dir.mkdir()
        for relative, source in selected:
            destination = source_root.joinpath(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        handoff = (PROJECT_ROOT / "docs/GPT56_PRO_EXTENDED_HANDOFF.md").read_text(
            encoding="utf-8"
        )
        prompt = (PROJECT_ROOT / "docs/GPT56_PRO_EXTENDED_PROMPT.md").read_text(
            encoding="utf-8"
        )
        _write_text(package_dir / "CONTEXT_GPT56.md", handoff)
        _write_text(package_dir / "PROMPT_GPT56.md", prompt)
        _write_text(
            package_dir / "SELECTED_WORKTREE_STATUS.txt",
            str(initial["selected_status"]) or "(无)\n",
        )
        _write_text(
            package_dir / "EXCLUDED_WORKTREE_FILES.txt",
            _excluded_status(
                str(initial["full_status"]),
                str(initial["selected_status"]),
            ),
        )
        _write_text(package_dir / "TRACKED_DIFF.patch", str(initial["tracked_diff"]))
        _write_text(
            package_dir / "SELECTED_UNTRACKED_FILES.tsv",
            _selected_untracked_tsv(
                list(initial["selected_untracked"]),
                dict(initial["sources"]),
            ),
        )

        final = _snapshot(selected, review_base)
        if final != initial:
            raise RuntimeError("构建期间 Git 状态、差异或源码发生变化，拒绝发布混合快照")
        for relative, copied in (
            (relative, source_root.joinpath(*PurePosixPath(relative).parts))
            for relative, _source in selected
        ):
            fingerprint = dict(initial["sources"])[relative]
            if (
                copied.stat().st_size != fingerprint["bytes"]
                or _sha256(copied) != fingerprint["sha256"]
            ):
                raise RuntimeError(f"复制后的源码与快照不一致: {relative}")

        gitleaks_version = _run_gitleaks(gitleaks, package_dir)
        metadata = {
            "format": "alphamaster_gpt56_review_package_v2",
            "allowlist_version": ALLOWLIST_VERSION,
            "review_base_commit": review_base,
            "workspace_head_commit": head,
            "tracked_diff_base_commit": review_base,
            "branch": branch,
            "worktree_dirty": bool(str(initial["full_status"]).strip()),
            "snapshot_sha256": _snapshot_digest(initial),
            "generated_at_utc": generated_at.isoformat(),
            "selected_source_files": len(selected),
            "selected_untracked_files": len(list(initial["selected_untracked"])),
            "gitleaks": {"version": gitleaks_version, "result": "no leaks found"},
            "security_declaration": {
                "contains_git_history": False,
                "contains_real_env_or_credentials": False,
                "contains_env_template": True,
                "contains_market_or_training_data": False,
                "contains_models_or_checkpoints": False,
                "contains_logs_or_runtime_state": False,
            },
            "warning": (
                "这是稳定代码审查基线之上的候选快照；必须结合 TRACKED_DIFF.patch、"
                "SELECTED_UNTRACKED_FILES.tsv 和两份状态文件阅读。"
            ),
        }
        _write_text(
            package_dir / "PACKAGE_METADATA.json",
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        )

        rows = _manifest_rows(package_dir)
        _write_text(
            package_dir / "SOURCE_MANIFEST.tsv",
            "".join(f"{digest}\t{size}\t{relative}\n" for digest, size, relative in rows),
        )
        if (package_dir / "SOURCE_MANIFEST.tsv").stat().st_size > MAX_FILE_BYTES:
            raise RuntimeError("manifest 超过 1 MiB")
        _verify_manifest(package_dir)

        with zipfile.ZipFile(temp_zip, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(package_dir.rglob("*")):
                if path.is_file():
                    archive.write(
                        path,
                        f"{package_name}/{path.relative_to(package_dir).as_posix()}",
                    )
        _verify_zip(temp_zip, package_dir, package_name)

        zip_digest = _sha256(temp_zip)
        _write_text(temp_hash, f"{zip_digest}  {zip_path.name}\n")
        os.replace(temp_hash, hash_path)
        os.replace(temp_zip, zip_path)
        return zip_path, hash_path
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT.parent / "handoff_packages" / "AlphaMaster",
        help="ZIP 与 SHA-256 sidecar 的输出目录（必须位于项目外）",
    )
    parser.add_argument("--review-base", default=DEFAULT_REVIEW_BASE)
    parser.add_argument("--gitleaks", type=Path, default=DEFAULT_GITLEAKS)
    args = parser.parse_args(argv)
    output_root = args.output_root.resolve()
    try:
        if output_root == PROJECT_ROOT or PROJECT_ROOT in output_root.parents:
            raise RuntimeError("输出目录必须位于 AlphaMaster 项目外")
        zip_path, hash_path = _build(output_root, args.review_base, args.gitleaks.resolve())
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"打包失败: {exc}", file=sys.stderr)
        return 1
    print(zip_path)
    print(hash_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

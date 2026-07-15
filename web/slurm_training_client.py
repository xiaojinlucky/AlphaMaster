"""Windows 本机到固定 Slurm 控制器的 OpenSSH 客户端。"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any


ALLOWED_COMPUTE_HOSTS = {
    "compute-node-11",
    "compute-node-12",
    "compute-node-13",
}
RUN_ID_RE = re.compile(r"^run_\d{8}T\d{6}Z_[0-9a-f]{8}$")
JOB_ID_RE = re.compile(r"^\d+$")
DATA_FILE_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}_(?:M1|M5|M15|M30|H1|H4|D1|W1|MN1)\.parquet$",
    re.IGNORECASE,
)
SAFE_REMOTE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_./:-]+$")
TRAINING_HISTORY_RE = re.compile(r"^training_history_[A-Za-z0-9._-]+\.json$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SlurmClientError(RuntimeError):
    """远端控制器、SSH 或产物校验失败。"""


class SlurmTransportError(SlurmClientError):
    """节点暂不可达、SSH/SCP 中断或超时；同一操作可安全重试。"""


class SlurmTrainingClient:
    def __init__(
        self,
        *,
        remote_root: str,
        selector_script: str,
        remote_python: str = "/hwdata/home/jinqc/.local/bin/python3.11",
        command_timeout: int = 45,
        transfer_timeout: int = 900,
    ) -> None:
        self.remote_root = self._validate_remote_absolute(remote_root, "remote_root")
        self.remote_python = self._validate_remote_absolute(remote_python, "remote_python")
        self.selector_script = Path(selector_script).expanduser().resolve()
        if not self.selector_script.is_file():
            raise SlurmClientError(f"节点选择脚本不存在: {self.selector_script}")
        self.command_timeout = int(command_timeout)
        self.transfer_timeout = int(transfer_timeout)
        self.ssh = shutil.which("ssh")
        self.scp = shutil.which("scp")
        self.powershell = shutil.which("powershell.exe") or shutil.which("powershell")
        if not self.ssh or not self.scp or not self.powershell:
            raise SlurmClientError("缺少系统 ssh、scp 或 powershell")

    @classmethod
    def from_environment(cls) -> "SlurmTrainingClient":
        return cls(
            remote_root=os.getenv(
                "SLURM_REMOTE_ROOT", "/hwdata/home/jinqc/Quant/AlphaMaster"
            ),
            selector_script=os.getenv(
                "SLURM_NODE_SELECTOR",
                r"D:\Desktop\codex-remote-tools\check-best-node.ps1",
            ),
            remote_python=os.getenv(
                "SLURM_REMOTE_PYTHON",
                "/hwdata/home/jinqc/.local/bin/python3.11",
            ),
            command_timeout=int(os.getenv("SLURM_COMMAND_TIMEOUT", "45")),
            transfer_timeout=int(os.getenv("SLURM_TRANSFER_TIMEOUT", "900")),
        )

    @staticmethod
    def _validate_remote_absolute(value: str, field: str) -> str:
        raw = str(value or "").strip()
        path = PurePosixPath(raw)
        if not raw.startswith("/") or ".." in path.parts:
            raise SlurmClientError(f"{field} 必须是无上跳的 POSIX 绝对路径")
        if not SAFE_REMOTE_TOKEN_RE.fullmatch(raw):
            raise SlurmClientError(f"{field} 含不允许字符")
        return raw.rstrip("/")

    @staticmethod
    def validate_run_id(run_id: str) -> str:
        if not RUN_ID_RE.fullmatch(run_id):
            raise SlurmClientError("run_id 格式非法")
        return run_id

    @staticmethod
    def validate_job_id(job_id: str | int) -> str:
        value = str(job_id)
        if not JOB_ID_RE.fullmatch(value):
            raise SlurmClientError("Slurm job ID 非法")
        return value

    @staticmethod
    def validate_data_filename(filename: str) -> str:
        value = str(filename)
        if Path(value).name != value or not DATA_FILE_RE.fullmatch(value):
            raise SlurmClientError("训练数据文件名非法")
        return value

    def select_compute_host(self) -> str:
        try:
            proc = subprocess.run(
                [
                    self.powershell,
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(self.selector_script),
                ],
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SlurmTransportError("计算节点选择暂时不可用") from exc
        if proc.returncode != 0:
            raise SlurmTransportError("计算节点选择失败")
        match = re.search(r"Recommended now:\s*(compute-node-(?:11|12|13))\b", proc.stdout)
        if not match or match.group(1) not in ALLOWED_COMPUTE_HOSTS:
            raise SlurmTransportError("节点选择器未返回允许的计算节点")
        return match.group(1)

    def _ssh_base(self, host: str) -> list[str]:
        if host not in ALLOWED_COMPUTE_HOSTS:
            raise SlurmClientError("拒绝在非计算节点执行远端控制命令")
        return [
            self.ssh,
            "-n",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "ConnectionAttempts=1",
            host,
        ]

    def _remote_call(
        self,
        action: str,
        *args: str,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        if action not in {
            "prepare",
            "finalize-upload",
            "submit",
            "status",
            "tail",
            "cancel",
            "result",
        }:
            raise SlurmClientError("不允许的远端控制动作")
        tokens = [
            self.remote_python,
            f"{self.remote_root}/scripts/slurm_control.py",
            action,
            *[str(arg) for arg in args],
        ]
        if any(not SAFE_REMOTE_TOKEN_RE.fullmatch(token) for token in tokens):
            raise SlurmClientError("远端控制参数含不允许字符")
        host = self.select_compute_host()
        command = shlex.join(tokens)
        try:
            proc = subprocess.run(
                [*self._ssh_base(host), command],
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout or self.command_timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SlurmTransportError(f"远端控制动作 {action} 传输中断") from exc
        if proc.returncode != 0:
            message = (proc.stderr or proc.stdout or "远端控制失败").strip()
            if proc.returncode == 255:
                raise SlurmTransportError(message[-1200:])
            raise SlurmClientError(message[-1200:])
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise SlurmClientError("远端控制器返回非 JSON 数据") from exc
        if not isinstance(payload, dict):
            raise SlurmClientError("远端控制器返回结构非法")
        return payload

    def prepare(self, run_id: str) -> dict[str, Any]:
        return self._remote_call("prepare", self.validate_run_id(run_id))

    def upload_inputs(
        self,
        *,
        run_id: str,
        data_file: Path,
        manifest_file: Path,
        data_sha256: str,
    ) -> dict[str, Any]:
        run_id = self.validate_run_id(run_id)
        filename = self.validate_data_filename(data_file.name)
        if not data_file.is_file() or not manifest_file.is_file():
            raise SlurmClientError("本机上传文件不存在")
        if not re.fullmatch(r"[0-9a-f]{64}", data_sha256):
            raise SlurmClientError("数据 SHA-256 非法")
        host = self.select_compute_host()
        remote_input = f"{self.remote_root}/runs/{run_id}/input"
        transfers = (
            (data_file, f"{host}:{remote_input}/{filename}.partial"),
            (manifest_file, f"{host}:{remote_input}/run_manifest.json.partial"),
        )
        for local_path, remote_spec in transfers:
            try:
                proc = subprocess.run(
                    [
                        self.scp,
                        "-q",
                        "-o",
                        "BatchMode=yes",
                        "-o",
                        "StrictHostKeyChecking=yes",
                        "-o",
                        "ConnectTimeout=8",
                        str(local_path),
                        remote_spec,
                    ],
                    shell=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.transfer_timeout,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise SlurmTransportError("SCP 上传中断") from exc
            if proc.returncode != 0:
                raise SlurmTransportError((proc.stderr or "SCP 上传失败").strip()[-1200:])
        return self._remote_call(
            "finalize-upload",
            run_id,
            filename,
            data_sha256,
            str(data_file.stat().st_size),
        )

    def submit(self, run_id: str) -> str:
        payload = self._remote_call("submit", self.validate_run_id(run_id))
        return self.validate_job_id(payload.get("job_id", ""))

    def status(self, run_id: str, job_id: str | int) -> dict[str, Any]:
        return self._remote_call(
            "status", self.validate_run_id(run_id), self.validate_job_id(job_id)
        )

    def tail(self, run_id: str, job_id: str | int, lines: int = 150) -> list[str]:
        count = max(1, min(int(lines), 500))
        payload = self._remote_call(
            "tail",
            self.validate_run_id(run_id),
            self.validate_job_id(job_id),
            str(count),
        )
        rows = payload.get("lines") or []
        if not isinstance(rows, list):
            raise SlurmClientError("远端日志结构非法")
        return [str(row) for row in rows[-count:]]

    def cancel(self, run_id: str, job_id: str | int) -> dict[str, Any]:
        return self._remote_call(
            "cancel", self.validate_run_id(run_id), self.validate_job_id(job_id)
        )

    @staticmethod
    def _validate_artifact_path(value: str) -> PurePosixPath:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise SlurmClientError("结果 manifest 含越界路径")
        allowed_root_file = len(path.parts) == 1 and bool(
            TRAINING_HISTORY_RE.fullmatch(path.name)
        )
        if path.parts[0] not in {"checkpoints", "strategies", "output"} and not allowed_root_file:
            raise SlurmClientError("结果 manifest 含非白名单路径")
        if not SAFE_REMOTE_TOKEN_RE.fullmatch(value):
            raise SlurmClientError("结果路径含不允许字符")
        return path

    def download_result(
        self,
        *,
        run_id: str,
        job_id: str | int,
        local_artifact_root: Path,
        expected_commit: str,
    ) -> dict[str, Any]:
        run_id = self.validate_run_id(run_id)
        job_id = self.validate_job_id(job_id)
        manifest = self._remote_call("result", run_id, job_id)
        if manifest.get("run_id") != run_id or str(manifest.get("slurm_job_id")) != job_id:
            raise SlurmClientError("结果 manifest 的 run/job 身份不匹配")
        if manifest.get("git_commit") != expected_commit:
            raise SlurmClientError("结果 manifest 的源码提交不匹配")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or len(artifacts) > 64:
            raise SlurmClientError("结果 manifest 的产物列表非法")

        local_artifact_root.mkdir(parents=True, exist_ok=True)
        host = self.select_compute_host()
        for row in artifacts:
            if not isinstance(row, dict):
                raise SlurmClientError("结果产物结构非法")
            rel = str(row.get("path") or "")
            path = self._validate_artifact_path(rel)
            size = int(row.get("size") or -1)
            digest = str(row.get("sha256") or "")
            if size < 0 or size > 2 * 1024 * 1024 * 1024:
                raise SlurmClientError("结果产物大小超限")
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise SlurmClientError("结果产物 SHA-256 非法")
            dest = local_artifact_root.joinpath(*path.parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            partial = dest.with_name(dest.name + ".partial")
            remote_spec = f"{host}:{self.remote_root}/runs/{run_id}/{rel}"
            try:
                proc = subprocess.run(
                    [
                        self.scp,
                        "-q",
                        "-o",
                        "BatchMode=yes",
                        "-o",
                        "StrictHostKeyChecking=yes",
                        "-o",
                        "ConnectTimeout=8",
                        remote_spec,
                        str(partial),
                    ],
                    shell=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.transfer_timeout,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise SlurmTransportError(f"结果下载中断: {rel}") from exc
            if proc.returncode != 0:
                raise SlurmTransportError((proc.stderr or "SCP 下载失败").strip()[-1200:])
            try:
                actual_size = partial.stat().st_size
                actual_digest = sha256_file(partial)
            except OSError as exc:
                raise SlurmTransportError(f"无法读取已下载结果: {rel}") from exc
            if actual_size != size or actual_digest != digest:
                partial.unlink(missing_ok=True)
                raise SlurmClientError(f"结果校验失败: {rel}")
            try:
                os.replace(partial, dest)
            except OSError as exc:
                raise SlurmTransportError(f"无法发布已下载结果: {rel}") from exc

        manifest_path = local_artifact_root / "output" / "result_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temp = manifest_path.with_suffix(".json.tmp")
        try:
            temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temp, manifest_path)
        except OSError as exc:
            raise SlurmTransportError("无法保存结果 manifest") from exc
        return manifest

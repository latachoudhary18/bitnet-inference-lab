
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_BITNET_CLI = Path("llama-cli.exe")  # override with the BITNET_CLI environment variable


def run_powershell(command):
    """Return PowerShell output, or None if the query fails."""
    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def get_windows_data():
    """Collect Windows hardware details using CIM."""
    commands = {
        "cpu": (
            "Get-CimInstance Win32_Processor | "
            "Select-Object -First 1 Name,NumberOfCores,"
            "NumberOfLogicalProcessors,Architecture | "
            "ConvertTo-Json -Compress"
        ),
        "memory": (
            "Get-CimInstance Win32_ComputerSystem | "
            "Select-Object TotalPhysicalMemory | ConvertTo-Json -Compress"
        ),
        "gpus": (
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,AdapterRAM,DriverVersion | "
            "ConvertTo-Json -Compress"
        ),
        "os": (
            "Get-CimInstance Win32_OperatingSystem | "
            "Select-Object Caption,Version,BuildNumber | "
            "ConvertTo-Json -Compress"
        ),
    }

    data = {}
    for key, command in commands.items():
        output = run_powershell(command)
        if output:
            try:
                data[key] = json.loads(output)
            except json.JSONDecodeError:
                data[key] = None
        else:
            data[key] = None

    return data


def as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def main():
    windows_data = get_windows_data()

    cpu = windows_data.get("cpu") or {}
    memory = windows_data.get("memory") or {}
    os_info = windows_data.get("os") or {}
    gpus = as_list(windows_data.get("gpus"))

    total_ram = memory.get("TotalPhysicalMemory")
    cli_path = Path(
        os.environ.get("BITNET_CLI", str(DEFAULT_BITNET_CLI))
    ).expanduser()

    gpu_info = []
    for gpu in gpus:
        if not isinstance(gpu, dict):
            continue
        gpu_info.append({
            "name": gpu.get("Name"),
            "adapter_ram_bytes_reported_by_windows": gpu.get("AdapterRAM"),
            "driver_version": gpu.get("DriverVersion"),
            "note": (
                "Windows AdapterRAM can be inaccurate for some GPUs; "
                "a missing value does not mean the GPU has no VRAM."
            ),
        })

    report = {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "windows_name": os_info.get("Caption"),
            "windows_version": os_info.get("Version"),
            "windows_build": os_info.get("BuildNumber"),
            "machine_architecture": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "cpu": {
            "model": cpu.get("Name"),
            "architecture_code_from_windows": cpu.get("Architecture"),
            "physical_cores_reported_by_windows": cpu.get("NumberOfCores"),
            "logical_processors_reported_by_windows": (
                cpu.get("NumberOfLogicalProcessors")
            ),
            "logical_processors_seen_by_python": os.cpu_count(),
        },
        "memory": {
            "total_bytes": total_ram,
            "total_gib": (
                round(int(total_ram) / (1024 ** 3), 2)
                if total_ram is not None
                else None
            ),
        },
        "gpus": gpu_info,
        "bitnet_cli": {
            "path": str(cli_path),
            "exists": cli_path.is_file(),
            "size_bytes": (
                cli_path.stat().st_size if cli_path.is_file() else None
            ),
        },
        "limitations": [
            "GPU detection uses Windows CIM; reported adapter memory may be inaccurate.",
            "GPU detection does not prove that BitNet uses the GPU.",
            "CPU instruction-set support and actual inference performance "
            "are not benchmarked by this profiler.",
            "Missing fields mean the information could not be collected.",
        ],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "system_info.json"
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Hardware profiling completed.")
    print(f"Report: {output_path}")
    print(f"CPU: {report['cpu']['model'] or 'Not detected'}")
    print(f"Logical processors: {report['cpu']['logical_processors_reported_by_windows']}")
    print(f"RAM: {report['memory']['total_gib']} GiB")
    print(f"GPU(s) detected: {len(gpu_info)}")
    print(f"BitNet CLI exists: {report['bitnet_cli']['exists']}")


if __name__ == "__main__":
    main()

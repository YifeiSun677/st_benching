"""
Resource profiling harness for the ST benchmark.

Designed to be model-agnostic: import this from every model's run script so the
efficiency table is produced by one instrument rather than four different ones.

Captures the four quantities on the efficiency slide:
    throughput          spots/s and s/section, training and inference
    peak GPU memory     torch-attributable (headline) + nvidia-smi (cross-check)
    peak host RAM       RSS summed over the process tree (includes dataloader workers)
    mean GPU util       sampled at 1 Hz over the measured window only

Usage
-----
    from profiling import ResourceMonitor, StepTimer, write_report

    mon = ResourceMonitor(interval=1.0)
    timer = StepTimer()
    with mon:
        for epoch in range(n_warmup):
            ...                       # warm-up, not measured
        mon.reset_window()            # <- steady state starts here
        for epoch in range(n_measure):
            for batch in loader:
                t0 = time.perf_counter()
                ...                   # one gradient step
                timer.add(time.perf_counter() - t0, n_spots=batch_n_spots)
    write_report("results/efficiency/hist2st_train.json", {**mon.summary(), **timer.summary()})
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
except Exception:  # pragma: no cover
    _NVML = False


# --------------------------------------------------------------------------- #
# GPU sampling
# --------------------------------------------------------------------------- #
def _gpu_sample(device_index: int = 0):
    """Return (util_pct, mem_used_bytes) or (None, None) if unavailable."""
    if _NVML:
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(h).used
            return float(util), float(mem)
        except Exception:
            return None, None
    # fallback: shell out to nvidia-smi
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        util_s, mem_s = out.decode().strip().split(",")
        return float(util_s), float(mem_s) * 1024 ** 2
    except Exception:
        return None, None


def _host_rss_bytes() -> float:
    """RSS of this process plus all children (dataloader workers live there)."""
    if psutil is None:
        # crude fallback: self only, via /proc
        try:
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) * 1024
        except Exception:
            pass
        return 0.0
    total = 0.0
    try:
        p = psutil.Process(os.getpid())
        total += p.memory_info().rss
        for c in p.children(recursive=True):
            try:
                total += c.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception:
        pass
    return total


# --------------------------------------------------------------------------- #
# Monitor
# --------------------------------------------------------------------------- #
class ResourceMonitor:
    """Background sampler. Samples are timestamped; summary() covers the window
    from the last reset_window() call (or start) to stop()."""

    def __init__(self, interval: float = 1.0, device_index: int = 0):
        self.interval = interval
        self.device_index = device_index
        self._samples: list[tuple[float, float | None, float | None, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._window_start = None
        self._stopped_at = None

    # -- lifecycle ---------------------------------------------------------- #
    def start(self):
        self._window_start = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            util, gmem = _gpu_sample(self.device_index)
            self._samples.append((time.perf_counter(), util, gmem, _host_rss_bytes()))
            self._stop.wait(self.interval)

    def reset_window(self):
        """Call after warm-up so summary() reflects steady state only."""
        self._window_start = time.perf_counter()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_index)

    def stop(self):
        self._stopped_at = time.perf_counter()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 3)
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    # -- results ------------------------------------------------------------ #
    def summary(self) -> dict:
        lo = self._window_start or 0.0
        hi = self._stopped_at or time.perf_counter()
        win = [s for s in self._samples if lo <= s[0] <= hi]

        utils = [s[1] for s in win if s[1] is not None]
        gmems = [s[2] for s in win if s[2] is not None]
        rsss = [s[3] for s in win if s[3] is not None]

        out = {
            "window_seconds": round(hi - lo, 2),
            "n_samples": len(win),
            "mean_gpu_util_pct": round(sum(utils) / len(utils), 2) if utils else None,
            "max_gpu_util_pct": round(max(utils), 2) if utils else None,
            "peak_gpu_mem_smi_gb": round(max(gmems) / 1e9, 3) if gmems else None,
            "peak_host_rss_gb": round(max(rsss) / 1e9, 3) if rsss else None,
        }
        if torch is not None and torch.cuda.is_available():
            out["peak_gpu_mem_torch_alloc_gb"] = round(
                torch.cuda.max_memory_allocated(self.device_index) / 1e9, 3
            )
            out["peak_gpu_mem_torch_reserved_gb"] = round(
                torch.cuda.max_memory_reserved(self.device_index) / 1e9, 3
            )
        return out


# --------------------------------------------------------------------------- #
# Step timing
# --------------------------------------------------------------------------- #
class StepTimer:
    """Accumulates per-step wall times and the number of spots in each step."""

    def __init__(self):
        self.times: list[float] = []
        self.spots: list[int] = []

    def add(self, seconds: float, n_spots: int = 0):
        self.times.append(seconds)
        self.spots.append(int(n_spots))

    def summary(self, prefix: str = "") -> dict:
        if not self.times:
            return {}
        total_t = sum(self.times)
        total_s = sum(self.spots)
        n = len(self.times)
        srt = sorted(self.times)
        return {
            f"{prefix}n_steps": n,
            f"{prefix}total_seconds": round(total_t, 3),
            f"{prefix}sec_per_step_mean": round(total_t / n, 4),
            f"{prefix}sec_per_step_median": round(srt[n // 2], 4),
            f"{prefix}sec_per_step_max": round(srt[-1], 4),
            f"{prefix}total_spots": total_s,
            f"{prefix}spots_per_sec": round(total_s / total_t, 2) if total_t else None,
        }


# --------------------------------------------------------------------------- #
# Environment + reporting
# --------------------------------------------------------------------------- #
def environment() -> dict:
    env = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": os.uname().nodename,
        "cpu_count": os.cpu_count(),
    }
    if psutil is not None:
        env["host_ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    if torch is not None:
        env["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_total_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
            env["cuda_version"] = torch.version.cuda
    return env


def write_report(path: str | Path, payload: dict, also_csv: str | Path | None = None):
    """Write a JSON report; optionally append a flat row to a benchmark-wide CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**environment(), **payload}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[profiling] wrote {path}")

    if also_csv is not None:
        import csv

        csv_path = Path(also_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        flat = {k: v for k, v in payload.items() if not isinstance(v, (dict, list))}
        exists = csv_path.exists()
        # union of existing header and this row, so models with extra fields still append
        fieldnames = list(flat.keys())
        if exists:
            with open(csv_path, newline="") as fh:
                old = csv.reader(fh)
                header = next(old, [])
            fieldnames = header + [k for k in flat if k not in header]
            if fieldnames != header:  # header grew -> rewrite file
                with open(csv_path, newline="") as fh:
                    rows = list(csv.DictReader(fh))
                with open(csv_path, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerows(rows)
        with open(csv_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            if not exists:
                w.writeheader()
            w.writerow(flat)
        print(f"[profiling] appended row to {csv_path}")

    return payload

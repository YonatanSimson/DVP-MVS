#!/usr/bin/env bash
# Resume-safe launcher for APD.
#
# WSL's GPU-paravirtualization layer (dxgkrnl) can wedge on a Windows-side
# session lock/unlock, which segfaults any in-flight CUDA call
# (dxgkio_query_adapter_info: Ioctl failed -22/-2 in `dmesg`) -- see
# microsoft/WSL#13734 / #11293. It's not a bug in APD, and it typically
# clears itself within a minute. This script reruns APD when that happens
# instead of losing the whole batch; APD's own iteration checkpointing
# (Checkpoint.h/.cpp) means each retry resumes from the last fully-completed
# iteration rather than starting over.
set -u

if [ "$#" -lt 1 ]; then
	echo "USAGE: $(basename "$0") <dense_folder> [gpu_index] [max_retries]" >&2
	exit 1
fi

dense_folder="$1"
gpu_index="${2:-0}"
max_retries="${3:-20}"
retry_delay_secs=30
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
apd_bin="${script_dir}/build/APD"

if [ ! -x "$apd_bin" ]; then
	echo "APD binary not found or not executable at ${apd_bin} -- build first." >&2
	exit 1
fi

attempt=0
while true; do
	attempt=$((attempt + 1))
	echo "[run_resilient] attempt ${attempt}/${max_retries}: ${apd_bin} ${dense_folder} ${gpu_index}"
	"$apd_bin" "$dense_folder" "$gpu_index"
	status=$?

	if [ "$status" -eq 0 ]; then
		echo "[run_resilient] APD finished successfully."
		exit 0
	fi

	if [ "$status" -eq 130 ] || [ "$status" -eq 143 ]; then
		echo "[run_resilient] Interrupted by user (exit ${status}), not retrying."
		exit "$status"
	fi

	if [ "$attempt" -ge "$max_retries" ]; then
		echo "[run_resilient] Giving up after ${attempt} attempts (last exit code: ${status})." >&2
		exit "$status"
	fi

	echo "[run_resilient] APD exited with code ${status} (likely the WSL GPU-paravirt crash, not an APD bug) -- waiting ${retry_delay_secs}s and resuming from checkpoint." >&2
	sleep "$retry_delay_secs"
done

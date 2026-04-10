"""
===============================================================================
STEP 2.2: RUN ABAQUS SIMULATIONS
===============================================================================
Purpose: Submit Abaqus jobs and collect results

For each .inp file:
    1. Create dedicated job directory
    2. Submit to Abaqus solver
    3. Monitor job status
    4. Collect output files (.odb, .dat, .msg, .sta)
    5. Record metadata and errors
    
Features:
    - Parallel execution control (cpus)
    - Range-based job selection (START_ID to END_ID)
    - Automatic error capture and logging
    - Results organized by job name
===============================================================================
"""

import subprocess
import shutil
import time
import os
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

# Abaqus executable
ABAQUS_CMD = os.environ.get("ABAQUS_CMD", "abaqus")

# Directories
ROOT = Path(__file__).resolve().parents[1]
INP_DIR = ROOT / "outputs_inp"
JOB_ROOT = ROOT / "abaqus_jobs"
RESULTS_ROOT = ROOT / "02_abaqus" / "results"

# Job control
START_ID = 0        # First sample to run
END_ID = 399        # Last sample to run (inclusive)
CPUS = 1            # Number of CPUs per job

# Create directories
JOB_ROOT.mkdir(exist_ok=True)
RESULTS_ROOT.mkdir(exist_ok=True)

# ============================================================
# JOB EXECUTION
# ============================================================

def run_abaqus_job(inp_file, job_dir, job_name):
    """
    Run a single Abaqus job.
    
    Parameters:
        inp_file: Path to .inp file
        job_dir: Working directory for job
        job_name: Name of the job
        
    Returns:
        status: "SUCCESS" or "FAILED"
        error_message: Error details if failed, empty string otherwise
        runtime: Job runtime in seconds
    """
    
    # Copy input file to job directory
    inp_local = job_dir / inp_file.name
    shutil.copy(inp_file, inp_local)
    
    # Construct Abaqus command
    cmd = [
        ABAQUS_CMD,
        f"job={job_name}",
        f"input={inp_local.name}",
        f"cpus={CPUS}",
        "interactive",
        "ask_delete=OFF"
    ]
    
    # Execute job
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=job_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        status = "SUCCESS"
        error_message = ""
        
    except subprocess.TimeoutExpired:
        status = "TIMEOUT"
        error_message = "Job exceeded 1 hour time limit"
        
    except subprocess.CalledProcessError as e:
        status = "FAILED"
        error_message = e.stderr if e.stderr else "Abaqus returned non-zero exit code"
        
        # Also check .msg file for additional error info
        msg_file = job_dir / f"{job_name}.msg"
        if msg_file.exists():
            try:
                msg_content = msg_file.read_text()
                # Extract last 500 characters for error context
                error_message += f"\n\nFrom .msg file:\n{msg_content[-500:]}"
            except:
                pass
    
    except Exception as e:
        status = "ERROR"
        error_message = f"Unexpected error: {str(e)}"
    
    runtime = time.time() - start_time
    
    return status, error_message, runtime


def collect_results(job_dir, result_dir, job_name, status, error_message, runtime):
    """
    Move output files to results directory and create metadata.
    
    Parameters:
        job_dir: Source directory (temporary job folder)
        result_dir: Destination directory (permanent results)
        job_name: Job identifier
        status: Job completion status
        error_message: Error details if failed
        runtime: Job execution time in seconds
    """
    
    result_dir.mkdir(exist_ok=True)
    
    # Files to collect
    extensions = [".odb", ".dat", ".msg", ".sta", ".log"]
    
    collected = []
    missing = []
    
    for ext in extensions:
        src_file = job_dir / f"{job_name}{ext}"
        dst_file = result_dir / f"{job_name}{ext}"
        
        if src_file.exists():
            # Move file to results
            shutil.move(str(src_file), str(dst_file))
            collected.append(ext)
        else:
            missing.append(ext)
    
    # Create metadata file
    metadata_path = result_dir / "metadata.txt"
    with open(metadata_path, "w") as f:
        f.write(f"job_name: {job_name}\n")
        f.write(f"timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"status: {status}\n")
        f.write(f"runtime: {runtime:.2f} seconds\n")
        f.write(f"solver: Abaqus Student Edition\n")
        f.write(f"cpus: {CPUS}\n")
        f.write(f"\nCollected files: {', '.join(collected)}\n")
        if missing:
            f.write(f"Missing files: {', '.join(missing)}\n")
        if error_message:
            f.write(f"\nError details:\n{error_message}\n")
    
    return len(collected), len(missing)


# ============================================================
# MAIN EXECUTION LOOP
# ============================================================

def main():
    
    print("=" * 70)
    print("ABAQUS JOB EXECUTION")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Job range:  sample_{START_ID:03d} to sample_{END_ID:03d}")
    print(f"CPUs/job:   {CPUS}")
    print("=" * 70)
    
    # Find all .inp files in range
    inp_files = sorted(INP_DIR.glob("sample_*.inp"))
    
    # Filter by ID range
    jobs_to_run = []
    for inp in inp_files:
        sample_id = int(inp.stem.split("_")[1])
        if START_ID <= sample_id <= END_ID:
            jobs_to_run.append((sample_id, inp))
    
    print(f"\nFound {len(jobs_to_run)} jobs to run\n")
    
    if len(jobs_to_run) == 0:
        print("⚠ No jobs found in specified range!")
        return
    
    # Statistics
    stats = {
        "success": 0,
        "failed": 0,
        "timeout": 0,
        "error": 0,
        "total_runtime": 0
    }
    
    # Execute jobs
    for i, (sample_id, inp_file) in enumerate(jobs_to_run, 1):
        
        job_name = f"sample_{sample_id:03d}"
        job_dir = JOB_ROOT / job_name
        result_dir = RESULTS_ROOT / job_name
        
        # Create job directory
        job_dir.mkdir(exist_ok=True)
        
        print(f"\n[{i}/{len(jobs_to_run)}] Running: {job_name}")
        print("-" * 70)
        
        # Run job
        status, error_message, runtime = run_abaqus_job(inp_file, job_dir, job_name)
        
        # Update statistics
        stats[status.lower()] = stats.get(status.lower(), 0) + 1
        stats["total_runtime"] += runtime
        
        # Print status
        if status == "SUCCESS":
            print(f"✓ {status} ({runtime:.1f}s)")
        else:
            print(f"✗ {status} ({runtime:.1f}s)")
            if error_message:
                print(f"  Error: {error_message[:200]}")
        
        # Collect results
        n_collected, n_missing = collect_results(
            job_dir, result_dir, job_name, 
            status, error_message, runtime
        )
        
        print(f"  Files collected: {n_collected}, missing: {n_missing}")
    
    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total jobs:       {len(jobs_to_run)}")
    print(f"Successful:       {stats['success']}")
    print(f"Failed:           {stats['failed']}")
    print(f"Timeout:          {stats['timeout']}")
    print(f"Errors:           {stats['error']}")
    print(f"Total runtime:    {stats['total_runtime']/60:.1f} minutes")
    print(f"Avg runtime/job:  {stats['total_runtime']/len(jobs_to_run):.1f} seconds")
    print(f"\nResults saved to: {RESULTS_ROOT}")
    print("=" * 70)
    
    # Create summary file
    summary_file = RESULTS_ROOT / "execution_summary.txt"
    with open(summary_file, "w") as f:
        f.write("ABAQUS EXECUTION SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Job range: sample_{START_ID:03d} to sample_{END_ID:03d}\n")
        f.write(f"Total jobs: {len(jobs_to_run)}\n")
        f.write(f"Successful: {stats['success']}\n")
        f.write(f"Failed: {stats['failed']}\n")
        f.write(f"Timeout: {stats['timeout']}\n")
        f.write(f"Errors: {stats['error']}\n")
        f.write(f"Total runtime: {stats['total_runtime']/60:.1f} minutes\n")
        f.write(f"Average runtime per job: {stats['total_runtime']/len(jobs_to_run):.1f} seconds\n")
    
    print(f"\n✓ Summary saved to: {summary_file}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

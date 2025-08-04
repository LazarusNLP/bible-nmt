# Spartan HPC Job Management Guide

This guide covers how to submit, monitor, and manage jobs on the University of Melbourne's Spartan HPC cluster.

## 🚨 Important Rules

**⚠️ Never run applications on the login node!**
- Login nodes are shared resources for file management, script creation, and job submission only
- Running compute-intensive jobs on login nodes will get your job killed and may result in account suspension
- Always submit jobs to the queue using `sbatch`

## 📋 Job Submission Basics

### 1. **Job Script Structure**

Every job script must have:
```bash
#!/bin/bash
#SBATCH --job-name=my-job-name
#SBATCH --account=punim0478
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@unimelb.edu.au

# Your commands here
module load CUDA/12.4.1
conda activate your_env
python your_script.py
```

### 2. **Common SBATCH Directives**

| Directive | Description | Example |
|-----------|-------------|---------|
| `--job-name` | Job name for identification | `--job-name=bible-training` |
| `--account` | Project account | `--account=punim0478` |
| `--partition` | Queue/partition | `--partition=deeplearn` |
| `--qos` | Quality of Service | `--qos=gpgpudeeplearn` |
| `--nodes` | Number of nodes | `--nodes=1` |
| `--ntasks` | Number of tasks | `--ntasks=1` |
| `--cpus-per-task` | CPUs per task | `--cpus-per-task=8` |
| `--gres` | Generic resources (GPU) | `--gres=gpu:1` |
| `--mem` | Memory per node | `--mem=32G` |
| `--time` | Wall time limit | `--time=0-12:00:00` |
| `--mail-type` | Email notifications | `--mail-type=BEGIN,END,FAIL` |

### 3. **Job Types**

#### Single Core Jobs
```bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
```

#### Multithreaded (SMP) Jobs
```bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
```

#### MPI Jobs
```bash
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
```

#### GPU Jobs
```bash
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --gres=gpu:1
```

## 🚀 Submitting Jobs

### Submit a Job
```bash
sbatch your_script.slurm
```

**Example output:**
```
Submitted batch job 13726820
```

### Submit with Different Account
```bash
sbatch --account=punim0478 your_script.slurm
```

## 👀 Monitoring Jobs

### Check Job Status
```bash
# All your jobs
squeue -u $USER

# Specific job
squeue -j JOBID

# Detailed job information
scontrol show job JOBID
```

### Job States
- **PD** (Pending): Waiting for resources
- **R** (Running): Currently executing
- **CG** (Completing): Job is finishing
- **CD** (Completed): Job finished successfully
- **F** (Failed): Job failed
- **CA** (Cancelled): Job was cancelled

### Common Reasons for Pending Jobs
- `(Priority)`: Lower priority, waiting in queue
- `(Resources)`: Waiting for available resources
- `(QOSMaxJobsPerUserLimit)`: Hit job limit for user
- `(None)`: General resource constraint

## 📊 Job Output and Logs

### Default Output Files
- **Standard output**: `slurm-JOBID.out`
- **Standard error**: `slurm-JOBID.err`

### Custom Output Files
```bash
#SBATCH --output="job-%N.%j.out"    # STDOUT
#SBATCH --error="job-%N.%j.err"     # STDERR
```

### Monitor Live Output
```bash
# Follow output in real-time
tail -f slurm-JOBID.out

# Follow errors
tail -f slurm-JOBID.err

# Both at once
tail -f slurm-JOBID.out & tail -f slurm-JOBID.err
```

### Handle Buffering Issues
For immediate output (no buffering):
```bash
stdbuf -o0 -e0 python your_script.py
```

## ❌ Cancelling Jobs

### Cancel Specific Job
```bash
scancel JOBID
```

### Cancel All Your Jobs
```bash
scancel -u $USER
```

### Cancel Jobs by Name
```bash
scancel -n job-name
```

### Cancel Jobs in Specific State
```bash
scancel -u $USER -t PENDING    # Cancel all pending jobs
```

## 📈 Job History and Accounting

### View Job History
```bash
# Your recent jobs
sacct -u $USER

# Specific job details
sacct -j JOBID

# Jobs from specific date
sacct -u $USER -S 2024-01-01

# Detailed format
sacct -u $USER --format=JobID,JobName,Account,Partition,State,ExitCode,Elapsed,CPUTime
```

### Useful sacct Options
- `-S YYYY-MM-DD`: Start date
- `-E YYYY-MM-DD`: End date
- `-X`: Show only job totals (not job steps)
- `--format=`: Customize output columns

## 💾 Memory Management

### Common Memory Issues
If you see "Killed" in your output:
```
/var/spool/slurm/job123/slurm_script: line 7: 12235 Killed python script.py
```

This usually means **out of memory**. Solutions:

### Request More Memory
```bash
#SBATCH --mem=64G                    # Total memory per node
#SBATCH --mem-per-cpu=8G            # Memory per CPU core
```

### Default Memory Allocation
- Default: `(Total node memory) / (Number of cores requested)`
- Example: 128GB node with 8 cores = 16GB per core by default

## 🔧 Partitions and QoS

### Available Partitions
- **sapphire**: Standard CPU jobs
- **cascade**: Standard CPU jobs
- **long**: Long-running jobs
- **bigmem**: High-memory jobs
- **gpu-a100**: A100 GPU jobs
- **gpu-h100**: H100 GPU jobs
- **deeplearn**: Deep learning GPU jobs

### Check Available Resources
```bash
# View partition info
sinfo -p partition-name

# View all partitions
sinfo

# Check node details
sinfo -N -l
```

### Quality of Service (QoS)
Common QoS options:
- `gpgpudeeplearn`: For GPU deep learning jobs
- Check your available QoS:
```bash
sacctmgr show user $USER withassoc
```

## 📧 Email Notifications

### Configure Email Alerts
```bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@unimelb.edu.au
```

### Email Types
- `BEGIN`: Job starts
- `END`: Job completes successfully
- `FAIL`: Job fails
- `ALL`: All events
- `TIME_LIMIT`: Job hits time limit

## 🛠️ Troubleshooting Common Issues

### Job Won't Submit
1. **Invalid QoS**: Check your QoS permissions
2. **Partition access**: Verify you can access the partition
3. **Resource limits**: Check if requesting too many resources

### Job Stuck in Queue
1. **Resource availability**: Check `sinfo` for free nodes
2. **Priority**: Other jobs may have higher priority
3. **Limits**: You may have hit job/resource limits

### Job Fails Immediately
1. **Module loading**: Check if modules load correctly
2. **File paths**: Verify all paths are correct
3. **Permissions**: Check file/directory permissions
4. **Memory**: Job may need more memory

### No Output Files
1. **Job hasn't started**: Output files only appear when job runs
2. **Directory permissions**: Check write permissions
3. **Job failed before output**: Check with `sacct -j JOBID`

## 📝 Best Practices

1. **Test jobs interactively first**:
   ```bash
   sinteractive --account=punim0478 --partition=deeplearn --gres=gpu:1
   ```

2. **Use meaningful job names** for easy identification

3. **Set appropriate time limits** (jobs killed if exceeded)

4. **Request appropriate resources** (not too much, not too little)

5. **Use email notifications** for long-running jobs

6. **Check job efficiency** with `seff JOBID` after completion

7. **Monitor disk usage** in your project directory

## 🔗 Quick Reference Commands

```bash
# Submit job
sbatch script.slurm

# Check queue
squeue -u $USER

# Cancel job
scancel JOBID

# Job details
scontrol show job JOBID

# Job history
sacct -u $USER

# Interactive session
sinteractive --account=punim0478 --partition=deeplearn --gres=gpu:1 --time=2:00:00

# Check efficiency
seff JOBID

# Monitor output
tail -f slurm-JOBID.out
```

## 📞 Getting Help

- **Spartan Documentation**: https://dashboard.hpc.unimelb.edu.au/
- **Help Desk**: hpc-support@unimelb.edu.au
- **Job Script Generator**: https://dashboard.hpc.unimelb.edu.au/forms/script_generator/

---

**Remember**: Always test your jobs with small datasets first, and never run compute jobs on the login nodes! 🎯
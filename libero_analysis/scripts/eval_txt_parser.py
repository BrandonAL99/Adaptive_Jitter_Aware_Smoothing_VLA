import re
import csv
import sys

# Usage: python eval_txt_parser.py [log_file] [output_csv]
# Defaults to libero_eval_spline/libero_object.log

log_file = sys.argv[1] if len(sys.argv) > 1 else "results/logs/libero_eval_spline/libero_object.log"
out_file  = sys.argv[2] if len(sys.argv) > 2 else log_file.replace(".log", ".csv")

# Matches:  Task 3: pick_up_the_bbq_sauce...
task_header = re.compile(r"Task\s+(\d+):\s+(.+)")

# Matches:  ep 1/10  success=True  sr=100%  Sm_raw=0.00179  Sm_proc=0.00138  SER_proc=0.8462  T=138
ep_line = re.compile(
    r"ep\s+(\d+)/\d+\s+"
    r"success=(True|False)\s+"
    r"sr=(\d+)%\s+"
    r"Sm_raw=([\d.]+)\s+"
    r"Sm_proc=([\d.]+)\s+"
    r"SER_proc=([\d.]+)\s+"
    r"T=(\d+)"
)

rows = []
current_task_id   = None
current_task_name = ""

with open(log_file) as f:
    for line in f:
        line = line.strip()

        m = task_header.search(line)
        if m:
            current_task_id   = int(m.group(1))
            current_task_name = m.group(2).strip()
            continue

        m = ep_line.search(line)
        if m and current_task_id is not None:
            rows.append([
                current_task_id,
                current_task_name,
                int(m.group(1)),          # ep number
                m.group(2).upper(),        # TRUE/FALSE
                int(m.group(3)),           # sr %
                float(m.group(4)),         # Sm_raw
                float(m.group(5)),         # Sm_proc
                float(m.group(6)),         # SER_proc
                int(m.group(7)),           # T
            ])

with open(out_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["task_id", "task_name", "ep", "success", "sr_pct",
                     "Sm_raw", "Sm_proc", "SER_proc", "T"])
    writer.writerows(rows)

print(f"Parsed {len(rows)} episodes → {out_file}")

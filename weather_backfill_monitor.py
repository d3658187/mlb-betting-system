import os
import time
import subprocess
from datetime import datetime

DB_URL = os.environ.get("DATABASE_URL")
if not DB_URL:
    raise SystemExit("DATABASE_URL is required")

SQL = """
select count(*) as missing_count
from games g
left join game_weather w on g.mlb_game_id = w.mlb_game_id
where g.game_datetime is not null and w.mlb_game_id is null;
"""

initial = None
next_report = 100

while True:
    result = subprocess.run(
        ["psql", DB_URL, "-t", "-A", "-c", SQL],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"{datetime.now().isoformat()} | psql error: {result.stderr.strip()}")
        time.sleep(300)
        continue
    output = result.stdout.strip()
    try:
        missing = int(output) if output else None
    except ValueError:
        print(f"{datetime.now().isoformat()} | unexpected output: {output}")
        time.sleep(300)
        continue

    if initial is None and missing is not None:
        initial = missing
        print(f"{datetime.now().isoformat()} | initial missing: {initial}")

    if missing is None or initial is None:
        time.sleep(300)
        continue

    completed = initial - missing
    while completed >= next_report:
        print(f"{datetime.now().isoformat()} | progress: {completed}/{initial} completed (missing {missing})")
        next_report += 100

    if missing == 0:
        print(f"{datetime.now().isoformat()} | DONE: missing 0")
        break

    time.sleep(300)

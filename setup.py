"""
One-command setup: generate data, train both models, initialize the database.

    python setup.py

After this, run:  streamlit run dashboard.py
"""

import subprocess
import sys


def run(module):
    print(f"\n>>> Running {module} ...")
    subprocess.run([sys.executable, "-m", module], check=True)


if __name__ == "__main__":
    run("fraud.train")
    run("credit.train")

    from database.db import init_db
    init_db()

    from database.seed import seed
    seed()

    print("\n✅ Setup complete. Launch with:  streamlit run dashboard.py")

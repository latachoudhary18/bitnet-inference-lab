\# Setup Guide (Windows, PowerShell)



1\. Clone: `git clone https://github.com/latachoudhary18/bitnet-inference-lab.git`

2\. Install Python 3.10 or newer.

3\. Run `pip install -r requirements.txt`

4\. Run the checks (none of these run model inference):

&#x20;  - `python -m pytest -q`

&#x20;  - `python -m scripts.demo\_integration`

&#x20;  - `python -m scripts.benchmark\_report --input results/benchmark\_log.jsonl --output results/benchmark\_summary.json`

5\. `python -m scripts.system\_info` records this machine's hardware profile. It overwrites the existing profile, so run it only if you mean to refresh it.

6\. Build the official BitNet code by following its own README. Commit used: not recorded (to be added by Lata).

7\. Model files are very large and are not in this repo. Keep them in a local `models/` folder.


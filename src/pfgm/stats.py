import json, os, sys, platform
from datetime import datetime

def summarize(outdir="results/tables"):
    os.makedirs(outdir, exist_ok=True)
    info = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    with open(os.path.join(outdir, "build_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("Wrote", os.path.join(outdir, "build_info.json"))

def main():
    summarize()

if __name__ == "__main__":
    main()

"""Download the SciFact dataset (BEIR format) into data/scifact if it is not there yet."""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
OUT = Path("data")


def main():
    if (OUT / "scifact" / "corpus.jsonl").exists():
        print("data/scifact already present")
        return
    print(f"Downloading {URL}")
    try:
        raw = urllib.request.urlopen(URL, timeout=120).read()
    except Exception as exc:
        sys.exit(f"Download failed: {exc!r}")
    OUT.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        z.extractall(OUT)
    assert (OUT / "scifact" / "corpus.jsonl").exists(), "unexpected zip layout"
    print("Done")


if __name__ == "__main__":
    main()
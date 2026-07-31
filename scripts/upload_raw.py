"""Upload raw WikiArt metadata to ADLS as the starting point of the pipeline."""

from pathlib import Path

from src.utils.datalake import DataLakeClient


def main() -> None:
    lake = DataLakeClient()
    local_csv = Path("data/wikiart/classes.csv")

    if not local_csv.exists():
        print(f"Raw metadata not found at {local_csv}")
        return

    print("Uploading raw metadata to ADLS...")
    lake.upload_file(local_csv, "raw/wikiart/classes.csv")
    print("Done.")


if __name__ == "__main__":
    main()
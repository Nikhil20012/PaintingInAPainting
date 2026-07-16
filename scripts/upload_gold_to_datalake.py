"""Upload Gold label CSVs to Azure Data Lake Gen2."""

import os
from pathlib import Path

from azure.storage.filedatalake import DataLakeServiceClient
from dotenv import load_dotenv


def get_datalake_client() -> DataLakeServiceClient:
    load_dotenv()
    account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")

    if not account_name or not account_key:
        raise ValueError("Missing AZURE_STORAGE_ACCOUNT_NAME or AZURE_STORAGE_ACCOUNT_KEY in .env")

    return DataLakeServiceClient(
        account_url=f"https://{account_name}.dfs.core.windows.net",
        credential=account_key,
    )


def upload_file(file_client, local_path: Path) -> None:
    with open(local_path, "rb") as f:
        file_client.upload_data(f, overwrite=True)


def main() -> None:
    container = "painting-data"
    remote_dir = "gold/wikiart/labels"
    local_dir = Path("data/gold/labels")

    files = [
        "gold_wikiart.csv",
        "gold_style_mapping.csv",
        "gold_artist_mapping.csv",
        "gold_genre_mapping.csv",
    ]

    client = get_datalake_client()
    fs_client = client.get_file_system_client(container)

    # ensure remote directory exists
    try:
        fs_client.get_directory_client(remote_dir).create_directory()
        print(f"Created directory: {remote_dir}")
    except Exception:
        print(f"Directory already exists: {remote_dir}")

    for fname in files:
        local_path = local_dir / fname
        if not local_path.exists():
            print(f"  SKIP (not found): {local_path}")
            continue

        remote_path = f"{remote_dir}/{fname}"
        file_client = fs_client.get_file_client(remote_path)
        upload_file(file_client, local_path)
        size_kb = local_path.stat().st_size / 1024
        print(f"  Uploaded: {remote_path} ({size_kb:.1f} KB)")

    print("\nDone. Gold labels are in Azure Data Lake.")


if __name__ == "__main__":
    main()
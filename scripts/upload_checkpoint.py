"""Upload best model checkpoint to Azure Data Lake Gen2."""

import os
from pathlib import Path

from azure.storage.filedatalake import DataLakeServiceClient
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")

    if not account_name or not account_key:
        raise ValueError("Missing Azure credentials in .env")

    client = DataLakeServiceClient(
        account_url=f"https://{account_name}.dfs.core.windows.net",
        credential=account_key,
    )

    container = "painting-data"
    remote_dir = "models/checkpoints"
    local_ckpt = Path("checkpoints/run/best.pth")

    if not local_ckpt.exists():
        print(f"No checkpoint found at {local_ckpt}")
        return

    fs_client = client.get_file_system_client(container)

    try:
        fs_client.get_directory_client(remote_dir).create_directory()
    except Exception:
        pass

    remote_path = f"{remote_dir}/best.pth"
    file_client = fs_client.get_file_client(remote_path)

    with open(local_ckpt, "rb") as f:
        file_client.upload_data(f, overwrite=True)

    size_mb = local_ckpt.stat().st_size / (1024 * 1024)
    print(f"Uploaded: {remote_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
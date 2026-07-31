"""Azure Data Lake Gen2 utility for upload and download operations."""

import os
from pathlib import Path

from azure.storage.filedatalake import DataLakeServiceClient
from dotenv import load_dotenv


class DataLakeClient:
    def __init__(self):
        load_dotenv()
        account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")

        if not account_name or not account_key:
            raise ValueError("Missing AZURE_STORAGE_ACCOUNT_NAME or AZURE_STORAGE_ACCOUNT_KEY in .env")

        self.client = DataLakeServiceClient(
            account_url=f"https://{account_name}.dfs.core.windows.net",
            credential=account_key,
        )
        self.container = "painting-data"
        self.fs = self.client.get_file_system_client(self.container)

    def _ensure_dir(self, remote_dir: str) -> None:
        try:
            self.fs.get_directory_client(remote_dir).create_directory()
        except Exception:
            pass

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        """Upload a single file to ADLS."""
        remote_dir = str(Path(remote_path).parent)
        self._ensure_dir(remote_dir)

        file_client = self.fs.get_file_client(remote_path)
        with open(local_path, "rb") as f:
            file_client.upload_data(f, overwrite=True)

        size_kb = local_path.stat().st_size / 1024
        print(f"  Uploaded: {remote_path} ({size_kb:.1f} KB)")

    def download_file(self, remote_path: str, local_path: Path) -> None:
        """Download a single file from ADLS."""
        local_path.parent.mkdir(parents=True, exist_ok=True)

        file_client = self.fs.get_file_client(remote_path)
        with open(local_path, "wb") as f:
            download = file_client.download_file()
            download.readinto(f)

        size_kb = local_path.stat().st_size / 1024
        print(f"  Downloaded: {remote_path} ({size_kb:.1f} KB)")

    def upload_directory(self, local_dir: Path, remote_dir: str) -> None:
        """Upload all files in a local directory to ADLS."""
        self._ensure_dir(remote_dir)
        for file_path in sorted(local_dir.iterdir()):
            if file_path.is_file():
                remote_path = f"{remote_dir}/{file_path.name}"
                self.upload_file(file_path, remote_path)

    def download_directory(self, remote_dir: str, local_dir: Path) -> None:
        """Download all files from an ADLS directory."""
        local_dir.mkdir(parents=True, exist_ok=True)
        dir_client = self.fs.get_directory_client(remote_dir)

        for item in dir_client.get_paths():
            if not item.is_directory:
                name = Path(item.name).name
                local_path = local_dir / name
                self.download_file(item.name, local_path)
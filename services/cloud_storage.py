"""
Cloud Storage Service for Google Cloud Storage (GCS) and local file system
"""
import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class CloudStorageService:
    """Service for handling file storage in both local and cloud environments"""

    def __init__(self, bucket_name: Optional[str] = None):
        self.bucket_name = bucket_name
        self.is_cloud_mode = os.getenv('CLOUD_MODE', 'false').lower() == 'true'

        if self.is_cloud_mode:
            try:
                from google.cloud import storage
                self.client = storage.Client()
                self.bucket = self.client.bucket(bucket_name)
                logger.info(f"✅ Initialized Google Cloud Storage with bucket: {bucket_name}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Google Cloud Storage: {e}")
                raise
        else:
            # Local mode - ensure directories exist
            os.makedirs('state', exist_ok=True)
            os.makedirs('logs', exist_ok=True)
            logger.info("✅ Initialized local file storage")

    def _get_local_path(self, filename: str) -> str:
        """Get local file path"""
        if filename.startswith('state_'):
            return f"state/{filename}"
        elif filename.endswith('.log'):
            return f"logs/{filename}"
        else:
            return filename

    def read_json(self, filename: str) -> Dict[str, Any]:
        """Read JSON file from storage"""
        try:
            if self.is_cloud_mode:
                blob = self.bucket.blob(filename)
                if not blob.exists():
                    logger.info(f"File {filename} does not exist in GCS, returning empty dict")
                    return {}

                content = blob.download_as_text()
                return json.loads(content)
            else:
                # Local mode
                filepath = self._get_local_path(filename)
                if not os.path.exists(filepath):
                    logger.info(f"File {filepath} does not exist locally, returning empty dict")
                    return {}

                with open(filepath, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")
            return {}

    def write_json(self, filename: str, data: Dict[str, Any]) -> bool:
        """Write JSON file to storage"""
        try:
            if self.is_cloud_mode:
                blob = self.bucket.blob(filename)
                content = json.dumps(data, indent=2)
                blob.upload_from_string(content, content_type='application/json')
                logger.info(f"✅ Uploaded {filename} to GCS")
                return True
            else:
                # Local mode
                filepath = self._get_local_path(filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)

                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"✅ Saved {filename} locally")
                return True
        except Exception as e:
            logger.error(f"Error writing {filename}: {e}")
            return False

    def read_text(self, filename: str) -> str:
        """Read text file from storage"""
        try:
            if self.is_cloud_mode:
                blob = self.bucket.blob(filename)
                if not blob.exists():
                    logger.info(f"File {filename} does not exist in GCS")
                    return ""

                return blob.download_as_text()
            else:
                # Local mode
                filepath = self._get_local_path(filename)
                if not os.path.exists(filepath):
                    logger.info(f"File {filepath} does not exist locally")
                    return ""

                with open(filepath, 'r') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")
            return ""

    def write_text(self, filename: str, content: str) -> bool:
        """Write text file to storage"""
        try:
            if self.is_cloud_mode:
                blob = self.bucket.blob(filename)
                blob.upload_from_string(content, content_type='text/plain')
                logger.info(f"✅ Uploaded {filename} to GCS")
                return True
            else:
                # Local mode
                filepath = self._get_local_path(filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)

                with open(filepath, 'w') as f:
                    f.write(content)
                logger.info(f"✅ Saved {filename} locally")
                return True
        except Exception as e:
            logger.error(f"Error writing {filename}: {e}")
            return False

    def append_text(self, filename: str, content: str) -> bool:
        """Append text to file in storage"""
        try:
            if self.is_cloud_mode:
                blob = self.bucket.blob(filename)
                # Read existing content
                existing_content = ""
                if blob.exists():
                    existing_content = blob.download_as_text()

                # Append new content
                new_content = existing_content + content

                # Upload updated content
                blob.upload_from_string(new_content, content_type='text/plain')
                logger.info(f"✅ Appended to {filename} in GCS")
                return True
            else:
                # Local mode
                filepath = self._get_local_path(filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)

                with open(filepath, 'a') as f:
                    f.write(content)
                logger.info(f"✅ Appended to {filename} locally")
                return True
        except Exception as e:
            logger.error(f"Error appending to {filename}: {e}")
            return False

    def file_exists(self, filename: str) -> bool:
        """Check if file exists in storage"""
        try:
            if self.is_cloud_mode:
                blob = self.bucket.blob(filename)
                return blob.exists()
            else:
                # Local mode
                filepath = self._get_local_path(filename)
                return os.path.exists(filepath)
        except Exception as e:
            logger.error(f"Error checking if {filename} exists: {e}")
            return False

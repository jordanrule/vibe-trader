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

        # Debug: Check cloud mode environment variable
        cloud_mode_env = os.getenv('CLOUD_MODE', 'false')
        logger.info(f"🔍 CLOUD_MODE environment variable: '{cloud_mode_env}'")

        self.is_cloud_mode = cloud_mode_env.lower() == 'true'
        logger.info(f"🔍 Cloud mode detected: {self.is_cloud_mode}")

        if self.is_cloud_mode:
            logger.info(f"🌐 Attempting to initialize cloud storage with bucket: {bucket_name}")
            if not bucket_name:
                logger.error("❌ CLOUD_MODE is true but no bucket_name provided")
                raise ValueError("Bucket name is required in cloud mode")

            try:
                from google.cloud import storage
                logger.info("🔍 Importing Google Cloud Storage client...")
                self.client = storage.Client()
                logger.info("🔍 Creating GCS client...")

                # Debug: Check if client was created successfully
                if self.client:
                    logger.info(f"✅ GCS client created successfully: {self.client}")

                self.bucket = self.client.bucket(bucket_name)
                logger.info(f"✅ GCS bucket reference created for: {bucket_name}")

                # Debug: Test bucket access
                try:
                    bucket_exists = self.bucket.exists()
                    logger.info(f"🔍 Bucket '{bucket_name}' exists: {bucket_exists}")
                except Exception as bucket_test_e:
                    logger.warning(f"⚠️ Could not verify bucket existence: {bucket_test_e}")

                logger.info(f"✅ Initialized Google Cloud Storage with bucket: {bucket_name}")
            except ImportError as ie:
                logger.error(f"❌ Google Cloud Storage import failed: {ie}")
                logger.error("❌ Make sure google-cloud-storage is installed: pip install google-cloud-storage")
                raise
            except Exception as e:
                logger.error(f"❌ Failed to initialize Google Cloud Storage: {e}")
                logger.error("❌ Check your GCP credentials and permissions")
                raise
        else:
            logger.info("💻 Running in LOCAL MODE - initializing local file storage")
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
            logger.info(f"📖 Attempting to read JSON file: {filename}")
            logger.info(f"🔍 Cloud mode: {self.is_cloud_mode}")

            if self.is_cloud_mode:
                logger.info(f"🌐 Reading from GCS bucket: {self.bucket_name}")
                blob = self.bucket.blob(filename)
                logger.info(f"🔍 Checking if blob exists: {filename}")

                blob_exists = blob.exists()
                logger.info(f"🔍 Blob exists: {blob_exists}")

                if not blob_exists:
                    logger.info(f"📄 File {filename} does not exist in GCS, returning empty dict")
                    return {}

                logger.info(f"📥 Downloading content from GCS: {filename}")
                content = blob.download_as_text()
                logger.info(f"📄 Downloaded {len(content)} characters from {filename}")

                result = json.loads(content)
                logger.info(f"✅ Successfully parsed JSON from {filename}")
                return result
            else:
                # Local mode
                logger.info("💻 Reading from local storage")
                filepath = self._get_local_path(filename)
                logger.info(f"📂 Local file path: {filepath}")

                if not os.path.exists(filepath):
                    logger.info(f"📄 File {filepath} does not exist locally, returning empty dict")
                    return {}

                logger.info(f"📖 Reading local file: {filepath}")
                with open(filepath, 'r') as f:
                    result = json.load(f)

                logger.info(f"✅ Successfully read JSON from {filepath}")
                return result
        except Exception as e:
            logger.error(f"❌ Error reading {filename}: {e}")
            logger.error(f"❌ Error type: {type(e)}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return {}

    def write_json(self, filename: str, data: Dict[str, Any]) -> bool:
        """Write JSON file to storage"""
        try:
            logger.info(f"📝 Attempting to write JSON file: {filename}")
            logger.info(f"🔍 Cloud mode: {self.is_cloud_mode}")

            if self.is_cloud_mode:
                logger.info(f"🌐 Writing to GCS bucket: {self.bucket_name}")
                blob = self.bucket.blob(filename)
                logger.info(f"🔍 Created blob reference for: {filename}")

                content = json.dumps(data, indent=2)
                logger.info(f"📄 JSON content length: {len(content)} characters")

                logger.info(f"📤 Uploading to GCS: {filename}")
                blob.upload_from_string(content, content_type='application/json')
                logger.info(f"✅ Successfully uploaded {filename} to GCS")
                return True
            else:
                # Local mode
                logger.info("💻 Writing to local storage")
                filepath = self._get_local_path(filename)
                logger.info(f"📂 Local file path: {filepath}")

                # Only create directories if the filepath has a directory component
                dir_path = os.path.dirname(filepath)
                if dir_path and dir_path != '.':
                    os.makedirs(dir_path, exist_ok=True)
                logger.info(f"📝 Writing JSON data to {filepath}")

                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)

                logger.info(f"✅ Successfully saved {filename} locally")
                return True
        except Exception as e:
            logger.error(f"❌ Error writing {filename}: {e}")
            logger.error(f"❌ Error type: {type(e)}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
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

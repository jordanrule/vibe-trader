"""
Secrets Service for Google Secrets Manager and local environment variables
"""
import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class SecretsService:
    """Service for handling secrets in both local and cloud environments"""

    def __init__(self, project_id: Optional[str] = None):
        self.project_id = project_id or os.getenv('GCP_PROJECT_ID')
        self.is_cloud_mode = os.getenv('CLOUD_MODE', 'false').lower() == 'true'

        if self.is_cloud_mode:
            try:
                from google.cloud import secretmanager
                self.client = secretmanager.SecretManagerServiceClient()
                logger.info(f"✅ Initialized Google Secrets Manager for project: {project_id}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Google Secrets Manager: {e}")
                raise
        else:
            logger.info("✅ Initialized local secrets (environment variables)")

    def get_secret(self, secret_name: str, version: str = 'latest') -> Optional[str]:
        """Get secret value"""
        try:
            if self.is_cloud_mode:
                if not self.project_id:
                    raise ValueError("GCP_PROJECT_ID must be set for cloud mode")

                name = f"projects/{self.project_id}/secrets/{secret_name}/versions/{version}"
                response = self.client.access_secret_version(request={"name": name})
                secret_value = response.payload.data.decode("UTF-8")
                logger.info(f"✅ Retrieved secret: {secret_name}")
                return secret_value
            else:
                # Local mode - get from environment variables
                env_value = os.getenv(secret_name)
                if env_value:
                    logger.info(f"✅ Retrieved local secret: {secret_name}")
                    return env_value
                else:
                    logger.warning(f"⚠️ Secret {secret_name} not found in environment variables")
                    return None
        except Exception as e:
            logger.error(f"❌ Error retrieving secret {secret_name}: {e}")
            return None

    def get_kraken_credentials(self) -> Dict[str, str]:
        """Get Kraken API credentials"""
        api_key = self.get_secret('KRAKEN_API_KEY')
        api_secret = self.get_secret('KRAKEN_API_SECRET')

        if not api_key or not api_secret:
            raise ValueError("Kraken API credentials not found in secrets")

        return {
            'api_key': api_key,
            'api_secret': api_secret
        }

    def get_openai_credentials(self) -> Dict[str, str]:
        """Get OpenAI API credentials"""
        api_key = self.get_secret('OPENAI_API_KEY')

        if not api_key:
            raise ValueError("OpenAI API key not found in secrets")

        return {
            'api_key': api_key
        }

    def get_telegram_credentials(self) -> Dict[str, str]:
        """Get Telegram API credentials"""
        api_token = self.get_secret('TELEGRAM_BOT_TOKEN')

        if not api_token:
            raise ValueError("Telegram bot token not found in secrets")

        return {
            'api_token': api_token
        }

    def get_gcp_config(self) -> Dict[str, str]:
        """Get GCP configuration"""
        bucket_name = self.get_secret('GCS_BUCKET_NAME')
        project_id = self.get_secret('GCP_PROJECT_ID') or self.project_id

        return {
            'bucket_name': bucket_name,
            'project_id': project_id
        }

    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration from secrets"""
        config = {}

        # Try to get all credentials
        try:
            config.update(self.get_kraken_credentials())
        except ValueError:
            logger.warning("Kraken credentials not available")

        try:
            config.update(self.get_openai_credentials())
        except ValueError:
            logger.warning("OpenAI credentials not available")

        try:
            config.update(self.get_telegram_credentials())
        except ValueError:
            logger.warning("Telegram credentials not available")

        try:
            config.update(self.get_gcp_config())
        except ValueError:
            logger.warning("GCP config not available")

        # Add other configuration values
        config.update({
            'live_mode': self.get_secret('LIVE_MODE', 'false').lower() == 'true',
            'max_trade_lifetime_hours': int(self.get_secret('MAX_TRADE_LIFETIME_HOURS', '6')),
            'log_level': self.get_secret('LOG_LEVEL', 'INFO'),
            'stop_loss_percentage': float(self.get_secret('STOP_LOSS_PERCENTAGE', '20.0')),
        })

        return config

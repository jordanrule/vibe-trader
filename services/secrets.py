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
        # Detect environment
        self.is_cloud_mode = os.getenv('CLOUD_MODE', 'false').lower() == 'true'

        # Resolve project ID from multiple sources
        self.project_id = (
            project_id
            or os.getenv('GCP_PROJECT_ID')
            or os.getenv('GOOGLE_CLOUD_PROJECT')
        )

        if self.is_cloud_mode and not self.project_id:
            # Try to infer from default credentials
            try:
                import google.auth
                _, inferred_project = google.auth.default()
                self.project_id = inferred_project or self.project_id
            except Exception:
                pass

        if self.is_cloud_mode:
            try:
                from google.cloud import secretmanager
                self.client = secretmanager.SecretManagerServiceClient()
                logger.info(f"✅ Initialized Google Secrets Manager for project: {self.project_id or 'UNKNOWN'}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Google Secrets Manager: {e}")
                raise
        else:
            logger.info("✅ Initialized local secrets (environment variables)")

    def get_secret(self, secret_name: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret value with safe fallback.

        Note: 'default' is a fallback value, NOT a Secret Manager version alias.
        """
        try:
            if self.is_cloud_mode:
                if not self.project_id:
                    logger.warning("GCP project ID not set; cannot fetch secrets from Secret Manager")
                    return default

                name = f"projects/{self.project_id}/secrets/{secret_name}/versions/latest"
                response = self.client.access_secret_version(request={"name": name})
                secret_value = response.payload.data.decode("UTF-8")
                logger.info(f"✅ Retrieved secret: {secret_name}")
                return secret_value
            else:
                # Local mode - get from environment variables
                env_value = os.getenv(secret_name)
                if env_value is not None:
                    logger.info(f"✅ Retrieved local secret: {secret_name}")
                    return env_value
                else:
                    logger.warning(f"⚠️ Secret {secret_name} not found in environment variables; using default")
                    return default
        except Exception as e:
            logger.error(f"❌ Error retrieving secret {secret_name}: {e}")
            return default

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

        # Try to get all credentials and normalize keys to app-wide names
        try:
            kraken = self.get_kraken_credentials()
            if kraken:
                # Normalize to expected keys
                config['kraken_api_key'] = kraken.get('api_key') or kraken.get('kraken_api_key')
                config['kraken_api_secret'] = kraken.get('api_secret') or kraken.get('kraken_api_secret')
        except ValueError:
            logger.warning("Kraken credentials not available")

        try:
            openai = self.get_openai_credentials()
            if openai:
                config['openai_api_key'] = openai.get('api_key') or openai.get('openai_api_key')
        except ValueError:
            logger.warning("OpenAI credentials not available")

        try:
            telegram = self.get_telegram_credentials()
            if telegram:
                config['telegram_token'] = telegram.get('api_token') or telegram.get('telegram_token')
        except ValueError:
            logger.warning("Telegram credentials not available")

        try:
            config.update(self.get_gcp_config())
        except ValueError:
            logger.warning("GCP config not available")

        # Add other configuration values with safe parsing and defaults
        live_mode_raw = self.get_secret('LIVE_MODE', 'false')
        live_mode = str(live_mode_raw).strip().lower() == 'true'

        max_hours_raw = self.get_secret('MAX_TRADE_LIFETIME_HOURS', '6')
        try:
            max_hours = int(str(max_hours_raw).strip())
        except Exception:
            max_hours = 6

        log_level = self.get_secret('LOG_LEVEL', 'INFO')

        stop_raw = self.get_secret('STOP_LOSS_PERCENTAGE', '20.0')
        try:
            stop_loss = float(str(stop_raw).strip())
        except Exception:
            stop_loss = 20.0

        config.update({
            'live_mode': live_mode,
            'max_trade_lifetime_hours': max_hours,
            'log_level': log_level,
            'stop_loss_percentage': stop_loss,
        })

        return config

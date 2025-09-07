"""
Telegram service for message monitoring and processing
"""
import json
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional
from .base import BaseService

logger = logging.getLogger(__name__)

class TelegramService(BaseService):
    """Service for handling Telegram Bot API interactions"""
    
    def __init__(self, config: Dict[str, str]):
        super().__init__(config)
        self.telegram_token = config.get('telegram_token')
        self.telegram_chat_id = config.get('telegram_chat_id')
        self.cloud_storage = config.get('cloud_storage')  # Will be passed from TradingAgent
        self.last_update_id = 0

        if not self.telegram_token:
            raise ValueError("telegram_token is required")

        # Load last update ID from disk
        self.last_update_id = self.load_last_update_id()
    
    def load_last_update_id(self) -> int:
        """Load the last processed Telegram update ID"""
        try:
            if self.cloud_storage:
                data = self.cloud_storage.read_json('state_last_update_id.json')
                return data.get('last_update_id', 0) if data else 0
            else:
                # Fallback for backward compatibility
                with open('state_last_update_id.json', 'r') as f:
                    data = json.load(f)
                    return data.get('last_update_id', 0)
        except FileNotFoundError:
            return 0
        except Exception as e:
            logger.error(f"Error loading last update ID: {e}")
            return 0

    def save_last_update_id(self):
        """Save the last processed Telegram update ID"""
        try:
            if self.cloud_storage:
                self.cloud_storage.write_json('state_last_update_id.json', {'last_update_id': self.last_update_id})
            else:
                # Fallback for backward compatibility
                with open('state_last_update_id.json', 'w') as f:
                    json.dump({'last_update_id': self.last_update_id}, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving last update ID: {e}")
    
    def check_messages(self) -> List[Dict]:
        """
        Check for new Telegram messages using last_update_id.

        Returns messages received since the last processed update.
        """
        try:
            logger.info("Checking for new Telegram messages (using last_update_id)")
            return self._get_recent_updates()
        except Exception as e:
            logger.error(f"Error checking Telegram messages: {e}")
            return []
    
    def _get_recent_updates(self) -> List[Dict]:
        """Get recent updates using getUpdates with offset tracking"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/getUpdates"
            params = {
                'timeout': 0,  # No long polling for historical data
                'limit': 100,   # Get up to 100 recent updates
                'offset': self.last_update_id + 1  # Only get updates after last processed
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get('ok'):
                logger.error(f"Telegram API error: {data}")
                return []
            
            updates = data.get('result', [])
            messages = []
            
            if updates:
                # Update the last processed update ID
                self.last_update_id = max(update['update_id'] for update in updates)
                logger.info(f"Updated last_update_id to {self.last_update_id}")
                # Save the updated ID
                self.save_last_update_id()
            
            for update in updates:
                if 'message' in update:
                    message = update['message']
                    message_timestamp = message.get('date', 0)
                    message_date = datetime.fromtimestamp(message_timestamp)

                    messages.append({
                        'update_id': update['update_id'],
                        'message_id': message['message_id'],
                        'text': message.get('text', ''),
                        'date': message_date,
                        'timestamp': message_timestamp,
                        'from': message.get('from', {}),
                        'chat': message.get('chat', {})
                    })

                    # Print the message for debugging
                    user = message.get('from', {})
                    username = user.get('username', user.get('first_name', 'Unknown'))
                    print(f"📱 Message from {username} ({message_date}): {message.get('text', '')}")

            logger.info(f"Found {len(messages)} new messages since last_update_id")
            return messages
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error checking Telegram messages: {e}")
            return []
        except Exception as e:
            logger.error(f"Error checking Telegram messages: {e}")
            return []

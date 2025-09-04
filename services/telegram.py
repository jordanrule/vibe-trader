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
        self.last_update_id = 0
        
        if not self.telegram_token:
            raise ValueError("telegram_token is required")
        
        # Load last update ID from disk
        self.last_update_id = self.load_last_update_id()
    
    def load_last_update_id(self) -> int:
        """Load the last processed Telegram update ID"""
        try:
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
            with open('state_last_update_id.json', 'w') as f:
                json.dump({'last_update_id': self.last_update_id}, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving last update ID: {e}")
    
    def check_messages(self, from_time: datetime, to_time: datetime) -> List[Dict]:
        """
        Check for Telegram messages between from_time and to_time
        
        Note: Telegram Bot API only supports recent messages via getUpdates.
        Historical messages by time range are not supported.
        
        Args:
            from_time: Start timestamp to collect messages from
            to_time: End timestamp to collect messages to
        """
        try:
            logger.info(f"Checking for Telegram messages between {from_time} and {to_time}")
            return self._get_recent_updates(from_time, to_time)
            
        except Exception as e:
            logger.error(f"Error checking Telegram messages: {e}")
            return []
    
    def _get_recent_updates(self, from_time: datetime, to_time: datetime) -> List[Dict]:
        """Get recent updates using getUpdates method with proper offset tracking"""
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
            
            # Convert to timestamps for comparison
            from_timestamp = from_time.timestamp()
            to_timestamp = to_time.timestamp()
            
            for update in updates:
                if 'message' in update:
                    message = update['message']
                    message_timestamp = message.get('date', 0)
                    message_date = datetime.fromtimestamp(message_timestamp)
                    
                    # Only process messages within the specified time range
                    if from_timestamp <= message_timestamp <= to_timestamp:
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
            
            logger.info(f"Found {len(messages)} recent messages between {from_time} and {to_time}")
            return messages
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error checking Telegram messages: {e}")
            return []
        except Exception as e:
            logger.error(f"Error checking Telegram messages: {e}")
            return []

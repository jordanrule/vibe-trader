#!/usr/bin/env python3
"""
Telegram Sentiment Trading Agent - Main Cycle
Implements the 5-step trading cycle
"""

import argparse
import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Import services
from services.telegram import TelegramService
from services.kraken import KrakenService
from services.openai import OpenAIService
from services.opportunity import OpportunityService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingAgent:
    """Trading agent implementing the 5-step cycle"""
    
    def __init__(self):
        # Load configuration from environment
        self.config = {
            'telegram_token': os.getenv('TELEGRAM_BOT_TOKEN'),
            'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID'),
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'kraken_api_key': os.getenv('KRAKEN_API_KEY'),
            'kraken_secret': os.getenv('KRAKEN_SECRET'),
            'live_mode': os.getenv('LIVE_MODE', 'false').lower() == 'true',
            'max_trade_lifetime_hours': int(os.getenv('MAX_TRADE_LIFETIME_HOURS', '6')),
            'stop_loss_percentage': float(os.getenv('STOP_LOSS_PERCENTAGE', '20'))
        }
        
        # Validate required configuration
        if not self.config['telegram_token']:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")

        # Log configuration
        logger.info(f"TradingAgent initialized (max trade lifetime: {self.config['max_trade_lifetime_hours']}h)")
        logger.info(f"Stop-loss configured: {self.config['stop_loss_percentage']}% below entry price")
        logger.info(f"Live trading mode: {self.config['live_mode']}")
        
        # Initialize services
        self.telegram_service = TelegramService(self.config)
        self.kraken_service = KrakenService(self.config)
        self.openai_service = OpenAIService(self.config)
        self.opportunity_service = OpportunityService(self.config)
        
        logger.info(f"TradingAgent initialized (max trade lifetime: {self.config['max_trade_lifetime_hours']}h)")
    
    async def run_cycle(self, from_time: datetime, to_time: datetime, is_backtest: bool = False):
        """
        Main trading cycle - 5 steps only
        
        Args:
            from_time: Start time to collect messages from
            to_time: End time to collect messages to
            is_backtest: If True, use historical data for analysis
        """
        logger.info(f"Starting trading cycle from {from_time} to {to_time}")
        
        try:
            # Step 1: Cleanup expired opportunities and record PnL
            await self._step1_cleanup_expired_opportunities()
            
            # Step 2: Update bandit model based on PnL outcomes
            await self._step2_update_bandit_model()
            
            # Step 3: Check for new opportunities from Telegram messages
            await self._step3_check_new_opportunities(from_time, to_time)
            
            # Step 4: Get RAG recommendation for position switching
            recommendation = await self._step4_get_rag_recommendation()

            # Step 5: Execute market orders if switching recommended
            if recommendation and recommendation.get('action') == 'switch':
                await self._step5_execute_market_orders(recommendation)
            else:
                logger.info("No position switching recommended - holding current positions")
            
            logger.info("Trading cycle completed successfully")
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    async def _step1_cleanup_expired_opportunities(self):
        """Step 1: Cleanup expired opportunities and record PnL"""
        logger.info("Step 1: Cleaning up expired opportunities and recording PnL")
        
        # Cleanup expired opportunities and record PnL
        self.opportunity_service.cleanup_expired_opportunities(self.kraken_service)
        
        logger.info(f"Step 1 completed - {len(self.opportunity_service.opportunities)} opportunities remaining")
    
    async def _step2_update_bandit_model(self):
        """Step 2: Update bandit model based on PnL outcomes"""
        logger.info("Step 2: Updating bandit model from PnL outcomes")
        
        # Update bandit model from recorded outcomes
        self.opportunity_service.update_bandit_model_from_outcomes()
        
        logger.info("Step 2 completed - bandit model updated")
    
    async def _step3_check_new_opportunities(self, from_time: datetime, to_time: datetime):
        """Step 3: Check for new opportunities from Telegram messages"""
        logger.info("Step 3: Checking for new opportunities from Telegram messages")
        
        # Get Telegram messages
        messages = self.telegram_service.check_messages(from_time, to_time)
        
        if not messages:
            logger.info("No new messages to process")
            return
        
        logger.info(f"Processing {len(messages)} messages for new opportunities")
        
        # Get tradeable pairs for detection
        all_pairs = self.kraken_service.get_tradeable_pairs()
        detection_pairs = self._create_detection_pairs(all_pairs)
        crypto_keywords = self.opportunity_service.get_all_crypto_keywords(detection_pairs)
        
        # Process each message
        new_opportunities_created = 0
        for msg in messages:
            opportunities_created = await self._process_message_for_opportunities(msg, crypto_keywords, detection_pairs)
            new_opportunities_created += opportunities_created
        
        logger.info(f"Step 3 completed - created {new_opportunities_created} new opportunities")
    
    def _create_detection_pairs(self, all_pairs: Dict) -> Dict:
        """Create detection pairs (USD and USDT pairs only)"""
        detection_pairs = {}
        for pair_id, pair_info in all_pairs.items():
            pair_name = pair_info.get('wsname', pair_id)
            base = pair_info.get('base', '')
            quote = pair_info.get('quote', '')
            
            if quote in ['ZUSD', 'USDT']:
                detection_pairs[pair_name] = {
                    'pair_id': pair_id,
                    'base': base,
                    'quote': quote,
                    'min_order_size': pair_info.get('ordermin', '0')
                }
        return detection_pairs
    
    async def _process_message_for_opportunities(self, msg: Dict, crypto_keywords: Dict, detection_pairs: Dict) -> int:
        """Process a single message for opportunities"""
        message_text = msg.get('text', '')
        user = msg.get('from', {})
        username = user.get('username', user.get('first_name', 'Unknown'))
        
        logger.info(f"Processing message from {username}: {message_text}")
        
        # Detect mentioned cryptocurrencies
        detected_assets = self._detect_cryptocurrencies(message_text, crypto_keywords, detection_pairs)
        
        if not detected_assets:
            logger.info("No cryptocurrencies detected in message")
            return 0
        
        # Process each detected asset
        opportunities_created = 0
        for base_currency, pair_name, pair_id in detected_assets:
            if await self._create_opportunity_from_asset(message_text, base_currency, pair_name, pair_id):
                opportunities_created += 1
        
        return opportunities_created
    
    def _detect_cryptocurrencies(self, message_text: str, crypto_keywords: Dict, detection_pairs: Dict) -> List[tuple]:
        """Detect cryptocurrencies mentioned in a message"""
        detected_assets = []
        message_lower = message_text.lower()
        
        # Common English words to exclude
        common_words = {
            'a', 'an', 'the', 'to', 'of', 'and', 'or', 'but', 'in', 'on', 'at', 'is', 'are', 'was', 'were',
            'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'can', 'must', 'shall', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
            'its', 'our', 'their', 'am', 'good', 'go', 'ok', 'hi', 'hey', 'sir', 'sent', 'how', 'what',
            'when', 'where', 'why', 'who', 'with', 'from', 'for', 'as', 'by', 'up', 'out', 'so', 'no',
            'not', 'all', 'any', 'get', 'got', 'how', 'now', 'new', 'old', 'big', 'own', 'say', 'way',
            'use', 'man', 'day', 'get', 'use', 'her', 'new', 'now', 'old', 'see', 'him', 'two', 'how',
            'its', 'who', 'oil', 'sit', 'set', 'hot', 'let', 'say', 'she', 'may', 'try', 'ask', 'too',
            'own', 'put', 'end', 'why', 'let', 'run', 'keep', 'feel', 'fact', 'hand', 'high', 'year',
            'work', 'life', 'call', 'last', 'help', 'away', 'move', 'make', 'live', 'back', 'only',
            'over', 'also', 'want', 'seem', 'give', 'take', 'come', 'show', 'tell', 'part', 'look',
            'know', 'find', 'think', 'thank', 'okay', 'investment', 'invest', 'process', 'works',
            'deposit', 'ready'
        }
        
        for base_currency, keywords in crypto_keywords.items():
            for keyword in keywords:
                # Skip single characters and common words
                if len(keyword) <= 1 or keyword in common_words:
                    continue
                
                # Use word boundaries for better matching
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, message_lower):
                    pair_name = self._find_pair_by_asset(base_currency, detection_pairs)
                    if pair_name:
                        detected_assets.append((base_currency, pair_name, detection_pairs[pair_name]['pair_id']))
                        break  # Only add each asset once per message
        
        return detected_assets
    
    def _find_pair_by_asset(self, asset: str, detection_pairs: Dict) -> Optional[str]:
        """Find pair name for an asset"""
        for pair_name, pair_info in detection_pairs.items():
            if pair_info['base'] == asset:
                return pair_name
        return None
    
    async def _create_opportunity_from_asset(self, message: str, asset: str, pair_name: str, pair_id: str) -> bool:
        """Create opportunity from detected asset"""
        try:
            # Get current price and technical indicators
            current_price = await self._get_current_price(asset)
            if not current_price:
                logger.warning(f"Could not get current price for {asset}")
                return False
            
            # Get technical indicators
            candles = self.kraken_service.get_candles(pair_id, interval=15, count=720)
            if not candles or len(candles) < 20:
                logger.warning(f"Insufficient candle data for {asset}")
                return False
            
            technical_indicators = self.opportunity_service.calculate_technical_indicators(candles)
            if not technical_indicators:
                logger.warning(f"Could not calculate technical indicators for {asset}")
                return False
            
            # Analyze sentiment with OpenAI
            analysis = self.openai_service.analyze_sentiment(message, asset, pair_name, technical_indicators)
            if not analysis:
                logger.info(f"No analysis returned for {asset}")
                return False
            
            # Only create opportunities for BUY/LONG recommendations
            if analysis.get('action', '').lower() not in ['buy', 'long']:
                logger.info(f"Skipping {asset} - not a BUY/LONG recommendation")
                return False
            
            # Create opportunity
            opportunity_id = self.opportunity_service.create_opportunity(analysis, asset, pair_name, current_price)
            
            if opportunity_id:
                logger.info(f"Created opportunity {opportunity_id} for {asset}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error creating opportunity for {asset}: {e}")
            return False
    
    def _map_to_common_asset(self, asset: str) -> str:
        """Map internal Kraken asset codes to common names for display"""
        # Comprehensive mapping for all X/XX prefixed assets on Kraken
        kraken_to_common = {
            # XX prefixed (double X)
            'XXBT': 'BTC',    # Bitcoin (display as BTC, but API uses XBT)
            'XXRP': 'XRP',    # Ripple
            'XXMR': 'XMR',    # Monero
            'XXDG': 'DOGE',   # Dogecoin
            'XXLM': 'XLM',    # Stellar
            'XYO': 'XYO',     # XYO Network

            # X prefixed (single X)
            'XETH': 'ETH',    # Ethereum
            'XLTC': 'LTC',    # Litecoin
            'XETC': 'ETC',    # Ethereum Classic
            'XREP': 'REP',    # Augur
            'XTZ': 'XTZ',     # Tezos
            'XZEC': 'ZEC',    # Zcash
            'XCN': 'XCN',     # Cryptonex
            'XDC': 'XDC',     # XinFin
            'XMLN': 'MLN',    # Melon
            'XNY': 'XNY',     # Nyancoin
            'XRT': 'XRT',     # Robonomics
            'XTER': 'TER',    # Terran Coin
        }

        return kraken_to_common.get(asset, asset)  # Return mapped name or original if no mapping
    
    async def _get_current_price(self, asset: str) -> Optional[float]:
        """Get current price for an asset"""
        try:
            common_asset = self._map_to_common_asset(asset)
            pair_id = self.kraken_service.get_pair_for_asset(common_asset)
            ticker = self.kraken_service.get_ticker(pair_id)
            
            if ticker and 'c' in ticker:
                return float(ticker['c'][0])  # Current price
            return None
        except Exception as e:
            logger.error(f"Error getting current price for {asset}: {e}")
            return None
    
    async def _step4_get_rag_recommendation(self) -> Optional[Dict]:
        """Step 4: Get RAG recommendation for position switching"""
        logger.info("Step 4: Getting RAG recommendation for position switching")

        # Get RAG recommendation
        recommendation = self.opportunity_service.get_rag_recommendation(self.kraken_service, self.openai_service)

        if recommendation:
            logger.info(f"RAG recommendation: {recommendation.get('action', 'unknown')} - {recommendation.get('reasoning', 'no reasoning')}")
        else:
            logger.info("No RAG recommendation available")

        return recommendation
    
    async def _step5_execute_market_orders(self, recommendation: Dict):
        """Step 5: Execute market orders for position switching"""
        logger.info("Step 5: Executing market orders for position switching")

        try:
            # Get the single recommended asset from OpenAI
            recommended_assets = recommendation.get('recommended_assets', [])
            if not recommended_assets or len(recommended_assets) != 1:
                logger.warning("No single asset recommendation received from OpenAI")
                return

            recommended_asset = recommended_assets[0]
            logger.info(f"🎯 Single recommended asset: {recommended_asset}")

            # Get current positions
            balances = self.kraken_service.get_available_balances()
            current_positions = {}
            total_balances = {}

            # Define fiat/stable assets
            fiat_assets = {'ZUSD', 'USDT', 'USDC'}

            for asset, balance_entry in balances.items():
                try:
                    if isinstance(balance_entry, dict):
                        balance = float(balance_entry.get('balance', 0) or 0)
                        hold_trade = float(balance_entry.get('hold_trade', 0) or 0)
                        available_bal = balance - hold_trade
                    else:
                        available_bal = float(balance_entry)
                        balance = available_bal
                except (TypeError, ValueError):
                    continue

                # Include all assets with balance > 0 (both available and total)
                if balance > 0:
                    current_positions[asset] = available_bal  # Available balance
                    total_balances[asset] = balance          # Total balance (including held)

            logger.info(f"📊 Current positions: {list(current_positions.keys())}")

            # Step 1: Liquidate ALL positions that are NOT the recommended asset
            assets_to_liquidate = []
            for asset in current_positions.keys():
                # Don't liquidate fiat assets (we'll use them for buying)
                if asset in fiat_assets:
                    continue

                # Check if this asset matches the recommended asset
                if self._assets_match(asset, recommended_asset):
                    logger.info(f"⏸️  Keeping recommended asset: {asset}")
                    continue

                assets_to_liquidate.append(asset)

            # Execute liquidation of non-recommended assets
            if assets_to_liquidate:
                logger.info(f"💰 Liquidating {len(assets_to_liquidate)} non-recommended assets: {assets_to_liquidate}")
                total_liquidated = 0

                for asset in assets_to_liquidate:
                    available_balance = current_positions.get(asset, 0)
                    total_balance = total_balances.get(asset, 0)

                    # Always attempt liquidation if there's any total balance (even if available is 0)
                    if total_balance > 0:
                        logger.info(f"🎯 Liquidating {asset}: {available_balance:.8f} available, {total_balance:.8f} total")
                        liquidated_amount = await self._execute_complete_liquidation(asset, available_balance, total_balance)
                        if liquidated_amount > 0:
                            total_liquidated += liquidated_amount
                        else:
                            logger.warning(f"⚠️ Liquidation of {asset} returned $0.00 - balance may be held by orders")
                    else:
                        logger.info(f"ℹ️ Skipping {asset} - no balance to liquidate")

                logger.info(f"✅ Liquidation completed: ${total_liquidated:.2f} in proceeds")
            else:
                logger.info("✅ No non-recommended assets to liquidate")

            # Step 2: Allocate ALL available cash to the recommended asset
            await asyncio.sleep(2)  # Wait for liquidations to settle
            balances = self.kraken_service.get_available_balances()

            # Calculate total available cash
            total_cash = 0
            for asset, balance_entry in balances.items():
                if asset not in fiat_assets:
                    continue

                try:
                    if isinstance(balance_entry, dict):
                        balance = float(balance_entry.get('balance', 0) or 0)
                        hold_trade = float(balance_entry.get('hold_trade', 0) or 0)
                        available_bal = balance - hold_trade
                    else:
                        available_bal = float(balance_entry)

                    if available_bal > 0.05:  # Count any meaningful cash amounts
                        total_cash += available_bal
                        logger.info(f"💵 Available {asset}: ${available_bal:.2f}")
                except (TypeError, ValueError):
                    continue

            if total_cash > 0.01:
                logger.info(f"💰 Total available cash: ${total_cash:.2f}")
                logger.info(f"🎯 Allocating 100% to {recommended_asset}")

                # Apply a more conservative buffer to account for stop-loss margin and timing issues
                buffered_cash = total_cash * 0.90  # 90% buffer instead of 95%
                logger.info(f"🛡️ Using buffered cash: ${buffered_cash:.2f} (90% of ${total_cash:.2f})")

                usd_used = await self._execute_complete_reallocation(recommended_asset, buffered_cash)
                if usd_used > 0:
                    logger.info(f"✅ Complete allocation successful: ${usd_used:.2f} deployed to {recommended_asset}")
                else:
                    logger.error(f"❌ Complete allocation failed for {recommended_asset}")
            else:
                logger.info("✅ No significant cash available for allocation")

            logger.info("Step 5 completed - market orders executed")

        except Exception as e:
            logger.error(f"Error executing market orders: {e}")

    def _assets_match(self, asset1: str, asset2: str) -> bool:
        """Check if two assets are the same (handles different naming conventions)"""
        # Normalize both assets for comparison
        def normalize_asset(asset):
            return asset.replace('X', '').replace('.F', '')

        return normalize_asset(asset1) == normalize_asset(asset2)

    def _matches_asset_variations(self, asset: str, order_pair: str, common_asset: str = None) -> bool:
        """
        Generalized asset matching that handles various naming conventions:
        - .F suffix indicates futures contracts (e.g., ETH.F -> ETH)
        - X prefix indicates extended naming (e.g., XETH -> ETH)
        - Matches base asset or base_asset + USD in order pairs
        """
        # Direct asset name matching
        if asset in order_pair or (common_asset and common_asset in order_pair):
            return True

        # Get base asset by normalizing the input asset
        def normalize_asset(asset_name):
            """Normalize asset name by removing X prefix and .F suffix"""
            # Remove X prefix if present (extended naming convention)
            if asset_name.startswith('X'):
                asset_name = asset_name[1:]
            # Remove .F suffix if present (futures contract)
            if asset_name.endswith('.F'):
                asset_name = asset_name[:-2]
            return asset_name

        base_asset = normalize_asset(asset)

        # Check for base asset or base_asset + USD in order pair
        if base_asset in order_pair or f"{base_asset}USD" in order_pair:
            return True

        # Also check the original asset name variations in the pair
        if asset in order_pair:
            return True

        return False
    
    async def _execute_complete_liquidation(self, asset: str, available_balance: float, total_balance: float = 0):
        """Execute complete liquidation of an asset position using iterative smart orders"""
        try:
            remaining_balance = available_balance  # Start with available balance
            common_asset = self._map_to_common_asset(asset)
            pair_id = self.kraken_service.get_pair_for_asset(common_asset)

            if not pair_id:
                logger.error(f"Could not find pair for {asset}")
                return 0

            logger.info(f"🎯 Starting complete liquidation of {asset} (available: {available_balance:.8f}, total: {total_balance:.8f})")

            # If available balance is 0 but we have total balance, we may need to cancel orders first
            needs_order_cancellation = (available_balance == 0 and total_balance > 0)
            if needs_order_cancellation:
                logger.info(f"🔄 Available balance is 0 but total balance is {total_balance:.8f} - will attempt order cancellation first")

            # Cancel ALL open orders for this asset first - this is critical
            open_orders = self.kraken_service.get_open_orders()
            cancelled_orders = []

            for order_id, order_info in open_orders.items():
                order_desc = order_info.get('descr', {})
                order_pair = order_desc.get('pair', '')
                order_type = order_desc.get('type', '')
                close_info = order_desc.get('close', '')

                # Check if this order is for our asset (be more comprehensive with matching)
                asset_matches = self._matches_asset_variations(asset, order_pair, common_asset)

                # Check for conditional close orders (stop-loss orders)
                # These might have different pair names but still hold our asset
                if not asset_matches and close_info:
                    # If this is a stop-loss order, it might be holding our asset
                    # Check if the close price suggests it's related to our asset
                    try:
                        if 'stop-loss' in str(close_info).lower():
                            # This is likely a stop-loss order that could be holding our asset
                            asset_matches = True
                            logger.info(f"🎯 Found potential stop-loss order {order_id} that may be holding {asset}")
                    except:
                        pass

                if asset_matches:
                    logger.info(f"❌ Cancelling order {order_id} for {asset} (pair: {order_pair}, type: {order_type})")
                    if self.kraken_service.cancel_order(order_id):
                        cancelled_orders.append(order_id)
                        logger.info(f"✅ Cancelled order {order_id}")
                    else:
                        logger.error(f"❌ Failed to cancel order {order_id}")

            # Additional check: Look for any remaining orders that might be stop-loss related
            if not cancelled_orders:
                logger.info("No orders cancelled - checking for hidden stop-loss orders...")
                # Try to get all orders including closed ones that might have conditional closes
                # This is a fallback mechanism

            # Wait for cancellations to take effect
            if cancelled_orders:
                logger.info(f"⏳ Waiting 5 seconds for {len(cancelled_orders)} order cancellations to process...")
                await asyncio.sleep(5)

            # Additional retry: Check again for any remaining orders that might have been missed
            logger.info("🔄 Performing secondary order cancellation check...")
            open_orders = self.kraken_service.get_open_orders()
            additional_cancelled = []

            for order_id, order_info in open_orders.items():
                order_desc = order_info.get('descr', {})
                order_pair = order_desc.get('pair', '')

                # More aggressive matching for any remaining orders using generalized asset matching
                if self._matches_asset_variations(asset, order_pair, common_asset):

                    logger.info(f"🔄 Cancelling additional order {order_id} for {asset} (pair: {order_pair})")
                    if self.kraken_service.cancel_order(order_id):
                        additional_cancelled.append(order_id)
                        logger.info(f"✅ Cancelled additional order {order_id}")
                    else:
                        logger.error(f"❌ Failed to cancel additional order {order_id}")

            if additional_cancelled:
                logger.info(f"⏳ Waiting additional 3 seconds for {len(additional_cancelled)} more cancellations...")
                await asyncio.sleep(3)

            # Refresh balances after all cancellations
                balances = self.kraken_service.get_available_balances()

                # Update available balance after cancellations
                for bal_asset, bal_entry in balances.items():
                    # Match the asset we're trying to liquidate using generalized asset matching
                    asset_match = self._assets_match(asset, bal_asset)

                    if asset_match:
                        if isinstance(bal_entry, dict):
                            new_balance = float(bal_entry.get('balance', 0) or 0)
                            new_hold = float(bal_entry.get('hold_trade', 0) or 0)
                            new_available = new_balance - new_hold
                            logger.info(f"📊 After all cancellations - {bal_asset}: {new_balance:.8f} total, {new_hold:.8f} hold, {new_available:.8f} available")
                            if new_available > remaining_balance:
                                released = new_available - remaining_balance
                                logger.info(f"✅ Released {released:.8f} {asset} from orders")
                                remaining_balance = new_available
                        else:
                            new_available = float(bal_entry or 0)
                            logger.info(f"📊 After all cancellations - {bal_asset}: {new_available:.8f} available")
                            remaining_balance = new_available

            # Now try to liquidate any available balance (after potential order cancellations)
            if remaining_balance > 0:
                logger.info(f"💰 Attempting to liquidate {remaining_balance} {asset} using smart orders")

                # For complete liquidation, don't use stop-loss to ensure full execution
                order_id = self.kraken_service.place_smart_order(pair_id, 'sell', remaining_balance, stop_loss=None)

                # If the order still fails with insufficient funds, try one final cancellation of ALL orders
                if not order_id:
                    logger.warning("❌ Liquidation order failed - attempting emergency cancellation of ALL open orders")
                    emergency_cancelled = []
                    open_orders = self.kraken_service.get_open_orders()

                    # Cancel ALL open orders as emergency measure
                    for order_id, order_info in open_orders.items():
                        logger.info(f"🚨 Emergency cancelling ALL order {order_id}")
                        if self.kraken_service.cancel_order(order_id):
                            emergency_cancelled.append(order_id)
                        else:
                            logger.error(f"❌ Failed emergency cancellation of {order_id}")

                    if emergency_cancelled:
                        logger.info(f"⏳ Waiting 5 seconds after emergency cancellation of {len(emergency_cancelled)} orders...")
                        await asyncio.sleep(5)

                        # Try liquidation one more time after emergency cancellation
                        final_balances = self.kraken_service.get_available_balances()
                        final_available = 0
                        for bal_asset, bal_entry in final_balances.items():
                            if self._assets_match(asset, bal_asset):
                                try:
                                    if isinstance(bal_entry, dict):
                                        balance = float(bal_entry.get('balance', 0) or 0)
                                        hold_trade = float(bal_entry.get('hold_trade', 0) or 0)
                                        final_available = balance - hold_trade
                                    else:
                                        final_available = float(bal_entry or 0)
                                    break
                                except (TypeError, ValueError):
                                    continue

                        if final_available > 0:
                            logger.info(f"🚨 Final emergency liquidation attempt with {final_available} {asset}")
                            order_id = self.kraken_service.place_market_order(pair_id, 'sell', final_available, stop_loss=None)
            elif total_balance > 0 and remaining_balance == 0:
                # All balance is held by orders, but we already tried to cancel them above
                # Check one more time if any balance was released
                final_balances = self.kraken_service.get_available_balances()
                final_available = 0
                for bal_asset, bal_entry in final_balances.items():
                    if self._assets_match(asset, bal_asset):
                        try:
                            if isinstance(bal_entry, dict):
                                balance = float(bal_entry.get('balance', 0) or 0)
                                hold_trade = float(bal_entry.get('hold_trade', 0) or 0)
                                final_available = balance - hold_trade
                            else:
                                final_available = float(bal_entry or 0)
                            break
                        except (TypeError, ValueError):
                            continue

                if final_available > 0:
                    logger.info(f"💰 After order cancellation, {final_available} {asset} became available - liquidating")
                    order_id = self.kraken_service.place_smart_order(pair_id, 'sell', final_available, stop_loss=None)
                else:
                    # If no available balance but we have total balance, try to liquidate total balance
                    if total_balance > 0:
                        logger.warning(f"⚠️ No available balance but total balance exists ({total_balance}) - attempting market liquidation")
                        order_id = self.kraken_service.place_market_order(pair_id, 'sell', total_balance)
                        if order_id:
                            logger.info(f"✅ Market liquidation order placed for total balance: {order_id}")
                            # Calculate approximate proceeds for return value (we'll get actual from polling)
                            return total_balance * 0.001  # Rough estimate, actual value will be tracked elsewhere
                        else:
                            logger.error(f"❌ Failed to place market liquidation order for total balance")
                            return 0
                    else:
                        logger.info(f"ℹ️ {asset} still has no available balance after order cancellation (total: {total_balance})")
                        return 0
            else:
                logger.info(f"ℹ️ No balance to liquidate for {asset}")
                return 0

            if order_id:
                logger.info(f"✅ Complete liquidation sell order placed for {asset}: {order_id}")

                # Wait up to 5 minutes for the order to complete
                logger.info("⏳ Polling for order completion (up to 5 minutes)...")
                start_time = asyncio.get_event_loop().time()
                timeout = 300  # 5 minutes

                while asyncio.get_event_loop().time() - start_time < timeout:
                    order_status = self.kraken_service.get_order_status(order_id)
                    if order_status and order_status.get('status') == 'closed':
                        executed_volume = float(order_status.get('vol_exec', 0))
                        logger.info(f"✅ Liquidation order completed: executed {executed_volume} {asset}")
                        return executed_volume
                    elif order_status and order_status.get('status') in ['canceled', 'expired']:
                        logger.warning(f"❌ Liquidation order was cancelled/expired: {order_id}")
                        return 0

                    await asyncio.sleep(10)  # Check every 10 seconds

                # Order timed out
                logger.warning(f"⏰ Liquidation order timed out after 5 minutes: {order_id}")
                # Cancel the order
                if self.kraken_service.cancel_order(order_id):
                    logger.info(f"✅ Cancelled timed-out liquidation order: {order_id}")
                return 0
            else:
                logger.error(f"❌ Failed to place liquidation sell order for {asset}")
                return 0
            # No else clause needed here - the above if/else covers all cases

        except Exception as e:
            logger.error(f"Error in complete liquidation for {asset}: {e}")
            return 0

    async def _execute_complete_reallocation(self, asset: str, usd_balance: float):
        """Execute complete capital reallocation to a single asset with full stop-loss protection"""
        try:
            logger.info(f"🎯 Starting complete reallocation of ${usd_balance:.2f} to {asset}")

            common_asset = self._map_to_common_asset(asset)
            pair_id = self.kraken_service.get_pair_for_asset(common_asset)
            if not pair_id:
                logger.error(f"Could not find pair for {asset}")
                return 0

            # Get current price
            current_price = await self._get_current_price(asset)
            if not current_price or current_price <= 0:
                logger.error(f"Could not get current price for {asset}")
                return 0

            # Use 100% of available USD for complete reallocation to single asset
            spendable_usd = usd_balance
            volume = spendable_usd / current_price

            # Round down volume and check minimum order size
            volume = self.kraken_service.round_down_volume(volume, pair_id)
            ordermin = self.kraken_service.get_ordermin(pair_id)

            if volume <= 0 or (ordermin and volume < ordermin):
                logger.error(f"Volume {volume} too small for {asset} (minimum: {ordermin})")
                return 0

            # Calculate stop-loss based on configuration
            stop_loss = current_price * (1 - self.config['stop_loss_percentage'] / 100)

            logger.info(f"📊 Buying {volume:.6f} {asset} with 100% of funds: ${spendable_usd:.2f}")
            logger.info(f"🛡️ Stop-loss: ${stop_loss:.4f} ({self.config['stop_loss_percentage']}% below entry)")

            # Get fresh balance right before placing order to ensure accuracy
            fresh_balances = self.kraken_service.get_available_balances()
            fresh_usd_balance = 0
            for bal_asset, bal_entry in fresh_balances.items():
                if bal_asset in ['ZUSD', 'USDT', 'USDC']:
                    try:
                        if isinstance(bal_entry, dict):
                            bal = float(bal_entry.get('balance', 0) or 0)
                            hold = float(bal_entry.get('hold_trade', 0) or 0)
                            available = bal - hold
                        else:
                            available = float(bal_entry)

                        if available > 0.01:
                            fresh_usd_balance += available
                    except (TypeError, ValueError):
                        continue

            # Check if we still have enough balance for this order
            required_usd = volume * current_price
            if required_usd > fresh_usd_balance * 0.95:  # Allow 5% buffer
                logger.warning(f"⚠️ Required USD (${required_usd:.2f}) exceeds fresh balance (${fresh_usd_balance:.2f})")
                logger.info("🔄 Adjusting volume to fit available balance")
                adjusted_volume = (fresh_usd_balance * 0.90) / current_price  # Use 90% of fresh balance
                adjusted_volume = self.kraken_service.round_down_volume(adjusted_volume, pair_id)
                volume = adjusted_volume

            # Place smart buy order with comprehensive stop-loss
            order_id = self.kraken_service.place_smart_order(pair_id, 'buy', volume, stop_loss)

            if order_id:
                usd_used = volume * current_price
                logger.info(f"✅ Complete reallocation buy order placed: {order_id}")
                logger.info(f"💸 USD deployed: ${usd_used:.2f}")
                return usd_used  # Return USD amount used
            else:
                logger.warning(f"❌ Initial allocation failed - trying with more conservative volume")
                # Try with 80% of the volume as a fallback
                fallback_volume = volume * 0.8
                logger.info(f"🔄 Fallback: Trying with {fallback_volume:.6f} {asset} (80% of original)")

                if fallback_volume >= (ordermin or 0):
                    fallback_order_id = self.kraken_service.place_smart_order(pair_id, 'buy', fallback_volume, stop_loss)
                    if fallback_order_id:
                        usd_used = fallback_volume * current_price
                        logger.info(f"✅ Fallback reallocation successful: {fallback_order_id}")
                        logger.info(f"💸 USD deployed: ${usd_used:.2f}")
                        return usd_used

                logger.error(f"❌ Failed to place reallocation buy order for {asset} even with fallback")
                return 0

        except Exception as e:
            logger.error(f"Error in complete reallocation to {asset}: {e}")
            return 0

    async def _execute_market_buy(self, asset: str):
        """Execute market buy order with stop-loss"""
        try:
            common_asset = self._map_to_common_asset(asset)
            pair_id = self.kraken_service.get_pair_for_asset(common_asset)
            if not pair_id:
                logger.error(f"Could not find pair for {asset}")
                return

            # Get current price for stop-loss calculation
            current_price = await self._get_current_price(asset)
            if not current_price:
                logger.error(f"Could not get current price for {asset}")
                return

            # Calculate volume (simplified - use fixed amount)
            volume = 100.0 / current_price  # $100 worth

            # Calculate stop-loss based on configuration
            stop_loss = current_price * (1 - self.config['stop_loss_percentage'] / 100)

            # Place smart buy order with stop-loss (limit first, market fallback)
            order_id = self.kraken_service.place_smart_order(pair_id, 'buy', volume, stop_loss)

            if order_id:
                logger.info(f"Smart buy order placed for {asset}: {order_id} (stop-loss: ${stop_loss:.4f}, {self.config['stop_loss_percentage']}% below)")
            else:
                logger.error(f"Failed to place smart buy order for {asset}")

        except Exception as e:
            logger.error(f"Error executing market buy for {asset}: {e}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Telegram Sentiment Trading Agent')
    
    parser.add_argument(
        '--from', '-f',
        dest='from_time',
        type=str,
        default=None,
        help='Start time in ISO format (default: 15 minutes ago)'
    )

    parser.add_argument(
        '--to', '-t',
        dest='to_time',
        type=str,
        default=None,
        help='End time in ISO format (default: now)'
    )
    
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Use historical data for analysis (for backtesting)'
    )
    
    return parser.parse_args()

async def main():
    """Main function - runs the trading cycle with default time range"""
    args = parse_arguments()
    agent = TradingAgent()

    try:
        # Use default time range: last 15 minutes
        current_time = datetime.now()
        from_time = current_time - timedelta(minutes=15)
        to_time = current_time

        logger.info(f"Processing messages from {from_time} to {to_time} (backtest: {args.backtest})")

        # Run the trading cycle
        await agent.run_cycle(from_time, to_time, args.backtest)
        
        logger.info("Trading cycle completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

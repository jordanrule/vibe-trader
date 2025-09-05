"""
Kraken API service for trading operations and market data
"""
import base64
import math
import hashlib
import hmac
import json
import logging
import requests
import time
import urllib.parse
from typing import Dict, List, Optional
from .base import BaseService

logger = logging.getLogger(__name__)

class KrakenService(BaseService):
    """Service for handling Kraken API interactions"""
    
    def __init__(self, config: Dict[str, str]):
        super().__init__(config)
        self.api_key = config.get('kraken_api_key')
        self.secret = config.get('kraken_secret')
        self.live_mode = config.get('live_mode', False)
        
        # Load pair cache
        self._pair_lookup_cache = self.load_pair_cache()
        
        if not self.api_key or not self.secret:
            logger.warning("Kraken API credentials not configured - trading will be disabled")
    
    def load_pair_cache(self) -> Dict[str, str]:
        """Load pair lookup cache from disk"""
        try:
            with open('state_pair_cache.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.error(f"Error loading pair cache: {e}")
            return {}
    
    def save_pair_cache(self):
        """Save pair lookup cache to disk"""
        try:
            with open('state_pair_cache.json', 'w') as f:
                json.dump(self._pair_lookup_cache, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving pair cache: {e}")
    
    def sign_request(self, urlpath: str, data: Dict) -> str:
        """Generate Kraken API signature"""
        if not self.secret:
            raise ValueError("KRAKEN_SECRET not configured")
        
        # Create POST data string
        postdata = urllib.parse.urlencode(data)
        
        # SHA256 of (nonce + POST data) - this is the critical fix
        # The nonce should be just the nonce value, not including "nonce=" prefix
        nonce_str = str(data['nonce'])
        sha256_input = (nonce_str + postdata).encode()
        sha256_digest = hashlib.sha256(sha256_input).digest()
        
        # HMAC-SHA512 of (URI path + SHA256 digest)
        message = urlpath.encode() + sha256_digest
        
        secret = base64.b64decode(self.secret)
        signature = hmac.new(secret, message, hashlib.sha512)
        
        return base64.b64encode(signature.digest()).decode()
    
    def private_request(self, endpoint: str, data: Dict = None) -> Dict:
        """Make authenticated request to Kraken private API"""
        if not self.api_key or not self.secret:
            raise ValueError("KRAKEN_API_KEY and KRAKEN_SECRET must be configured for trading")
        
        if data is None:
            data = {}
        
        # Add nonce
        data['nonce'] = str(int(time.time() * 1000))
        
        url = f"https://api.kraken.com/0/private/{endpoint}"
        urlpath = f"/0/private/{endpoint}"
        
        headers = {
            'API-Key': self.api_key,
            'API-Sign': self.sign_request(urlpath, data),
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        try:
            response = requests.post(url, headers=headers, data=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get('error'):
                logger.error(f"Kraken API error: {result['error']}")
                return {}
            
            return result.get('result', {})
            
        except Exception as e:
            logger.error(f"Kraken private API request failed: {e}")
            return {}
    
    def public_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make request to Kraken public API"""
        if params is None:
            params = {}
        
        url = f"https://api.kraken.com/0/public/{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get('error'):
                logger.error(f"Kraken API error: {result['error']}")
                return {}
            
            return result.get('result', {})
            
        except Exception as e:
            logger.error(f"Kraken public API request failed: {e}")
            return {}
    
    def get_pair_id(self, pair_name: str) -> Optional[str]:
        """Get the correct Kraken pair ID for a given pair name (with caching)"""
        if pair_name in self._pair_lookup_cache:
            return self._pair_lookup_cache[pair_name]
        
        try:
            # Fetch all pairs if cache is empty
            if not self._pair_lookup_cache:
                all_pairs = self.public_request('AssetPairs')
                
                # Build cache
                for p_id, p_info in all_pairs.items():
                    ws_name = p_info.get('wsname', '')
                    if ws_name:
                        self._pair_lookup_cache[ws_name] = p_id
                
                # Save cache to disk for faster loading
                self.save_pair_cache()
                logger.info(f"Built pair lookup cache with {len(self._pair_lookup_cache)} pairs")
            
            return self._pair_lookup_cache.get(pair_name)
            
        except Exception as e:
            logger.error(f"Error building pair lookup cache: {e}")
            return None
    
    def map_asset_to_pair(self, asset: str) -> str:
        """
        Map internal Kraken asset codes to proper trading pairs
        
        Args:
            asset: Internal Kraken asset code (e.g., 'XETH', 'XXBT', 'HFT')
            
        Returns:
            Proper trading pair (e.g., 'ETH/USD', 'BTC/USD', 'HFT/USD')
        """
        # Map internal Kraken asset codes to their common names
        # Note: Kraken uses XBT internally for Bitcoin, not BTC
        asset_mapping = {
            'XETH': 'ETH',
            'XXBT': 'XBT',  # Keep as XBT for cache lookup
            'XBT': 'XBT',
            'XLTC': 'LTC',
            'XREP': 'REP',
            'XXRP': 'XRP',
            'XXLM': 'XLM',
            'XZEC': 'ZEC',
            'XETC': 'ETC',
            'XSTR': 'STR',
            'XMR': 'XMR',
            'XDAI': 'DAI',
            'XUSDT': 'USDT',
            'XUSDC': 'USDC',
            'XPYUSD': 'PYUSD',
            'XADA': 'ADA',
            'XDOT': 'DOT',
            'XLINK': 'LINK',
            'XUNI': 'UNI',
            'XMATIC': 'MATIC',
            'XAVAX': 'AVAX',
            'XATOM': 'ATOM',
            'XNEAR': 'NEAR',
            'XFTM': 'FTM',
            'XALGO': 'ALGO',
            'XVET': 'VET',
            'XTHETA': 'THETA',
            'XFIL': 'FIL',
            'XICP': 'ICP',
            'XAPT': 'APT',
            'XSUI': 'SUI',
            'XARB': 'ARB',
            'XOP': 'OP',
            'XSTRK': 'STRK',
            'XIMX': 'IMX',
            'XMASK': 'MASK',
            'XINJ': 'INJ',
            'XTIA': 'TIA',
            'XJUP': 'JUP',
            'XPYTH': 'PYTH',
            'XWIF': 'WIF',
            'XBONK': 'BONK',
            'XPEPE': 'PEPE',
            'XSHIB': 'SHIB',
            'XDOGE': 'DOGE',
            'XHFT': 'HFT',
            'XSOL': 'SOL'
        }
        
        # Get the common name, or use the original if no mapping exists
        common_name = asset_mapping.get(asset, asset)
        
        # Return the proper trading pair format
        return f"{common_name}/USD"
    
    def get_pair_for_asset(self, asset: str) -> str:
        """
        Get the Kraken trading pair for a given asset

        Args:
            asset: Asset symbol (e.g., 'ETH', 'BTC', 'HFT', 'ETH.F')

        Returns:
            Kraken pair ID (e.g., 'XETHZUSD', 'XXBTZUSD', 'HFTUSD')
        """
        # Handle futures contracts
        if asset.endswith('.F'):
            # For futures, try to use the spot equivalent
            base_asset = asset[:-2]  # Remove '.F' suffix
            logger.info(f"Treating {asset} as spot equivalent {base_asset} for trading")

            # Just use the spot trading pair for now
            # ETH.F -> ETH -> XETHZUSD
            asset = base_asset

        # Handle spot trading pairs
        pair_name = self.map_asset_to_pair(asset)

        # Get the Kraken pair ID
        pair_id = self.get_pair_id(pair_name)

        if not pair_id:
            logger.warning(f"Could not find Kraken pair for {asset} ({pair_name})")
            return None

        return pair_id
    
    def get_price_precision(self, pair: str) -> int:
        """Get price precision for a trading pair"""
        try:
            pairs = self.public_request('AssetPairs')
            pair_info = pairs.get(pair, {})
            return pair_info.get('pair_decimals', 2)
        except Exception as e:
            logger.error(f"Error getting price precision for {pair}: {e}")
            return 2  # Default precision

    def round_price(self, price: float, precision: int) -> float:
        """Round price to the specified decimal precision"""
        return round(price, precision)

    def get_current_price(self, pair: str) -> Optional[float]:
        """Get current market price for a trading pair"""
        try:
            # Use Ticker endpoint to get current price
            result = self.public_request('Ticker', {'pair': pair})

            if result and pair in result:
                ticker_data = result[pair]
                # Use 'c' (last trade closed) price
                last_price = ticker_data.get('c', [None])[0]
                if last_price:
                    return float(last_price)

            logger.warning(f"Could not get current price for pair {pair}")
            return None

        except Exception as e:
            logger.error(f"Error getting current price for {pair}: {e}")
            return None

    def get_volume_precision(self, pair: str) -> int:
        """Get volume precision for a trading pair"""
        try:
            pairs = self.public_request('AssetPairs')
            pair_info = pairs.get(pair, {})
            return pair_info.get('lot_decimals', 8)
        except Exception as e:
            logger.error(f"Error getting volume precision for {pair}: {e}")
            return 8  # Default precision
    
    def get_ordermin(self, pair: str) -> float:
        """Get minimum order size for a trading pair"""
        try:
            pairs = self.public_request('AssetPairs')
            pair_info = pairs.get(pair, {})
            ordermin = pair_info.get('ordermin')
            if ordermin is None:
                return 0.0
            return float(ordermin)
        except Exception as e:
            logger.error(f"Error getting ordermin for {pair}: {e}")
            return 0.0
    
    def round_volume(self, volume: float, pair: str) -> float:
        """Round volume to appropriate precision"""
        precision = self.get_volume_precision(pair)
        return round(volume, precision)

    def round_down_volume(self, volume: float, pair: str) -> float:
        """Round volume down to avoid exceeding available balance due to rounding"""
        try:
            precision = self.get_volume_precision(pair)
            factor = 10 ** precision
            return math.floor(float(volume) * factor) / factor
        except Exception:
            return max(0.0, float(volume))
    
    def place_market_order(self, pair: str, order_type: str, volume: float, stop_loss: Optional[float] = None) -> Optional[str]:
        """
        Place a market order on Kraken with optional stop-loss

        Args:
            pair: Trading pair (e.g., 'XXBTZUSD')
            order_type: 'buy' or 'sell'
            volume: Amount to trade (in base currency)
            stop_loss: Optional stop loss price

        Returns:
            Order ID if successful, None otherwise
        """
        try:
            # Round volume down to appropriate precision to avoid insufficient funds
            rounded_volume = self.round_down_volume(volume, pair)
            ordermin = self.get_ordermin(pair)
            if rounded_volume <= 0 or (ordermin and rounded_volume < ordermin):
                logger.warning(f"Skipping {order_type} market order for {pair} - volume {rounded_volume} below ordermin {ordermin}")
                return None

            data = {
                'pair': pair,
                'type': order_type,
                'ordertype': 'market',
                'volume': str(rounded_volume)
            }

            # Add conditional close (stop-loss) if provided
            if stop_loss:
                # Get price precision for this pair and round stop-loss price
                price_precision = self.get_price_precision(pair)
                rounded_stop_loss = self.round_price(stop_loss, price_precision)

                if order_type == 'buy':
                    # For buy orders, stop-loss is a sell order
                    data['close[ordertype]'] = 'stop-loss'
                    data['close[price]'] = str(rounded_stop_loss)
                else:  # sell orders
                    # For sell orders, stop-loss is a buy order
                    data['close[ordertype]'] = 'stop-loss'
                    data['close[price]'] = str(rounded_stop_loss)

                logger.info(f"Placing {order_type} market order: {rounded_volume} {pair} with stop-loss at ${rounded_stop_loss}")
            else:
                logger.info(f"Placing {order_type} market order: {rounded_volume} {pair}")

            # Handle paper trading mode vs live trading mode
            if not self.live_mode:
                # Paper trading mode - simulate order placement
                simulated_order_id = f"PAPER_{pair}_{int(time.time())}"
                logger.info(f"📝 Paper trading mode: Simulated order {simulated_order_id}")
                return simulated_order_id
            else:
                # Live trading mode - place real order
                result = self.private_request('AddOrder', data)

                if result and 'txid' in result:
                    order_id = result['txid'][0]
                    logger.info(f"Market order placed successfully: {order_id}")
                    return order_id
                else:
                    logger.error(f"Failed to place market order: {result}")
                    return None

        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return None

    def place_limit_order(self, pair: str, order_type: str, volume: float, price: float, stop_loss: Optional[float] = None) -> Optional[str]:
        """
        Place a limit order on Kraken with optional stop-loss

        Args:
            pair: Trading pair (e.g., 'XXBTZUSD')
            order_type: 'buy' or 'sell'
            volume: Amount to trade (in base currency)
            price: Limit price
            stop_loss: Optional stop loss price

        Returns:
            Order ID if successful, None otherwise
        """
        try:
            # Round volume down to appropriate precision to avoid insufficient funds
            rounded_volume = self.round_down_volume(volume, pair)
            ordermin = self.get_ordermin(pair)
            if rounded_volume <= 0 or (ordermin and rounded_volume < ordermin):
                logger.warning(f"Skipping {order_type} limit order for {pair} - volume {rounded_volume} below ordermin {ordermin}")
                return None

            # Round price to appropriate precision
            price_precision = self.get_price_precision(pair)
            rounded_price = self.round_price(price, price_precision)

            data = {
                'pair': pair,
                'type': order_type,
                'ordertype': 'limit',
                'volume': str(rounded_volume),
                'price': str(rounded_price)
            }

            # Add conditional close (stop-loss) if provided
            if stop_loss:
                rounded_stop_loss = self.round_price(stop_loss, price_precision)

                if order_type == 'buy':
                    # For buy orders, stop-loss is a sell order
                    data['close[ordertype]'] = 'stop-loss'
                    data['close[price]'] = str(rounded_stop_loss)
                else:  # sell orders
                    # For sell orders, stop-loss is a buy order
                    data['close[ordertype]'] = 'stop-loss'
                    data['close[price]'] = str(rounded_stop_loss)

                logger.info(f"Placing {order_type} limit order: {rounded_volume} {pair} at ${rounded_price} with stop-loss at ${rounded_stop_loss}")
            else:
                logger.info(f"Placing {order_type} limit order: {rounded_volume} {pair} at ${rounded_price}")

            # Handle paper trading mode vs live trading mode
            if not self.live_mode:
                # Paper trading mode - simulate order placement
                simulated_order_id = f"PAPER_LIMIT_{pair}_{int(time.time())}"
                logger.info(f"📝 Paper trading mode: Simulated limit order {simulated_order_id}")
                return simulated_order_id
            else:
                # Live trading mode - place real order
                result = self.private_request('AddOrder', data)

                if result and 'txid' in result:
                    order_id = result['txid'][0]
                    logger.info(f"Limit order placed successfully: {order_id}")
                    return order_id
                else:
                    logger.error(f"Failed to place limit order: {result}")
                    return None

        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return None

    def get_open_orders(self) -> Dict:
        """Get all open orders"""
        try:
            result = self.private_request('OpenOrders')
            return result.get('open', {})
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return {}
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order"""
        try:
            data = {'txid': order_id}
            result = self.private_request('CancelOrder', data)
            
            if result and 'count' in result:
                count = result['count']
                logger.info(f"Cancelled {count} orders")
                return count > 0
            else:
                logger.error(f"Failed to cancel order: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    def get_account_balance(self) -> Dict:
        """Get account balance"""
        try:
            logger.info("Attempting to get account balance from Kraken API...")
            result = self.private_request('Balance')
            if result:
                logger.info(f"Successfully retrieved account balance: {len(result)} balance entries")
            else:
                logger.warning("Account balance request returned empty result")
            return result
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return {}
    
    def get_available_balances(self) -> Dict:
        """Get available (free) balances using BalanceEx if possible, else fallback to Balance"""
        try:
            logger.info("Attempting to get available balances from Kraken API (BalanceEx)...")
            result = self.private_request('BalanceEx')
            if result:
                logger.info(f"Successfully retrieved available balances: {len(result)} entries")
                return result
            # Fallback
            logger.warning("BalanceEx returned empty result, falling back to Balance")
            return self.get_account_balance()
        except Exception as e:
            logger.warning(f"BalanceEx not available or failed: {e}. Falling back to Balance")
            return self.get_account_balance()

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get the status of a specific order"""
        try:
            data = {'txid': order_id}
            result = self.private_request('QueryOrders', data)
            if result and order_id in result:
                return result[order_id]
            return None
        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {e}")
            return None
    
    def get_tradeable_pairs(self) -> Dict[str, Dict]:
        """Get list of all tradeable asset pairs from Kraken"""
        try:
            response = requests.get("https://api.kraken.com/0/public/AssetPairs", timeout=15)
            response.raise_for_status()
            data = response.json()
            return data.get('result', {})
        except Exception as e:
            logger.error(f"Error fetching tradeable pairs: {e}")
            return {}
    
    def get_candles(self, pair: str, interval: int = 15, count: int = 720) -> List[Dict]:
        """Get OHLC candle data for a trading pair"""
        try:
            url = f"https://api.kraken.com/0/public/OHLC"
            params = {
                'pair': pair,
                'interval': interval,
                'count': count
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            if data.get('error'):
                logger.error(f"Kraken API error getting candles: {data['error']}")
                return []
            
            result = data.get('result', {})
            if not result:
                return []
            
            # Get the first (and usually only) pair data
            pair_data = list(result.values())[0]
            
            candles = []
            for candle in pair_data:
                candles.append({
                    'time': int(candle[0]),
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[6])
                })
            
            return candles
            
        except Exception as e:
            logger.error(f"Error getting candles for {pair}: {e}")
            return []
    
    def get_ticker(self, pair: str) -> Dict:
        """Get ticker data for a trading pair"""
        try:
            result = self.public_request('Ticker', {'pair': pair})
            if result:
                # Get the first (and usually only) pair data
                return list(result.values())[0]
            return {}
        except Exception as e:
            logger.error(f"Error getting ticker for {pair}: {e}")
            return {}
    
    def is_order_filled(self, order_id: str) -> bool:
        """Check if an order has been filled"""
        try:
            # Handle paper trading order IDs
            if order_id.startswith('PAPER_'):
                logger.info(f"Paper trading order {order_id} - simulating as filled")
                return True
            
            # Check open orders
            open_orders = self.get_open_orders()
            
            # If order is not in open orders, it's likely filled or cancelled
            if order_id not in open_orders:
                logger.info(f"Order {order_id} not found in open orders - assuming filled")
                return True
            
            logger.info(f"Order {order_id} still open")
            return False
            
        except Exception as e:
            logger.error(f"Error checking if order {order_id} is filled: {e}")
            return False

    def wait_for_order_completion(self, order_id: str, timeout_seconds: int = 300, poll_interval: int = 10) -> bool:
        """
        Wait for an order to be filled or cancelled, with timeout

        Args:
            order_id: The order ID to monitor
            timeout_seconds: Maximum time to wait (default 5 minutes)
            poll_interval: How often to check status (default 10 seconds)

        Returns:
            True if order was filled, False if timed out or cancelled
        """
        import time

        start_time = time.time()
        logger.info(f"Waiting for order {order_id} to complete (timeout: {timeout_seconds}s)")

        while time.time() - start_time < timeout_seconds:
            if self.is_order_filled(order_id):
                logger.info(f"Order {order_id} completed successfully")
                return True

            logger.info(f"Order {order_id} still pending, waiting {poll_interval}s...")
            time.sleep(poll_interval)

        # Timeout reached
        logger.warning(f"Order {order_id} timed out after {timeout_seconds}s")

        # Try to cancel the order
        if self.cancel_order(order_id):
            logger.info(f"Cancelled timed-out order {order_id}")
        else:
            logger.warning(f"Failed to cancel timed-out order {order_id}")

        return False

    def place_smart_order(self, pair: str, order_type: str, volume: float, stop_loss: Optional[float] = None) -> Optional[str]:
        """
        Intelligently place orders: try iterative limit orders, fallback to market order

        For buy orders: Try progressively more aggressive limit prices
        For sell orders: Try progressively more aggressive limit prices
        Poll for 2 minutes per iteration, then fallback to market order if all fail

        Args:
            pair: Trading pair (e.g., 'XXBTZUSD')
            order_type: 'buy' or 'sell'
            volume: Amount to trade (in base currency)
            stop_loss: Optional stop loss price

        Returns:
            Order ID if successful, None otherwise
        """
        try:
            # Iterative approach with progressively more aggressive pricing
            max_iterations = 3

            for iteration in range(max_iterations):
                # Get fresh price for each iteration
                current_price = self.get_current_price(pair)
                if not current_price:
                    logger.warning(f"Could not get current price for {pair}, falling back to market order")
                    return self.place_market_order(pair, order_type, volume, stop_loss)

                # Calculate limit price with progressive aggression
                # Start with 0.5%, increase by 0.1% per iteration for more aggressive pricing
                price_offset_pct = 0.005 + (iteration * 0.001)

                if order_type == 'buy':
                    limit_price = current_price * (1 + price_offset_pct)
                    logger.info(f"🔄 Iteration {iteration + 1}: Limit buy at ${limit_price:.4f} (current: ${current_price:.4f})")
                else:  # sell
                    limit_price = current_price * (1 - price_offset_pct)
                    logger.info(f"🔄 Iteration {iteration + 1}: Limit sell at ${limit_price:.4f} (current: ${current_price:.4f})")

                # Try limit order
                limit_order_id = self.place_limit_order(pair, order_type, volume, limit_price, stop_loss)
                if not limit_order_id:
                    logger.warning(f"Limit order failed on iteration {iteration + 1}")
                    if iteration == max_iterations - 1:
                        logger.warning("All limit order attempts failed, falling back to market order")
                        return self.place_market_order(pair, order_type, volume, stop_loss)
                    continue

                # Wait for limit order to complete (2 minutes per iteration)
                if self.wait_for_order_completion(limit_order_id, timeout_seconds=120):
                    logger.info(f"✅ Limit order {limit_order_id} completed successfully on iteration {iteration + 1}")
                    return limit_order_id
                else:
                    logger.warning(f"❌ Limit order {limit_order_id} timed out on iteration {iteration + 1}")
                    if iteration == max_iterations - 1:
                        logger.warning("All limit order iterations timed out, falling back to market order")
                        return self.place_market_order(pair, order_type, volume, stop_loss)

            # This should never be reached, but fallback just in case
            logger.warning("Unexpected end of iterations, falling back to market order")
            return self.place_market_order(pair, order_type, volume, stop_loss)

        except Exception as e:
            logger.error(f"Error in smart order placement: {e}")
            # Fallback to market order on any error
            return self.place_market_order(pair, order_type, volume, stop_loss)


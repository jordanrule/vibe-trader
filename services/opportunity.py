"""
Opportunity service for managing trading opportunities and positions
"""
import json
import logging
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .base import BaseService

logger = logging.getLogger(__name__)

class OpportunityService(BaseService):
    """Service for managing trading opportunities and positions"""
    
    def __init__(self, config: Dict[str, any]):
        super().__init__(config)
        self.max_trade_lifetime_hours = config.get('max_trade_lifetime_hours', 6)
        self.opportunities = self.load_opportunities()
        self.bandit_source_stats = self.load_bandit_model()
    
    def load_opportunities(self) -> Dict:
        """Load trading opportunities from storage"""
        try:
            with open('state_opportunities.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.error(f"Error loading opportunities: {e}")
            return {}
    
    def save_opportunities(self):
        """Save trading opportunities to storage"""
        try:
            with open('state_opportunities.json', 'w') as f:
                json.dump(self.opportunities, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving opportunities: {e}")
    
    def load_bandit_model(self) -> Dict:
        """Load bandit model source statistics"""
        try:
            with open('state_bandit_model.json', 'r') as f:
                data = json.load(f)
                source_stats = data.get('source_stats', {})
                logger.info(f"Loaded bandit model with {len(source_stats)} sources")
                return source_stats
        except FileNotFoundError:
            logger.info("No bandit model found, starting fresh")
            return {}
        except Exception as e:
            logger.error(f"Error loading bandit model: {e}")
            return {}
    
    def save_bandit_model(self):
        """Save bandit model to disk"""
        try:
            with open('state_bandit_model.json', 'w') as f:
                json.dump({'source_stats': self.bandit_source_stats}, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving bandit model: {e}")
    
    def has_opportunity(self, asset: str) -> bool:
        """Check if we have an opportunity for the given asset"""
        for opportunity in self.opportunities.values():
            if opportunity.get('asset') == asset:
                return True
        return False
    
    def create_opportunity(self, analysis: Dict, asset: str, pair_name: str, current_price: float) -> str:
        """Create a new trading opportunity from analysis"""
        try:
            # Never create USDT/USD opportunities (USDT is a stablecoin)
            if asset == 'USDT':
                logger.info(f"🚫 Skipping USDT opportunity creation - USDT is a stablecoin, not a tradeable opportunity")
                return None

            # Generate unique opportunity ID
            timestamp = int(datetime.now().timestamp())
            opportunity_id = f"{asset}_{timestamp}"

            # Simple opportunity: buy at current price, hold for max_trade_lifetime_hours
            opportunity = {
                'id': opportunity_id,
                'asset': asset,
                'pair_name': pair_name,
                'entry_price': current_price,
                'created_at': datetime.now().isoformat(),
                'source': analysis.get('source', 'Unknown'),
                'sentiment': analysis.get('sentiment', 'neutral'),
                'confidence': analysis.get('confidence', 0.5),
                'action': 'long',  # Always long for simplicity
                'duration_hours': self.max_trade_lifetime_hours,
                'reasoning': analysis.get('reasoning', ''),
                'status': 'pending'
            }

            # Store opportunity
            self.opportunities[opportunity_id] = opportunity
            self.save_opportunities()

            logger.info(f"Created opportunity {opportunity_id} for {asset} at ${current_price:.4f} (will hold {self.max_trade_lifetime_hours}h)")
            return opportunity_id

        except Exception as e:
            logger.error(f"Error creating opportunity: {e}")
            return None
    
    def cleanup_expired_opportunities(self, kraken_service):
        """Remove opportunities older than max_trade_lifetime_hours and record PnL"""
        if not self.opportunities:
            return
        
        current_time = datetime.now()
        max_age = timedelta(hours=self.max_trade_lifetime_hours)
        opportunities_to_remove = []
        
        for opp_id, opportunity in self.opportunities.items():
            try:
                # Check created_at for opportunity age
                created_at = datetime.fromisoformat(opportunity.get('created_at', ''))
                age = current_time - created_at
                
                if age > max_age:
                    # Record PnL for this expired opportunity
                    self._record_opportunity_pnl(opportunity, kraken_service)
                    opportunities_to_remove.append(opp_id)
                    logger.info(f"Marking expired opportunity {opp_id} ({opportunity.get('asset', 'unknown')}) for removal - age: {age.total_seconds()/3600:.1f}h")
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse created_at for opportunity {opp_id}: {e}")
        
        # Remove expired opportunities
        for opp_id in opportunities_to_remove:
            del self.opportunities[opp_id]
        
        if opportunities_to_remove:
            self.save_opportunities()
            logger.info(f"Removed {len(opportunities_to_remove)} expired opportunities older than {self.max_trade_lifetime_hours}h")
    
    def _record_opportunity_pnl(self, opportunity: Dict, kraken_service):
        """Record PnL for an expired opportunity - simple buy and hold"""
        try:
            asset = opportunity.get('asset')
            entry_price = opportunity.get('entry_price', 0)
            source = opportunity.get('source', 'Unknown')
            created_at = opportunity.get('created_at', '')

            # Get current price for exit (what we would sell at now)
            exit_price = self._get_current_price(asset, kraken_service)
            if not exit_price:
                logger.warning(f"Could not get current price for {asset}, skipping PnL calculation")
                return

            # Calculate simple buy-and-hold PnL
            if entry_price > 0:
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                pnl_usd = exit_price - entry_price
            else:
                pnl_pct = 0
                pnl_usd = 0

            # Calculate actual duration
            try:
                created_dt = datetime.fromisoformat(created_at)
                duration_actual = (datetime.now() - created_dt).total_seconds() / 3600
            except:
                duration_actual = self.max_trade_lifetime_hours

            # Record to state_performance.json with simplified data
            outcome = {
                'timestamp': datetime.now().isoformat(),
                'position_id': opportunity.get('id'),
                'asset': asset,
                'source': source,
                'action': 'long',
                'entry_price': entry_price,
                'close_price': exit_price,
                'close_reason': 'time_based_exit',
                'pnl_percent': pnl_pct,
                'confidence': opportunity.get('confidence', 0.5),
                'duration_planned_hours': self.max_trade_lifetime_hours,
                'duration_actual_hours': duration_actual
            }

            with open('state_performance.json', 'a') as f:
                f.write(json.dumps(outcome) + '\n')

            logger.info(f"Recorded simple buy-and-hold PnL for {asset}: {pnl_pct:+.2f}% (${pnl_usd:+.2f}) - held {duration_actual:.1f}h")

        except Exception as e:
            logger.error(f"Error recording PnL for opportunity {opportunity.get('id', 'unknown')}: {e}")
    
    def _get_current_price(self, asset: str, kraken_service) -> Optional[float]:
        """Get current price for an asset"""
        try:
            pair_id = kraken_service.get_pair_for_asset(asset)
            ticker = kraken_service.get_ticker(pair_id)
            
            if ticker and 'c' in ticker:
                return float(ticker['c'][0])  # Current price
            return None
        except Exception as e:
            logger.error(f"Error getting current price for {asset}: {e}")
            return None
    
    def get_all_crypto_keywords(self, pairs: Dict[str, Dict]) -> Dict[str, List[str]]:
        """Generate keywords for all cryptocurrencies available on Kraken"""
        
        # Common name mappings for major cryptos
        common_names = {
            'XXBT': ['bitcoin', 'btc'],
            'XETH': ['ethereum', 'eth'],
            'SOL': ['solana', 'sol'],
            'XXRP': ['ripple', 'xrp'],
            'ADA': ['cardano', 'ada'],
            'DOT': ['polkadot', 'dot'],
            'LINK': ['chainlink', 'link'],
            'XLTC': ['litecoin', 'ltc'],
            'BCH': ['bitcoin cash', 'bch'],
            'XXMR': ['monero', 'xmr'],
            'AVAX': ['avalanche', 'avax'],
            'ALGO': ['algorand', 'algo'],
            'ATOM': ['cosmos', 'atom'],
            'UNI': ['uniswap', 'uni'],
            'AAVE': ['aave'],
            'COMP': ['compound', 'comp'],
            'MKR': ['maker', 'mkr'],
            'YFI': ['yearn', 'yfi'],
            'SNX': ['synthetix', 'snx'],
            'CRV': ['curve', 'crv'],
            'SUSHI': ['sushiswap', 'sushi'],
            'BAL': ['balancer', 'bal'],
            'ZRX': ['0x', 'zrx'],
            'GRT': ['the graph', 'grt'],
            'FIL': ['filecoin', 'fil'],
            'ICP': ['internet computer', 'icp'],
            'NEAR': ['near protocol', 'near'],
            'FLOW': ['flow'],
            'KSM': ['kusama', 'ksm'],
            'MANA': ['decentraland', 'mana'],
            'SAND': ['sandbox', 'sand'],
            'ENJ': ['enjin', 'enj'],
            'CHZ': ['chiliz', 'chz'],
            'BAT': ['basic attention token', 'bat'],
            'ZEC': ['zcash', 'zec'],
            'DASH': ['dash'],
            'ETC': ['ethereum classic', 'etc'],
            'LTC': ['litecoin', 'ltc'],
            'SHIB': ['shiba inu', 'shib'],
            'PEPE': ['pepe'],
            'BONK': ['bonk'],
            'WIF': ['dogwifhat', 'wif'],
            'FLOKI': ['floki'],
            'DOGE': ['dogecoin', 'doge'],
            # 'USDT': ['tether', 'usdt'],  # Excluded - USDT is a stablecoin, not tradeable
            'USDC': ['usd coin', 'usdc'],
            'DAI': ['dai'],
            'PYUSD': ['paypal usd', 'pyusd'],
            'RLUSD': ['ripple usd', 'rlusd']
        }
        
        crypto_keywords = {}
        
        # Get all base currencies that trade against USD or USDT
        for pair_name, pair_info in pairs.items():
            base = pair_info['base']
            quote = pair_info['quote']
            
            # Skip USDT (stablecoin) - never trade USDT/USD
            if base == 'USDT':
                continue
            
            # Only include USD or USDT denominated pairs
            if quote in ['ZUSD', 'USDT']:
                if base not in crypto_keywords:
                    crypto_keywords[base] = []
                
                # Add the Kraken currency code itself
                crypto_keywords[base].append(base.lower())
                
                # Remove common prefixes for easier matching
                clean_base = base
                if base.startswith('X') and len(base) > 1:
                    clean_base = base[1:]  # XETH -> ETH, XXBT -> XBT
                    crypto_keywords[base].append(clean_base.lower())
                    
                if base.startswith('XX') and len(base) > 2:
                    clean_base = base[2:]  # XXBT -> BT, but we want BTC
                    if base == 'XXBT':
                        clean_base = 'BTC'
                    elif base == 'XXRP':
                        clean_base = 'XRP'
                    elif base == 'XXMR':
                        clean_base = 'XMR'
                    elif base == 'XXDG':
                        clean_base = 'DOGE'
                    crypto_keywords[base].append(clean_base.lower())
                
                # Add common names if available
                if base in common_names:
                    crypto_keywords[base].extend(common_names[base])
                
                # Remove duplicates
                crypto_keywords[base] = list(set(crypto_keywords[base]))
        
        return crypto_keywords
    
    def update_bandit_model_from_outcomes(self):
        """Update bandit model based on newly recorded PnL outcomes"""
        try:
            # Read the state_performance.json file
            if not os.path.exists('state_performance.json'):
                logger.info("No performance data file found, skipping bandit model update")
                return

            with open('state_performance.json', 'r') as f:
                lines = f.readlines()

            # Track which outcomes we've already processed
            processed_outcomes = set()
            if os.path.exists('state_bandit_model.json'):
                try:
                    with open('state_bandit_model.json', 'r') as f:
                        existing_data = json.load(f)
                        # Track processed outcomes by position_id + timestamp
                        processed_outcomes = set()
                        for source_data in existing_data.get('processed_outcomes', []):
                            processed_outcomes.add(source_data)
                except:
                    pass

            # Process only new outcomes
            new_outcomes = []
            all_outcomes = []

            for line in lines:
                if line.strip():
                    try:
                        outcome = json.loads(line.strip())
                        outcome_key = f"{outcome.get('position_id', '')}_{outcome.get('timestamp', '')}"
                        all_outcomes.append((outcome, outcome_key))

                        if outcome_key not in processed_outcomes:
                            new_outcomes.append(outcome)
                    except json.JSONDecodeError:
                        continue

            if not new_outcomes:
                logger.info("No new outcomes to process for bandit model update")
                return
            
            # Update bandit stats for each source
            for outcome in new_outcomes:
                source = outcome.get('source', 'Unknown')
                
                # Handle multiple formats: old format (pnl_pct/pnl_usd) and new format (pnl_percent)
                pnl_pct = outcome.get('pnl_percent', outcome.get('pnl_pct', outcome.get('pnl_percent', 0)))
                pnl_usd = outcome.get('pnl_usd', 0)

                # Derive outcome type from pnl percentage
                outcome_type = 'success' if pnl_pct > 0 else 'failure'

                # If we don't have pnl_usd, calculate it from pnl_pct and entry_price
                if pnl_usd == 0 and pnl_pct != 0:
                    entry_price = outcome.get('entry_price', 1)
                    pnl_usd = entry_price * (pnl_pct / 100)
                
                if source not in self.bandit_source_stats:
                    self.bandit_source_stats[source] = {
                        'total_trades': 0,
                        'successful_trades': 0,
                        'total_profit': 0.0,
                        'avg_profit': 0.0,
                        'success_rate': 0.0
                    }
                
                stats = self.bandit_source_stats[source]
                
                # Initialize missing fields if they don't exist
                if 'total_trades' not in stats:
                    stats['total_trades'] = 0
                if 'total_profit' not in stats:
                    stats['total_profit'] = 0.0
                if 'successful_trades' not in stats:
                    stats['successful_trades'] = 0
                
                stats['total_trades'] += 1
                stats['total_profit'] += pnl_usd
                
                logger.info(f"Processing outcome for {source}: pnl_pct={pnl_pct}, pnl_usd={pnl_usd}, outcome_type={outcome_type}")
                
                if outcome_type == 'success':
                    stats['successful_trades'] += 1
                
                stats['avg_profit'] = stats['total_profit'] / stats['total_trades']
                stats['success_rate'] = stats['successful_trades'] / stats['total_trades']
                
                logger.info(f"Updated bandit stats for {source}: {stats['success_rate']:.2%} success rate, ${stats['avg_profit']:.2f} avg profit")
            
            # Save updated stats with processed outcomes tracking
            bandit_data = {
                'source_stats': self.bandit_source_stats,
                'processed_outcomes': [outcome_key for _, outcome_key in all_outcomes],
                'last_updated': datetime.now().isoformat()
            }

            with open('state_bandit_model.json', 'w') as f:
                json.dump(bandit_data, f, indent=2)

            logger.info(f"Updated bandit model with {len(new_outcomes)} new outcomes (total processed: {len(all_outcomes)})")

        except Exception as e:
            logger.error(f"Error updating bandit model from outcomes: {e}")
    
    def get_source_trust_score(self, source: str) -> float:
        """Get trust score for a source based on bandit model"""
        if source not in self.bandit_source_stats:
            return 0.5  # Default neutral score
        
        stats = self.bandit_source_stats[source]
        
        # Combine success rate and average profit for trust score
        success_rate = stats['success_rate']
        avg_profit = stats['avg_profit']
        
        # Normalize profit (assuming -10% to +10% range)
        normalized_profit = max(0, min(1, (avg_profit + 0.1) / 0.2))
        
        # Weighted combination: 70% success rate, 30% profit
        trust_score = 0.7 * success_rate + 0.3 * normalized_profit
        
        return min(1.0, max(0.0, trust_score))
    
    def calculate_technical_indicators(self, candles: List[Dict]) -> Dict:
        """Calculate technical indicators from candle data"""
        if len(candles) < 20:
            return {}
        
        try:
            # Extract price and volume data
            closes = np.array([candle['close'] for candle in candles])
            highs = np.array([candle['high'] for candle in candles])
            lows = np.array([candle['low'] for candle in candles])
            volumes = np.array([candle['volume'] for candle in candles])
            
            # Calculate moving averages
            sma_10 = np.mean(closes[-10:])
            sma_20 = np.mean(closes[-20:])
            
            # Calculate EMAs
            ema_12 = self._calculate_ema(closes, 12)
            ema_26 = self._calculate_ema(closes, 26)
            
            # Calculate MACD
            macd = ema_12 - ema_26
            macd_signal = self._calculate_ema(np.array([macd]), 9) if len(closes) >= 26 else 0
            macd_histogram = macd - macd_signal
            
            # Calculate Bollinger Bands
            bb_middle = sma_20
            bb_std = np.std(closes[-20:])
            bb_upper = bb_middle + (2 * bb_std)
            bb_lower = bb_middle - (2 * bb_std)
            bb_position = (closes[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            
            # Calculate RSI
            rsi = self._calculate_rsi(closes)
            
            # Calculate price changes
            price_change_1h = ((closes[-1] - closes[-4]) / closes[-4] * 100) if len(closes) >= 4 else 0
            price_change_24h = ((closes[-1] - closes[-96]) / closes[-96] * 100) if len(closes) >= 96 else 0
            
            # Calculate volume metrics
            current_volume = volumes[-1]
            avg_volume_10 = np.mean(volumes[-10:])
            
            return {
                'current_price': closes[-1],
                'sma_10': sma_10,
                'sma_20': sma_20,
                'ema_12': ema_12,
                'ema_26': ema_26,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'bb_position': bb_position,
                'rsi': rsi,
                'price_change_1h': price_change_1h,
                'price_change_24h': price_change_24h,
                'current_volume': current_volume,
                'avg_volume_10': avg_volume_10
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_rag_recommendation(self, kraken_service, openai_service) -> Optional[Dict]:
        """Get position recommendation from OpenAI using RAG query"""
        try:
            # Get current Kraken positions, excluding fiat/stable assets
            balances = kraken_service.get_account_balance()
            current_positions = {}
            stable_or_fiat_assets = {
                'ZUSD', 'USD', 'ZEUR', 'EUR', 'ZGBP', 'GBP', 'ZCAD', 'CAD', 'ZJPY', 'JPY', 'ZCHF', 'CHF',
                'USDT', 'USDC', 'DAI', 'PYUSD', 'RLUSD'
            }
            for asset, balance in balances.items():
                try:
                    bal = float(balance)
                except (TypeError, ValueError):
                    continue
                if bal <= 0:
                    continue
                # Include all assets, including fiat/stable for allocation checking
                current_positions[asset] = bal

            # If no opportunities but we have positions, ask OpenAI what to do
            if not self.opportunities:
                if current_positions:
                    logger.info(f"No opportunities but {len(current_positions)} positions held - asking OpenAI for recommendation")

                    # Build current positions data for OpenAI
                    portfolio_data = []
                    valid_positions = {}

                    for asset in current_positions.keys():
                        # Skip assets that don't have valid trading pairs
                        pair_id = kraken_service.get_pair_for_asset(asset)
                        if not pair_id:
                            logger.warning(f"Skipping asset {asset} - no valid trading pair found")
                            continue

                        # Check if asset is tradeable in current jurisdiction
                        if not kraken_service.is_asset_tradeable(asset):
                            logger.warning(f"Skipping asset {asset} - trading restricted in current jurisdiction")
                            continue

                        # Get current price
                        ticker = kraken_service.get_ticker(pair_id)
                        if not ticker or 'c' not in ticker:
                            logger.warning(f"Skipping asset {asset} - no ticker data available")
                            continue

                        current_price = float(ticker['c'][0])
                        valid_positions[asset] = current_price

                        # Try to get technical indicators (optional)
                        candles = kraken_service.get_candles(pair_id, interval=15, count=720)
                        technical_indicators = {}
                        if candles and len(candles) >= 20:
                            technical_indicators = self.calculate_technical_indicators(candles)

                        # Use basic data even if technical indicators fail
                        portfolio_data.append({
                            'asset': asset,
                            'current_price': current_price,
                            'entry_price': current_price,  # We don't have actual entry prices for existing positions
                            'source': 'Existing Position',
                            'trust_score': 0.5,  # Neutral for existing positions without historical data
                            'sentiment': 'hold',  # Default for existing positions
                            'confidence': 0.5,
                            'technical_indicators': technical_indicators
                        })

                    if portfolio_data:
                        # Create OpenAI prompt for position evaluation
                        current_positions_summary = []
                        for pos in portfolio_data:
                            current_positions_summary.append(f"{pos['asset']}: ${pos['current_price']:.2f}")

                        portfolio_details = []
                        for item in portfolio_data:
                            tech = item['technical_indicators']
                            trust_score = item.get('trust_score', 0.5)
                            detail = f"""
ASSET: {item['asset']}
CURRENT PRICE: ${item['current_price']:.4f}
TRUST SCORE: {trust_score:.2f} (historical performance from inception to {self.max_trade_lifetime_hours}h based on buy-and-hold returns)
TECHNICAL INDICATORS:
- RSI: {tech.get('rsi', 0):.1f}
- MACD: {tech.get('macd', 0):.6f}
- Bollinger Position: {tech.get('bb_position', 0):.2f}
- 1h Change: {tech.get('price_change_1h', 0):.2f}%
- 24h Change: {tech.get('price_change_24h', 0):.2f}%
"""
                            portfolio_details.append(detail)

                        prompt = f"""
You are an expert cryptocurrency portfolio manager. You have {len(current_positions)} existing positions but no new opportunities to consider.

CURRENT POSITIONS:
{', '.join(current_positions_summary)}

DETAILED ANALYSIS:
{chr(10).join(portfolio_details)}

TASK:
Evaluate whether to hold these existing positions or convert them to cash (sell everything to USD).

IMPORTANT HOLDING PREFERENCE:
- Prefer to HOLD current positions unless there's a compelling reason to switch
- Only switch positions if there's a recent high-quality opportunity with excellent trust score (>0.8), strong conviction, and clearly advantageous technical indicators
- Do not sell positions simply because "no new opportunities" exist - this is normal market behavior
- Consider that positions may continue to perform well even without new signals
- Only recommend switching if the current position shows clear negative signals (bearish technicals, high risk) AND there's a superior alternative

Consider:
- Current market conditions and technical indicators
- Risk of holding without new opportunities (generally LOW - this is normal)
- Transaction costs (0.5% per trade to sell - avoid unnecessary trading)
- Market volatility and momentum
- Whether current positions show signs of continued upside potential
- Trust scores and historical performance of current positions

AGGRESSIVE SWITCHING POLICY: When opportunities exist, prefer SWITCHING to recommended assets unless current positions are clearly superior. Transaction costs are acceptable when switching to recommended assets with positive sentiment and technicals.

RESPOND IN THIS EXACT JSON FORMAT:
{{
    "action": "hold|switch",
    "reasoning": "<detailed explanation of your recommendation>",
    "recommended_assets": ["ETH"] or [],
    "expected_pnl_pct": <expected profit percentage for holding>,
    "risk_score": <risk score 1-10>
}}

If action is "switch", recommended_assets should be ["ETH"] to switch to ETH or [] to go to cash.
If action is "hold", keep current positions.
"""

                        try:
                            response = openai_service.client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": "You are an expert cryptocurrency portfolio manager focused on maximizing risk-adjusted returns."},
                                    {"role": "user", "content": prompt}
                                ]
                            )

                            response_text = response.choices[0].message.content.strip()
                            logger.info(f"OpenAI Position Evaluation: {response_text}")

                            # Parse JSON response
                            try:
                                json_text = response_text
                                if response_text.startswith('```json'):
                                    json_text = response_text.replace('```json', '').replace('```', '').strip()
                                elif response_text.startswith('```'):
                                    json_text = response_text.replace('```', '').strip()

                                recommendation = json.loads(json_text)

                                # Validate required fields
                                if 'action' in recommendation and 'reasoning' in recommendation:
                                    # If OpenAI recommends switching to ETH, change to empty array (cash)
                                    if recommendation.get('action') == 'switch' and recommendation.get('recommended_assets') == ['ETH']:
                                        recommendation['recommended_assets'] = []  # Convert to cash
                                        recommendation['reasoning'] += " (Converting to cash instead of switching to ETH)"

                                    return recommendation
                                else:
                                    logger.warning("OpenAI response missing required fields")
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse OpenAI response as JSON: {e}")

                        except Exception as e:
                            logger.error(f"Error getting OpenAI recommendation: {e}")

                    # Fallback if OpenAI fails
                    logger.warning("OpenAI recommendation failed, defaulting to hold positions")
                    return {
                        'action': 'hold',
                        'reasoning': 'Unable to get OpenAI recommendation, holding existing positions.',
                        'recommended_assets': list(valid_positions.keys()) if valid_positions else [],
                        'expected_pnl_pct': 0.0,
                        'risk_score': 5
                    }
                else:
                    logger.info("No opportunities and no positions - nothing to do")
                    return None

            # Get distinct assets from opportunities
            distinct_assets = set()
            for opportunity in self.opportunities.values():
                asset = opportunity.get('asset')
                if asset:
                    distinct_assets.add(asset)

            logger.info(f"Getting RAG recommendation for {len(distinct_assets)} distinct assets: {list(distinct_assets)}")
            
            # Build portfolio data for RAG query
            portfolio_data = []
            for asset in distinct_assets:
                # Check if asset is tradeable in current jurisdiction
                if not kraken_service.is_asset_tradeable(asset):
                    logger.warning(f"Skipping opportunity for {asset} - trading restricted in current jurisdiction")
                    continue

                # Get current price and technical indicators
                pair_id = kraken_service.get_pair_for_asset(asset)
                if not pair_id:
                    logger.warning(f"Skipping opportunity for {asset} - no valid trading pair found")
                    continue

                candles = kraken_service.get_candles(pair_id, interval=15, count=720)
                if not candles or len(candles) < 20:
                    logger.warning(f"Skipping opportunity for {asset} - insufficient candle data")
                    continue
                
                technical_indicators = self.calculate_technical_indicators(candles)
                if not technical_indicators:
                    continue
                
                current_price = technical_indicators.get('current_price', 0)

                # Find matching opportunity
                matching_opportunity = None
                for opportunity in self.opportunities.values():
                    if opportunity.get('asset') == asset:
                        matching_opportunity = opportunity
                        break

                if matching_opportunity:
                    # Get trust score for this source
                    source = matching_opportunity.get('source', 'Unknown')
                    trust_score = self.get_source_trust_score(source)

                    portfolio_data.append({
                                'asset': asset,
                                'current_price': current_price,
                                'entry_price': matching_opportunity.get('entry_price', 0),
                                'exit_price': matching_opportunity.get('exit_price', 0),
                                'stop_loss': matching_opportunity.get('stop_loss', 0),
                                'source': source,
                                'trust_score': trust_score,
                                'sentiment': matching_opportunity.get('sentiment', 'neutral'),
                                'confidence': matching_opportunity.get('confidence', 0.5),
                                'technical_indicators': technical_indicators
                            })
            
            if not portfolio_data:
                logger.warning("No portfolio data available for RAG query")
                return None
            
            # Create RAG prompt
            portfolio_summary = []
            for item in portfolio_data:
                tech = item['technical_indicators']
                trust_score = item.get('trust_score', 0.5)
                summary = f"""
ASSET: {item['asset']}
CURRENT PRICE: ${item['current_price']:.4f}
ORIGINAL ENTRY: ${item['entry_price']:.4f}
TARGET EXIT: ${item['exit_price']:.4f}
STOP LOSS: ${item['stop_loss']:.4f}
SOURCE: {item['source']}
TRUST SCORE: {trust_score:.2f} (historical performance from inception to {self.max_trade_lifetime_hours}h based on buy-and-hold returns)
SENTIMENT: {item['sentiment']}
CONFIDENCE: {item['confidence']:.2f}
TECHNICAL INDICATORS:
- RSI: {tech.get('rsi', 0):.1f}
- MACD: {tech.get('macd', 0):.6f}
- MACD Signal: {tech.get('macd_signal', 0):.6f}
- Bollinger Position: {tech.get('bb_position', 0):.2f}
- 1h Change: {tech.get('price_change_1h', 0):.2f}%
- 24h Change: {tech.get('price_change_24h', 0):.2f}%
- Volume Ratio: {(tech.get('current_volume', 0) / max(tech.get('avg_volume_10', 1), 1)):.1f}x
"""
                portfolio_summary.append(summary)
            
            current_positions_summary = []
            for asset, balance in current_positions.items():
                current_positions_summary.append(f"{asset}: {balance} units")
            
            # Find the best opportunity based on trust score and confidence
            # Include ALL opportunities - we want to switch TO the best opportunity,
            # even if we already hold some of that asset
            best_opportunity = None
            best_score = 0


            for item in portfolio_data:
                asset = item['asset']
                trust_score = item.get('trust_score', 0.5)
                confidence = item.get('confidence', 0.5)
                combined_score = (trust_score + confidence) / 2

                # Always consider opportunities - we want to switch TO the best one
                if combined_score > best_score:
                    best_score = combined_score
                    best_opportunity = item

            # If we have opportunities and current positions, evaluate switching
            if best_opportunity and current_positions:
                best_asset = best_opportunity['asset']

                # Check if we already hold the recommended asset
                holding_recommended = False
                holding_only_recommended = True

                for pos_asset in current_positions.keys():
                    # Normalize for comparison
                    normalized_pos = pos_asset.replace('X', '').replace('.F', '')
                    normalized_rec = best_asset.replace('X', '').replace('.F', '')

                    if normalized_pos == normalized_rec:
                        holding_recommended = True
                    elif pos_asset not in stable_or_fiat_assets:  # Ignore fiat/stable assets
                        holding_only_recommended = False

                # If we hold the recommended asset, check for available cash for additional allocation
                if holding_recommended:
                    # Check balances directly for available cash (including fiat assets)
                    total_cash = 0
                    for asset, balance in balances.items():
                        if asset in ['ZUSD', 'USDT', 'USDC']:  # Fiat assets
                            try:
                                if isinstance(balance, dict):
                                    bal = float(balance.get('balance', 0) or 0)
                                    hold = float(balance.get('hold_trade', 0) or 0)
                                    available = bal - hold
                                else:
                                    available = float(balance)

                                if available > 0.01:
                                    total_cash += available
                            except (TypeError, ValueError):
                                continue

                    # If we have any meaningful cash available, proceed with allocation
                    logger.info(f"Total cash detected: ${total_cash:.2f}")
                    if total_cash > 0.1:  # Lower threshold - any meaningful cash should trigger allocation
                        logger.info(f"Detected ${total_cash:.2f} cash available - can allocate more to {best_asset}")
                        # Don't return hold - let the normal switching logic proceed
                    else:
                        # We have the recommended asset and truly minimal cash - optimally positioned
                        logger.info(f"Truly minimal cash (${total_cash:.2f}) - optimally positioned")
                        return {
                            'action': 'hold',
                            'reasoning': f'Already optimally positioned with {best_asset} and minimal cash available.',
                            'recommended_assets': [best_asset],
                            'expected_pnl_pct': 0.0,
                            'risk_score': 3
                        }

            # If no opportunities found, recommend holding
            if not best_opportunity:
                return {
                    'action': 'hold',
                    'reasoning': 'No opportunities available for evaluation.',
                    'recommended_assets': list(current_positions.keys()) if current_positions else [],
                    'expected_pnl_pct': 0.0,
                    'risk_score': 3
                }

            best_asset = best_opportunity['asset']

            prompt = f"""
You are an expert cryptocurrency portfolio manager. Analyze this portfolio of opportunities and current positions to recommend the optimal trading action.

CURRENT PORTFOLIO POSITIONS:
{', '.join(current_positions_summary) if current_positions_summary else 'No current positions'}

AVAILABLE OPPORTUNITIES:
{chr(10).join(portfolio_summary)}

TASK:
Given the current portfolio positions and available opportunities, recommend whether to:
1. HOLD current positions (if they are performing well)
2. SWITCH to the best available opportunity if it offers better risk-adjusted returns

Consider:
- Current position performance vs opportunity potential
- Transaction costs (0.5% per trade)
- Risk management and stop-loss levels
- Technical indicators alignment
- Source credibility and confidence scores
- Trust scores and historical performance

Switching preference:
- Actively SWITCH to recommended opportunities unless current positions are clearly superior. Transaction costs are acceptable when switching to assets with positive sentiment and reasonable technical indicators.

BEST OPPORTUNITY IDENTIFIED: {best_asset} (highest combined trust + confidence score)

RESPOND IN THIS EXACT JSON FORMAT:
{{
    "action": "hold|switch",
    "reasoning": "<brief explanation of your recommendation>",
    "recommended_assets": ["{best_asset}"],
    "expected_pnl_pct": <expected profit percentage>,
    "risk_score": <risk score 1-10>
}}

If action is "switch", recommended_assets should be ["{best_asset}"].
If action is "hold", keep current positions and recommended_assets should be empty array.
"""

            # Get recommendation from OpenAI
            if not openai_service.client:
                logger.warning("OpenAI not available for RAG recommendation")
                return None
            
            response = openai_service.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert cryptocurrency portfolio manager focused on maximizing risk-adjusted returns."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.info(f"RAG Recommendation: {response_text}")
            
            # Parse JSON response
            try:
                json_text = response_text
                if response_text.startswith('```json'):
                    json_text = response_text.replace('```json', '').replace('```', '').strip()
                elif response_text.startswith('```'):
                    json_text = response_text.replace('```', '').strip()
                
                recommendation = json.loads(json_text)
                
                # Validate required fields
                if 'action' in recommendation and 'reasoning' in recommendation:
                    return recommendation
                else:
                    logger.warning("RAG response missing required fields")
                    return None
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse RAG response as JSON: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting RAG recommendation: {e}")
            return None

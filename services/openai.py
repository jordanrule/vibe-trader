"""
OpenAI service for sentiment analysis and trade selection
"""
import json
import logging
import re
from typing import Dict, List, Optional
from .base import BaseService

logger = logging.getLogger(__name__)

# Known signal sources detected from Telegram message analysis
KNOWN_SIGNAL_SOURCES = {
    'RAVEN Signals Pro - Crypto Forex': 'Raven Signals Pro',
    'Binance Killers': 'Binance Killers',
    'BitDegree Signals': 'BitDegree Signals',
    'Rocket Wallet Signals': 'Rocket Wallet Signals',
    'ravensignalspro.com': 'Raven Signals Pro',
    'altsignals.io': 'AltSignals.io',
    'CryptoSignals.Org (free)': 'CryptoSignals.Org (Free)',
    'CryptoSignals.Org': 'CryptoSignals.Org (Free)',
    'Coinbase Crypto Signals': 'Coinbase Crypto Signals',
    'Evening Trader Official ®': 'Evening Trader Official',
    'CryptoNinjas Trading': 'CryptoNinjas Trading',
    'CoinCodeCap Classic': 'CoinCodeCap Classic',
    'Fat Pig Signals': 'Fat Pig Signals',
    'My Crypto Paradise OFFICIAL': 'My Crypto Paradise Official',
    'Crypto Inner Circle (Free)': 'Crypto Inner Circle (Free)'
}

class OpenAIService(BaseService):
    """Service for handling OpenAI API interactions"""
    
    def __init__(self, config: Dict[str, str]):
        super().__init__(config)
        self.api_key = config.get('openai_api_key')
        self.client = None

        if self.api_key:
            try:
                import openai

                # Try the newer client syntax first (OpenAI v1.x)
                try:
                    self.client = openai.OpenAI(api_key=self.api_key)
                    logger.info("OpenAI client initialized (v1.x syntax)")
                except TypeError as e:
                    # Fall back to older syntax for OpenAI v0.x
                    if "unexpected keyword argument" in str(e):
                        openai.api_key = self.api_key
                        self.client = openai
                        logger.info("OpenAI client initialized (legacy syntax)")
                    else:
                        raise e

                logger.info("OpenAI API key configured")
            except ImportError:
                logger.warning("OpenAI not installed. Run: pip install openai==1.12.0")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
        else:
            logger.warning("OpenAI API key not found. Sentiment analysis will be disabled.")

    def analyze_sentiment(self, message: str, asset: str, pair_name: str, technical_indicators: Dict) -> Optional[Dict]:
        """Send message and technical data to OpenAI for sentiment analysis"""
        
        # Never analyze USDT/USD opportunities (USDT is a stablecoin)
        if asset == 'USDT' and pair_name == 'USDT/USD':
            logger.info(f"🚫 Skipping USDT/USD analysis - USDT is a stablecoin, not a tradeable opportunity")
            return None
        
        if not self.client:
            logger.warning("OpenAI not available for sentiment analysis")
            return None
        
        try:
            # Create comprehensive prompt for OpenAI
            # Format KNOWN_SIGNAL_SOURCES for the prompt
            known_sources_formatted = "\n".join([f"  - {k}: {v}" for k, v in KNOWN_SIGNAL_SOURCES.items()])

            prompt = f"""
You are an expert cryptocurrency trader focused on risk-adjusted profit maximization. Analyze the following trading message and technical indicators to provide trading recommendations.

MESSAGE TO ANALYZE:
"{message}"

ASSET: {asset} ({pair_name})

KNOWN SIGNAL SOURCES (identify which one this message comes from, if any):
{known_sources_formatted}

TECHNICAL INDICATORS:
- Current Price: ${technical_indicators.get('current_price', 0):.4f}
- SMA 10: ${technical_indicators.get('sma_10', 0):.4f}
- SMA 20: ${technical_indicators.get('sma_20', 0):.4f}
- EMA 12: ${technical_indicators.get('ema_12', 0):.4f}
- EMA 26: ${technical_indicators.get('ema_26', 0):.4f}
- MACD: {technical_indicators.get('macd', 0):.6f}
- MACD Signal: {technical_indicators.get('macd_signal', 0):.6f}
- MACD Histogram: {technical_indicators.get('macd_histogram', 0):.6f}
- Bollinger Upper: ${technical_indicators.get('bb_upper', 0):.4f}
- Bollinger Middle: ${technical_indicators.get('bb_middle', 0):.4f}
- Bollinger Lower: ${technical_indicators.get('bb_lower', 0):.4f}
- Bollinger Position: {technical_indicators.get('bb_position', 0):.2f} (0=lower band, 1=upper band)
- RSI: {technical_indicators.get('rsi', 50):.1f}
- 1h Price Change: {technical_indicators.get('price_change_1h', 0):.2f}%
- 24h Price Change: {technical_indicators.get('price_change_24h', 0):.2f}%
- Current Volume vs 10-period avg: {(technical_indicators.get('current_volume', 0) / max(technical_indicators.get('avg_volume_10', 1), 1)):.1f}x

TASK:
1. Identify the signal source from the KNOWN SIGNAL SOURCES list above, or classify as one of: Binance Futures, BingX Futures, Bitget Futures, ByBit USDT, KuCoin Futures, OKX Futures, Trading Signal Service, Technical Analysis, News/Fundamental Analysis, Unknown Source
2. Analyze the sentiment of the message combined with the technical indicators
3. Provide trading recommendations

RESPOND IN THIS EXACT JSON FORMAT:
{{
    "channel": "telegram",
    "source": "<identified signal source from the KNOWN SIGNAL SOURCES list or fallback category>",
    "sentiment": "bullish|bearish|neutral",
    "action": "long|short|hold",
    "entry_price": <recommended entry price as number>,
    "exit_price": <recommended exit/take-profit price as number>,
    "duration_hours": <recommended hold duration in hours as integer>,
    "confidence": <confidence score 0.0-1.0>,
    "reasoning": "<brief explanation of your analysis including source identification>"
}}

SOURCE IDENTIFICATION RULES:
- First, check if the message matches any of the KNOWN SIGNAL SOURCES listed above
- Look for exact name matches, URLs, or clear references to these sources
- If no match found, classify based on content (exchange signals, general trading signals, technical analysis, news, etc.)
- Be specific when possible - prefer exact matches from KNOWN SIGNAL SOURCES over generic categories

Consider:
1. Message sentiment (bullish/bearish tone, excitement level, credibility)
2. Signal source credibility (established exchanges vs unknown sources)
3. Technical indicators alignment (trend direction, momentum, volume)
4. Risk management (appropriate entry/exit levels)
5. Market context (overbought/oversold conditions)

Be conservative with confidence scores. Only recommend trades with confidence > 0.6.
Adjust confidence based on source credibility - established exchanges should get higher confidence.
"""

            logger.info(f"Sending sentiment analysis request to OpenAI for {asset}")
            logger.debug(f"OpenAI Query: {prompt[:500]}...")  # Log first 500 chars

            # Handle both new and legacy OpenAI client syntax
            if hasattr(self.client, 'chat'):  # New syntax (v1.x)
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert cryptocurrency trader focused on risk-adjusted profit maximization."},
                        {"role": "user", "content": prompt}
                    ]
                )
            else:  # Legacy syntax
                response = self.client.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert cryptocurrency trader focused on risk-adjusted profit maximization."},
                        {"role": "user", "content": prompt}
                    ]
                )
            
            response_text = response.choices[0].message.content.strip()
            logger.info(f"OpenAI Response: {response_text}")
            
            # Try to parse JSON response (handle code blocks from gpt-4o-mini)
            try:
                # Remove code block markers if present
                json_text = response_text
                if response_text.startswith('```json'):
                    json_text = response_text.replace('```json', '').replace('```', '').strip()
                elif response_text.startswith('```'):
                    json_text = response_text.replace('```', '').strip()
                
                analysis = json.loads(json_text)

                # Validate required fields
                required_fields = ['channel', 'source', 'sentiment', 'action', 'entry_price', 'exit_price', 'duration_hours', 'confidence', 'reasoning']
                if all(field in analysis for field in required_fields):
                    # OpenAI has already identified the source from our KNOWN_SIGNAL_SOURCES list
                    identified_source = analysis['source']

                    # Add trust score calculation based on the identified source
                    trust_score = 0.5  # Default neutral
                    try:
                        from services.opportunity import OpportunityService
                        opportunity_service = OpportunityService({'max_trade_lifetime_hours': 6})
                        trust_score = opportunity_service.get_source_trust_score(identified_source)
                    except:
                        pass

                    analysis['trust_score'] = trust_score
                    logger.info(f"OpenAI identified source: {identified_source} (trust: {trust_score:.2f})")

                    return analysis
                else:
                    logger.error(f"OpenAI response missing required fields: {response_text}")
                    return None
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse OpenAI JSON response: {e}")
                logger.error(f"Raw response: {response_text}")
                return None
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return None
    
    def select_best_trade(self, portfolio_data: List[Dict]) -> Optional[Dict]:
        """Use OpenAI to select the best trade from the portfolio based on technical analysis"""
        if not self.client:
            logger.warning("OpenAI not available for trade selection")
            return None
        
        try:
            logger.info("🤖 Using OpenAI to analyze portfolio and select best trade...")
            
            if not portfolio_data:
                logger.info("No portfolio data available for OpenAI analysis")
                return None
            
            # Create portfolio summary for OpenAI
            portfolio_summary = []
            for item in portfolio_data:
                trade = item['trade']
                tech = item['technical_indicators']
                
                # Handle None values safely
                current_price = item['current_price'] or 0
                entry_price = trade.get('entry_price', 0) or 0
                exit_price = trade.get('exit_price', 0) or 0
                stop_loss = trade.get('stop_loss', 0) or 0
                trust_score = trade.get('trust_score', 0) or 0
                sentiment_score = trade.get('sentiment_score', 0) or 0
                rsi = tech.get('rsi', 0) or 0
                macd = tech.get('macd', 0) or 0
                macd_signal = tech.get('macd_signal', 0) or 0
                bb_position = tech.get('bb_position', 0) or 0
                price_change_1h = tech.get('price_change_1h', 0) or 0
                price_change_24h = tech.get('price_change_24h', 0) or 0
                current_volume = tech.get('current_volume', 0) or 0
                avg_volume_10 = tech.get('avg_volume_10', 1) or 1
                volume_ratio = (current_volume / max(avg_volume_10, 1)) if avg_volume_10 > 0 else 0
                
                summary = f"""
ASSET: {item['asset']}
TRADE ID: {trade.get('id', 'unknown')}
CURRENT PRICE: ${current_price:.4f}
ORIGINAL ENTRY: ${entry_price:.4f}
TARGET EXIT: ${exit_price:.4f}
STOP LOSS: ${stop_loss:.4f}
TRUST SCORE: {trust_score:.2f}
SENTIMENT SCORE: {sentiment_score:.2f}
TECHNICAL INDICATORS:
- RSI: {rsi:.1f}
- MACD: {macd:.6f}
- MACD Signal: {macd_signal:.6f}
- Bollinger Position: {bb_position:.2f}
- 1h Change: {price_change_1h:.2f}%
- 24h Change: {price_change_24h:.2f}%
- Volume Ratio: {volume_ratio:.1f}x
"""
                portfolio_summary.append(summary)
            
            # Create OpenAI prompt
            prompt = f"""
You are an expert cryptocurrency portfolio manager. Analyze this portfolio of potential trades and select the ONE that looks most likely to be profitable.

PORTFOLIO OF TRADES:
{chr(10).join(portfolio_summary)}

TASK:
Given this portfolio of potential trades, pick ONE that looks most likely to be profitable over the next 6 hours.  If the delta in risk-adjusted return does not justify a 1% switching fee, then hold the current position.
Suggest an entry price for that trade that is likely to be triggered within the next 15 minutes.

RESPOND IN THIS EXACT JSON FORMAT:
{{
    "id": "pos_HFT_1756836649_1756845108",
    "entry_price": 0.0850
}}

SELECTION CRITERIA:
1. Technical indicators alignment (RSI, MACD, Bollinger Bands)
2. Current price proximity to original entry
3. Volume and momentum indicators
4. Risk-reward ratio (entry vs exit vs stop-loss)
5. Overall market conditions

IMPORTANT:
- Only return the trade ID and suggested entry price
- The entry price should be realistic for the next 15 minutes
- Consider current price, support/resistance levels, and momentum
- Pick the trade with the highest probability of success
- The trade should be profitable over the next 6 hours
"""

            logger.info(f"Sending portfolio analysis request to OpenAI for {len(portfolio_data)} assets")

            # Handle both new and legacy OpenAI client syntax
            if hasattr(self.client, 'chat'):  # New syntax (v1.x)
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert cryptocurrency portfolio manager focused on selecting the most profitable trades."},
                        {"role": "user", "content": prompt}
                    ]
                )
            else:  # Legacy syntax
                response = self.client.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert cryptocurrency portfolio manager focused on selecting the most profitable trades."},
                        {"role": "user", "content": prompt}
                    ]
                )
            
            response_text = response.choices[0].message.content.strip()
            logger.info(f"OpenAI Portfolio Response: {response_text}")
            
            # Parse JSON response
            try:
                json_text = response_text
                if response_text.startswith('```json'):
                    json_text = response_text.replace('```json', '').replace('```', '').strip()
                elif response_text.startswith('```'):
                    json_text = response_text.replace('```', '').strip()
                
                analysis = json.loads(json_text)
                
                # Validate required fields
                if 'id' in analysis and 'entry_price' in analysis:
                    trade_id = analysis['id']
                    suggested_entry_price = float(analysis['entry_price'])
                    
                    # Find the selected trade
                    selected_trade = None
                    for item in portfolio_data:
                        if item['trade'].get('id') == trade_id:
                            selected_trade = item['trade'].copy()
                            selected_trade['entry_price'] = suggested_entry_price
                            logger.info(f"OpenAI selected {item['asset']} with suggested entry: ${suggested_entry_price:.4f}")
                            break
                    
                    return selected_trade
                else:
                    logger.warning("OpenAI response missing required fields")
                    return None
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse OpenAI response as JSON: {e}")
                return None
            except Exception as e:
                logger.error(f"Error processing OpenAI response: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error in OpenAI portfolio analysis: {e}")
            return None


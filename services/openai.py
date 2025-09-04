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
    'Binance Futures': 'Binance Futures',
    'BingX Futures': 'BingX Futures', 
    'Bitget Futures': 'Bitget Futures',
    'ByBit USDT': 'ByBit USDT',
    'ByBit Futures': 'ByBit Futures',
    'KuCoin Futures': 'KuCoin Futures',
    'OKX Futures': 'OKX Futures',
    'Gate.io Futures': 'Gate.io Futures',
    'Binance': 'Binance Spot',
    'Bitget': 'Bitget Spot',
    'ravensignalspro.com': 'Raven Signals Pro',
    'altsignals.io': 'AltSignals.io',
    'CryptoSignals.Org': 'CryptoSignals.Org (Free)',
    'Trading View': 'TradingView Analysis',
    'Technical Analysis': 'Technical Analysis',
    'News': 'News/Fundamental Analysis',
    'Unknown': 'Unknown Source'
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
                self.client = openai.OpenAI(api_key=self.api_key)
                logger.info("OpenAI API key configured")
            except ImportError:
                logger.warning("OpenAI not installed. Run: pip install openai==1.14.0")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        else:
            logger.warning("OpenAI API key not found. Sentiment analysis will be disabled.")
    
    def map_source_to_known(self, openai_source: str) -> str:
        """Map OpenAI detected source to known source categories"""
        openai_lower = openai_source.lower()
        
        # Direct mapping
        for known_key, known_value in KNOWN_SIGNAL_SOURCES.items():
            if known_key.lower() in openai_lower:
                return known_value
        
        # Fuzzy matching for common patterns
        if any(word in openai_lower for word in ['binance', 'futures']):
            if 'binance' in openai_lower:
                return 'Binance Futures'
            elif any(ex in openai_lower for ex in ['bingx', 'bing']):
                return 'BingX Futures'
            elif any(ex in openai_lower for ex in ['bitget', 'bit']):
                return 'Bitget Futures'
            elif any(ex in openai_lower for ex in ['bybit', 'by']):
                return 'ByBit USDT'
            elif any(ex in openai_lower for ex in ['kucoin', 'ku']):
                return 'KuCoin Futures'
            elif any(ex in openai_lower for ex in ['okx', 'ok']):
                return 'OKX Futures'
            else:
                return 'Multi-Exchange Futures Signal'
        elif any(word in openai_lower for word in ['signal', 'trading']):
            return 'Trading Signal Service'
        elif any(word in openai_lower for word in ['technical', 'analysis']):
            return 'Technical Analysis'
        elif any(word in openai_lower for word in ['news', 'fundamental']):
            return 'News/Fundamental Analysis'
        else:
            return 'Unknown Source'
    
    def detect_message_source(self, message: str) -> str:
        """Detect the source of a trading message"""
        message_lower = message.lower()
        
        # Check for known signal sources
        for source_key in KNOWN_SIGNAL_SOURCES.keys():
            if source_key.lower() in message_lower:
                return KNOWN_SIGNAL_SOURCES[source_key]
        
        # Check for exchange patterns
        if any(exchange in message_lower for exchange in ['binance', 'bybit', 'bitget', 'kucoin', 'okx']):
            return 'Multi-Exchange Signal'
        
        # Check for signal service patterns
        if any(pattern in message_lower for pattern in ['signal', 'alert', 'call']):
            return 'Trading Signal Service'
        
        # Check for technical analysis patterns
        if any(pattern in message_lower for pattern in ['technical', 'analysis', 'chart', 'indicator']):
            return 'Technical Analysis'
        
        # Check for news patterns
        if any(pattern in message_lower for pattern in ['news', 'announcement', 'update']):
            return 'News/Fundamental Analysis'
        
        return 'Unknown Source'
    
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
            # Detect the message source
            detected_source = self.detect_message_source(message)

            # Get trust score for this source (historical performance from inception to max_trade_lifetime_hours)
            trust_score = 0.5  # Default neutral
            try:
                from services.opportunity import OpportunityService
                # We need to get the opportunity service instance, but for now we'll use a simple lookup
                # This should be passed in from the caller
                opportunity_service = OpportunityService({})
                trust_score = opportunity_service.get_source_trust_score(detected_source)
            except:
                pass

            # Create comprehensive prompt for OpenAI
            prompt = f"""
You are an expert cryptocurrency trader focused on risk-adjusted profit maximization. Analyze the following trading message and technical indicators to provide trading recommendations.

MESSAGE TO ANALYZE:
"{message}"

ASSET: {asset} ({pair_name})

SOURCE TRUST SCORE: {trust_score:.2f} (historical performance from inception to {getattr(self, 'max_trade_lifetime_hours', 4)}h based on buy-and-hold returns)

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

DETECTED SOURCE: {detected_source}

TASK:
Analyze the sentiment of the message combined with the technical indicators. Also identify the signal source type from the message content.

RESPOND IN THIS EXACT JSON FORMAT:
{{
    "channel": "telegram", 
    "source": "<signal source - try to identify from message content or use one of: Binance Futures, BingX Futures, Bitget Futures, ByBit USDT, KuCoin Futures, OKX Futures, Trading Signal Service, Technical Analysis, News/Fundamental Analysis, Unknown Source>",
    "sentiment": "bullish|bearish|neutral",
    "action": "long|short|hold",
    "entry_price": <recommended entry price as number>,
    "exit_price": <recommended exit/take-profit price as number>,
    "duration_hours": <recommended hold duration in hours as integer>,
    "confidence": <confidence score 0.0-1.0>,
    "reasoning": "<brief explanation of your analysis>"
}}

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
            
            response = self.client.chat.completions.create(
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
                    # Validate and map the source to known sources
                    openai_source = analysis['source']
                    mapped_source = self.map_source_to_known(openai_source)
                    analysis['source'] = mapped_source
                    analysis['detected_source'] = detected_source  # Add our detected source for comparison
                    
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
You are an expert cryptocurrency portfolio manager. Analyze this portfolio of potential trades and select the ONE that looks most likely to be profitable in the next 15 minutes.

PORTFOLIO OF TRADES:
{chr(10).join(portfolio_summary)}

TASK:
Given this portfolio of potential trades, pick ONE that looks most likely to be profitable. 
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
"""

            logger.info(f"Sending portfolio analysis request to OpenAI for {len(portfolio_data)} assets")
            
            response = self.client.chat.completions.create(
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


#!/usr/bin/env python3
"""
Analyze Telegram messages to identify common signal sources
"""

import sys
import os
import re
from collections import Counter
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import TradingAgent

def extract_sources_from_messages():
    """Analyze recent Telegram messages to identify signal sources"""
    print("🔍 Analyzing Telegram messages for signal sources...\n")
    
    agent = TradingAgent()
    
    try:
        # Get messages from last 24 hours
        to_time = datetime.now()
        from_time = to_time - timedelta(hours=24)
        
        messages = agent.check_telegram_messages(from_time, to_time)
        print(f"📨 Found {len(messages)} messages in last 24 hours")
        
        # Patterns that typically indicate signal sources
        source_patterns = [
            r'([\w\s\.]+(?:Futures|Signals?|Trading|Crypto|Exchange|\.org|\.com|\.io)[\w\s\.\(\)]*)',
            r'#(\w+/\w+)',  # Trading pairs like #BTC/USDT
            r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+\([^)]+\))?)',  # Title Case Names with optional parentheses
            r'(Binance|KuCoin|Bybit|ByBit|Bitget|OKX|Gate\.io|Coinbase)[\s\w]*',
            r'([A-Z]{2,}[\w\s]*(?:Signals?|Trading|Crypto)[\w\s]*)',
        ]
        
        all_sources = []
        message_analysis = []
        
        for msg in messages:
            text = msg.get('text', '').strip()
            if not text:
                continue
                
            # Try to extract source information
            detected_sources = []
            
            for pattern in source_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] if match[0] else match[1] if len(match) > 1 else ''
                    if match and len(match.strip()) > 2:
                        detected_sources.append(match.strip())
            
            # Look for exchange names
            exchanges = ['Binance', 'KuCoin', 'Bybit', 'ByBit', 'Bitget', 'OKX', 'Gate.io', 'Coinbase']
            for exchange in exchanges:
                if exchange.lower() in text.lower():
                    # Extract the full context around the exchange name
                    words = text.split()
                    for i, word in enumerate(words):
                        if exchange.lower() in word.lower():
                            # Get surrounding context
                            start = max(0, i-2)
                            end = min(len(words), i+3)
                            context = ' '.join(words[start:end])
                            detected_sources.append(context)
                            break
            
            if detected_sources:
                all_sources.extend(detected_sources)
                message_analysis.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'sources': detected_sources,
                    'timestamp': msg['date']
                })
        
        # Count and analyze sources
        source_counter = Counter(all_sources)
        
        print(f"\n📊 Most common sources detected:")
        for source, count in source_counter.most_common(20):
            print(f"   {count:2d}x: {source}")
        
        print(f"\n🔍 Sample message analysis:")
        for analysis in message_analysis[:10]:
            if analysis['sources']:
                print(f"\n   📱 \"{analysis['text']}\"")
                print(f"      🎯 Detected sources: {analysis['sources']}")
        
        # Generate suggested source mapping
        print(f"\n💡 Suggested source mapping for main.py:")
        print("KNOWN_SIGNAL_SOURCES = {")
        
        common_sources = source_counter.most_common(10)
        for source, count in common_sources:
            # Clean up the source name
            clean_source = source.strip()
            if len(clean_source) > 3 and count >= 2:
                print(f"    '{clean_source}': '{clean_source}',")
        
        print("    'Unknown': 'Unknown Source'")
        print("}")
        
        return source_counter
        
    except Exception as e:
        print(f"❌ Error analyzing sources: {e}")
        return {}

if __name__ == "__main__":
    extract_sources_from_messages()

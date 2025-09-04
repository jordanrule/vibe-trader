#!/usr/bin/env python3
"""
List all cryptocurrencies tradeable on Kraken
"""

import sys
import os

# Add current directory to path to import main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import TradingAgent

def list_all_cryptos():
    """Get and display all cryptocurrencies tradeable on Kraken"""
    agent = TradingAgent()
    
    # Get all tradeable pairs (not just filtered ones)
    import requests
    
    try:
        url = "https://api.kraken.com/0/public/AssetPairs"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data and data['error']:
            print(f"❌ Kraken API error: {data['error']}")
            return
            
        pairs = data.get('result', {})
        
        # Extract all unique base currencies that trade against USD
        crypto_assets = {}
        fiat_currencies = {'ZEUR', 'ZUSD', 'ZGBP', 'ZJPY', 'ZAUD', 'ZCAD', 'ZCHF'}
        
        for pair_id, pair_info in pairs.items():
            base = pair_info.get('base', '')
            quote = pair_info.get('quote', '')
            pair_name = pair_info.get('wsname', pair_id)
            
            # Only include pairs that trade against major fiat currencies
            if quote in fiat_currencies and base not in fiat_currencies:
                if base not in crypto_assets:
                    crypto_assets[base] = []
                crypto_assets[base].append({
                    'pair_name': pair_name,
                    'quote': quote,
                    'min_order': pair_info.get('ordermin', '0')
                })
        
        print(f"🪙 Found {len(crypto_assets)} unique cryptocurrencies on Kraken:\n")
        
        # Sort alphabetically and display
        for crypto in sorted(crypto_assets.keys()):
            usd_pairs = [p for p in crypto_assets[crypto] if p['quote'] == 'ZUSD']
            if usd_pairs:
                usd_pair = usd_pairs[0]
                print(f"{crypto:<12} -> {usd_pair['pair_name']:<15} (min: {usd_pair['min_order']})")
            else:
                # Show any pair if no USD pair available
                any_pair = crypto_assets[crypto][0]
                print(f"{crypto:<12} -> {any_pair['pair_name']:<15} (min: {any_pair['min_order']}) [No USD pair]")
        
        print(f"\n📊 Summary:")
        print(f"   Total cryptocurrencies: {len(crypto_assets)}")
        
        usd_tradeable = len([c for c in crypto_assets.values() if any(p['quote'] == 'ZUSD' for p in c)])
        print(f"   USD-tradeable: {usd_tradeable}")
        
        eur_tradeable = len([c for c in crypto_assets.values() if any(p['quote'] == 'ZEUR' for p in c)])
        print(f"   EUR-tradeable: {eur_tradeable}")
        
        # Show top 20 most popular (most trading pairs)
        popularity = [(crypto, len(pairs)) for crypto, pairs in crypto_assets.items()]
        popularity.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🔥 Top 20 most supported cryptos (by number of trading pairs):")
        for i, (crypto, pair_count) in enumerate(popularity[:20], 1):
            print(f"   {i:2d}. {crypto:<12} ({pair_count} pairs)")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    list_all_cryptos()

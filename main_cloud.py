#!/usr/bin/env python3
"""
Cloud Run HTTP server for the trading system
This file serves as the main entry point when deployed to Google Cloud Run
"""

import asyncio
import logging
import os
import json
from flask import Flask, request, jsonify
from main import TradingAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global trading agent instance
trading_agent = None

def get_trading_agent():
    """Get or create trading agent instance"""
    global trading_agent
    if trading_agent is None:
        logger.info("🔧 Initializing trading agent...")
        trading_agent = TradingAgent()
    return trading_agent

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint for Cloud Run"""
    return jsonify({
        'status': 'healthy',
        'service': 'vibe-trader',
        'version': '1.0.0'
    })

@app.route('/run-cycle', methods=['POST'])
def run_cycle():
    """Run a trading cycle"""
    try:
        logger.info("🚀 Starting trading cycle via HTTP request")

        # Get trading agent
        agent = get_trading_agent()

        # Run the trading cycle asynchronously
        # Note: Flask doesn't support async routes easily, so we'll run synchronously
        result = asyncio.run(agent.run_cycle())

        logger.info("✅ Trading cycle completed successfully")

        return jsonify({
            'status': 'success',
            'message': 'Trading cycle completed successfully'
        })

    except Exception as e:
        logger.error(f"❌ Error in trading cycle: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get system status"""
    try:
        agent = get_trading_agent()
        return jsonify({
            'status': 'operational',
            'cloud_mode': agent.is_cloud_mode,
            'live_mode': agent.config.get('live_mode', False)
        })
    except ImportError as e:
        if 'google' in str(e):
            # Handle case where Google Cloud libraries are not available (local testing)
            logger.warning("Google Cloud libraries not available - returning mock status for testing")
            return jsonify({
                'status': 'operational',
                'cloud_mode': True,
                'live_mode': False,
                'note': 'Google Cloud libraries not available (local testing mode)'
            })
        else:
            logger.error(f"❌ Import error: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    except Exception as e:
        logger.error(f"❌ Error getting status: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Kubernetes-style health check"""
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    # For local development
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"🚀 Starting Cloud Run server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)

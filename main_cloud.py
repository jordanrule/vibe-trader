#!/usr/bin/env python3
"""
Cloud Function entry point for the trading system
This file serves as the main entry point when deployed to Google Cloud Functions
"""

import asyncio
import logging
from main import TradingAgent

# Configure logging for Cloud Functions
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

async def run_trading_cycle(request):
    """
    Cloud Function entry point for running a trading cycle

    Args:
        request: HTTP request (ignored, we run the full cycle)

    Returns:
        HTTP response with status
    """
    try:
        logger.info("🚀 Starting trading cycle via Cloud Function")

        # Initialize trading agent (will automatically detect cloud mode)
        agent = TradingAgent()

        # Run the trading cycle (15 minutes of messages)
        await agent.run_cycle()

        logger.info("✅ Trading cycle completed successfully")

        return {
            'statusCode': 200,
            'body': {
                'status': 'success',
                'message': 'Trading cycle completed successfully'
            }
        }

    except Exception as e:
        logger.error(f"❌ Error in trading cycle: {e}")
        return {
            'statusCode': 500,
            'body': {
                'status': 'error',
                'message': str(e)
            }
        }

def cloud_function_handler(request):
    """
    Synchronous wrapper for Cloud Functions
    """
    return asyncio.run(run_trading_cycle(request))

# For local testing
if __name__ == "__main__":
    result = cloud_function_handler(None)
    print(f"Result: {result}")

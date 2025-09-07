#!/usr/bin/env python3
"""
Live Trading System Runner
Simplified version that runs main.py every hour
"""

import argparse
import asyncio
import os
import sys
import time
import signal
import logging
import subprocess
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class LiveTradingRunner:
    """Runs the trading system by calling main.py every hour"""
    
    def __init__(self):
        self.running = False
        self.cycle_count = 0
        self.start_time = None
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    async def run_cycle(self):
        """Run a single trading cycle by calling main.py"""
        try:
            self.cycle_count += 1
            current_time = datetime.now()

            logger.info(f"🔄 Starting trading cycle #{self.cycle_count} at {current_time}")

            # Build command - simplified, just run main.py
            cmd = [
                sys.executable, 'main.py'
            ]

            logger.info(f"Executing: {' '.join(cmd)}")

            # Run main.py as subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout (increased for complex operations)
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Trading cycle #{self.cycle_count} completed successfully")
                # Pass through all output from main.py (logs go to stderr)
                if result.stderr:
                    for line in result.stderr.strip().split('\n'):
                        if line.strip():
                            # Print directly to maintain original formatting
                            print(line)
                if result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            print(line)
            else:
                logger.error(f"❌ Trading cycle #{self.cycle_count} failed with return code {result.returncode}")
                # Pass through all stderr from main.py
                if result.stderr:
                    for line in result.stderr.strip().split('\n'):
                        if line.strip():
                            print(line)
                # Also show stdout for debugging
                if result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            print(line)
            
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Trading cycle #{self.cycle_count} timed out after 10 minutes")
        except Exception as e:
            logger.error(f"❌ Error in trading cycle #{self.cycle_count}: {e}")
    
    async def run(self):
        """Main run loop - executes every hour"""
        try:
            logger.info("🚀 Starting Live Trading System...")
            logger.info("📅 Running main.py every hour")
            logger.info("⏰ Press Ctrl+C to stop")

            self.running = True
            self.start_time = datetime.now()

            # Run cycles every hour
            while self.running:
                # Run cycle immediately
                await self.run_cycle()

                # Wait for 1 hour before next cycle
                if self.running:
                    logger.info("⏳ Waiting 1 hour before next cycle...")
                    for _ in range(3600):  # 3600 seconds = 1 hour
                        if not self.running:
                            break
                        await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        if self.start_time:
            uptime = datetime.now() - self.start_time
            logger.info(f"🛑 Live Trading System shutting down after {uptime}")
            logger.info(f"📊 Completed {self.cycle_count} trading cycles")
        else:
            logger.info("🛑 Live Trading System shutting down")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Live Trading System Runner')
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run a single test cycle and exit'
    )
    return parser.parse_args()

async def main():
    """Main function"""
    args = parse_arguments()
    runner = LiveTradingRunner()

    if args.test:
        logger.info("🧪 Running test cycle...")
        await runner.run_cycle()
        logger.info("✅ Test cycle completed")
    else:
        await runner.run()

if __name__ == "__main__":
    asyncio.run(main())

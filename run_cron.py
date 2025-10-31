#!/usr/bin/env python3
"""
Cron scheduler script for running the LangGraph agent periodically.
This script creates dummy runs on a schedule using the schedule library.
"""

import os
import schedule
import time
from datetime import datetime
from dotenv import load_dotenv

from my_agent.graph import graph

# Load environment variables
load_dotenv()


def run_agent():
    """Execute a single run of the agent graph."""
    print(f"\n{'='*60}")
    print(f"Starting agent run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    try:
        # Initial state with an empty message
        initial_state = {
            "messages": [{"role": "user", "content": "Hello, start processing"}]
        }

        # Run the graph
        result = graph.invoke(
            initial_state,
            config={
                "configurable": {
                    "model_name": "anthropic"
                }
            }
        )

        print(f"\n✓ Agent run completed successfully")
        print(f"Final state: {result}")

    except Exception as e:
        print(f"\n✗ Error during agent run: {e}")
        import traceback
        traceback.print_exc()

    print(f"{'='*60}\n")


def main():
    """Main function to set up and run the scheduler."""
    print("LangGraph Cron Scheduler Starting...")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Schedule the job to run every 5 minutes
    # You can adjust this to your preferred schedule
    # schedule.every(5).minutes.do(run_agent)

    # Alternative schedules (uncomment the one you want):
    # schedule.every(10).seconds.do(run_agent)  # Every 10 seconds (for testing)
    schedule.every(1).minutes.do(run_agent)   # Every minute
    # schedule.every().hour.do(run_agent)       # Every hour
    # schedule.every().day.at("10:00").do(run_agent)  # Every day at 10:00

    print("Scheduler configured. Running initial job...")

    # Run once immediately
    run_agent()

    print("Waiting for scheduled runs... (Press Ctrl+C to stop)")

    # Keep running scheduled jobs
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nScheduler stopped by user")


if __name__ == "__main__":
    main()

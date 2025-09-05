#!/usr/bin/env python3
"""
Extract signal source names from agent_trading.log
Usage: python extract_signals.py [bot_name] [--update-sources] [--log-file FILE]
"""
import re
import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime
import os

def extract_signal_sources_from_log(log_file='agent_trading.log', bot_name='botty'):
    """
    Extract signal source names from the agent trading log file

    Args:
        log_file: Path to the log file (default: agent_trading.log)
        bot_name: Bot name to search for (default: botty)

    Returns:
        dict: Dictionary containing extracted sources and analysis
    """
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return {}

    print(f"🔍 Extracting signal sources from {log_file}")
    print("-" * 60)

    # Patterns to match message processing lines
    message_pattern = rf"Processing message from {re.escape(bot_name)}[:\s]+(.+)"

    sources_found = []
    message_details = []
    source_contexts = defaultdict(list)

    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                # Find message processing lines
                message_match = re.search(message_pattern, line)
                if message_match:
                    message_content = message_match.group(1).strip()

                    # Skip empty or very short messages
                    if len(message_content) < 3:
                        continue

                    # Extract source from message content
                    detected_sources = extract_sources_from_message(message_content)

                    if detected_sources:
                        sources_found.extend(detected_sources)
                        message_details.append({
                            'line': line_num,
                            'timestamp': extract_timestamp_from_line(line),
                            'message': message_content,
                            'sources': detected_sources
                        })

                        # Store context for each source
                        for source in detected_sources:
                            source_contexts[source].append(message_content)

                        print(f"📱 Line {line_num}: {message_content}")
                        print(f"   🎯 Sources: {detected_sources}")
                        print()

    except Exception as e:
        print(f"❌ Error reading log file: {e}")
        return {}

    # Analyze results
    source_counter = Counter(sources_found)

    results = {
        'total_messages': len(message_details),
        'unique_sources': len(source_counter),
        'source_counter': dict(source_counter.most_common()),
        'message_details': message_details,
        'source_contexts': dict(source_contexts)
    }

    return results

def extract_sources_from_message(message_content):
    """
    Extract signal source names from a single message content
    """
    detected_sources = []

    # Clean and prepare the message
    message_lower = message_content.lower()

    # Skip messages that are clearly not signals
    skip_indicators = [
        'test', 'empty', 'no cryptocurrencies detected',
        'skip', 'unknown', 'error', 'failed'
    ]

    if any(indicator in message_lower for indicator in skip_indicators):
        return detected_sources

    # Try different extraction methods
    lines = message_content.split('\n')
    first_line = lines[0] if lines else message_content

    # Method 1: Direct header extraction
    header_sources = extract_header_sources(first_line)
    detected_sources.extend(header_sources)

    # Method 2: Pattern-based extraction
    pattern_sources = extract_pattern_sources(message_content)
    detected_sources.extend(pattern_sources)

    # Remove duplicates while preserving order
    seen = set()
    unique_sources = []
    for source in detected_sources:
        if source not in seen and len(source.strip()) > 2:
            seen.add(source)
            unique_sources.append(source.strip())

    return unique_sources

def extract_header_sources(first_line):
    """
    Extract sources from the first line of a message
    """
    sources = []

    # Common header patterns
    header_patterns = [
        r'^([^\n:]+)',  # Everything before first colon
        r'^([^(\n]+)',  # Everything before first parenthesis
        r'^([^\d\n]+)',  # Everything before first number
    ]

    for pattern in header_patterns:
        matches = re.findall(pattern, first_line.strip())
        for match in matches:
            candidate = match.strip()
            # Filter out common non-source text
            if (len(candidate) > 3 and
                not candidate.lower().startswith(('binance', 'kucoin', 'bybit', 'okx')) and
                not any(word in candidate.lower() for word in ['futures', 'spot', 'usd', 'usdt', 'btc', 'eth']) and
                not candidate.isdigit()):
                sources.append(candidate)
                break  # Take first good match

    return sources

def extract_pattern_sources(message_content):
    """
    Extract sources using pattern matching
    """
    sources = []

    # Pattern for signal sources
    signal_patterns = [
        r'([A-Z][a-zA-Z\s]*(?:Trading|Signals?|Crypto|Official|Pro|Premium|Club|Paradise)[\w\s]*)',
        r'([A-Z][a-zA-Z\s]*(?:Signals?|Trading|Crypto)[\w\s]*)',
    ]

    for pattern in signal_patterns:
        matches = re.findall(pattern, message_content, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            candidate = match.strip()
            if len(candidate) > 4:  # Minimum length filter
                sources.append(candidate)

    return sources

def extract_timestamp_from_line(line):
    """
    Extract timestamp from a log line
    """
    # Pattern for log timestamps
    timestamp_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})'
    match = re.search(timestamp_pattern, line)
    if match:
        return match.group(1)
    return "Unknown"

def display_results(results):
    """
    Display the analysis results
    """
    if not results:
        print("❌ No results to display")
        return

    print("\n" + "="*80)
    print("📊 SIGNAL SOURCE EXTRACTION RESULTS")
    print("="*80)

    print("\n📈 Summary:")
    print(f"   • Messages processed: {results['total_messages']}")
    print(f"   • Unique sources found: {results['unique_sources']}")

    if results['source_counter']:
        print("\n🔝 Most Common Sources:")
        for i, (source, count) in enumerate(results['source_counter'].items(), 1):
            print(f"   {i}. {source}: {count} times")
    if results['message_details']:
        print("\n📝 Sample Messages:")
        for i, detail in enumerate(results['message_details'][:5], 1):
            print(f"\n   {i}. {detail['message']}")
            print(f"      Sources: {detail['sources']}")

def update_known_sources(results, openai_file='services/openai.py'):
    """
    Update the KNOWN_SIGNAL_SOURCES in openai.py with newly found sources
    """
    if not results or not results['source_counter']:
        print("❌ No sources to update")
        return

    try:
        # Read current openai.py file
        with open(openai_file, 'r') as f:
            content = f.read()

        # Find the KNOWN_SIGNAL_SOURCES section
        sources_pattern = r'KNOWN_SIGNAL_SOURCES\s*=\s*\{([^}]+)\}'
        sources_match = re.search(sources_pattern, content, re.DOTALL)

        if not sources_match:
            print(f"❌ Could not find KNOWN_SIGNAL_SOURCES in {openai_file}")
            return

        current_sources = sources_match.group(1)

        # Parse existing sources
        existing_sources = set()
        for line in current_sources.split('\n'):
            line = line.strip()
            if ':' in line and not line.startswith('#'):
                key = line.split(':')[0].strip().strip("'\"")
                existing_sources.add(key)

        # Find new sources
        new_sources = []
        for source in results['source_counter'].keys():
            if source not in existing_sources:
                new_sources.append(source)

        if not new_sources:
            print("✅ No new sources to add - all sources already exist")
            return

        # Add new sources to the file
        print(f"📝 Adding {len(new_sources)} new sources to {openai_file}:")

        # Find the insertion point (before the last entry)
        lines = content.split('\n')
        insert_index = -1
        for i, line in enumerate(lines):
            if line.strip().startswith("'Unknown':"):
                insert_index = i
                break

        if insert_index == -1:
            print("❌ Could not find insertion point")
            return

        # Prepare new source lines
        new_source_lines = []
        for source in sorted(new_sources):
            new_source_lines.append(f"    '{source}': '{source}',")
            print(f"   + '{source}'")

        # Insert new sources
        lines[insert_index:insert_index] = new_source_lines

        # Write back to file
        with open(openai_file, 'w') as f:
            f.write('\n'.join(lines))

        print(f"✅ Successfully updated {openai_file}")

    except Exception as e:
        print(f"❌ Error updating sources: {e}")

def main():
    """Main function"""
    # Handle positional argument for bot name
    bot_name = 'Jordan'  # Default
    remaining_args = sys.argv[1:]

    # Check if first argument is a bot name (doesn't start with --)
    if remaining_args and not remaining_args[0].startswith('--'):
        bot_name = remaining_args[0]
        remaining_args = remaining_args[1:]

    # Parse remaining arguments
    parser = argparse.ArgumentParser(description='Extract signal sources from agent trading log')
    parser.add_argument('--log-file', '-f', default='agent_trading.log',
                       help='Log file to analyze (default: agent_trading.log)')
    parser.add_argument('--update-sources', '-u', action='store_true',
                       help='Update KNOWN_SIGNAL_SOURCES in openai.py with new sources')
    parser.add_argument('--output', '-o', default=None,
                       help='Save results to JSON file')

    args = parser.parse_args(remaining_args)

    # Extract sources
    results = extract_signal_sources_from_log(args.log_file, bot_name)

    if not results:
        return

    # Display results
    display_results(results)

    # Save to file if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {args.output}")

    # Update sources if requested
    if args.update_sources:
        print("\n🔄 Updating KNOWN_SIGNAL_SOURCES...")
        update_known_sources(results)

if __name__ == "__main__":
    main()
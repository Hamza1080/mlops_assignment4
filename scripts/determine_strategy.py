#!/usr/bin/env python3
"""
Determine retraining strategy based on drift analysis.
Decides whether to retrain or continue monitoring.
"""

import argparse
import json
import sys
from typing import Dict, Any


def determine_strategy(drift_report: Dict[str, Any]) -> Dict[str, str]:
    """
    Determine retraining strategy based on drift severity.
    
    Args:
        drift_report: Drift analysis results
    
    Returns:
        Strategy decision with action and reason
    """
    strategy = {
        'action': 'monitor',
        'reason': 'Drift within acceptable threshold',
        'confidence': 0.95
    }
    
    severity = drift_report.get('severity', 'unknown')
    
    if severity == 'critical':
        strategy.update({
            'action': 'retrain',
            'reason': 'Critical performance degradation detected',
            'confidence': 0.99
        })
    
    elif severity == 'high':
        strategy.update({
            'action': 'retrain',
            'reason': 'Significant data drift detected (PSI > 0.20)',
            'confidence': 0.90
        })
    
    elif severity == 'medium':
        strategy.update({
            'action': 'monitor_closely',
            'reason': 'Moderate drift detected. Monitor for escalation.',
            'confidence': 0.75
        })
    
    else:
        strategy.update({
            'action': 'monitor',
            'reason': 'No significant drift detected',
            'confidence': 0.80
        })
    
    print(f"Strategy: {strategy['action']}")
    print(f"Reason: {strategy['reason']}")
    print(f"Confidence: {strategy['confidence']:.0%}")
    
    return strategy


def main():
    """Main strategy determination routine."""
    parser = argparse.ArgumentParser(description='Determine retraining strategy')
    parser.add_argument('--drift-report', type=str, required=True, help='Path to drift report')
    parser.add_argument('--output', type=str, default='strategy.json', help='Output file')
    
    args = parser.parse_args()
    
    # Load drift report
    try:
        with open(args.drift_report, 'r') as f:
            drift_report = json.load(f)
    except Exception as e:
        print(f"Failed to load drift report: {e}")
        sys.exit(1)
    
    # Determine strategy
    strategy = determine_strategy(drift_report)
    
    # Add drift context
    strategy['drift_context'] = drift_report
    
    # Save strategy
    with open(args.output, 'w') as f:
        json.dump(strategy, f, indent=2)
    
    print(f"Strategy saved to {args.output}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Script to extract loss values from TensorBoard logs
"""

import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_loss_from_logs(log_dir):
    """Extract loss values from TensorBoard logs"""
    
    print(f"Reading logs from: {log_dir}")
    
    # Find all event files in the log directory
    event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    
    if not event_files:
        print("No TensorBoard event files found!")
        return
    
    # Process the most recent event file
    latest_file = sorted(event_files)[-1]
    event_file_path = os.path.join(log_dir, latest_file)
    
    print(f"Processing: {latest_file}")
    
    # Initialize the event accumulator
    ea = EventAccumulator(event_file_path)
    ea.Reload()  # loads events from file
    
    print("Available tags:", ea.Tags()['scalars'])
    
    # Try different possible loss tag names
    loss_tags = ['train/loss', 'loss', 'train_loss', 'training_loss', 'Loss']
    
    for tag in loss_tags:
        if tag in ea.Tags()['scalars']:
            print(f"\nFound loss data in tag: {tag}")
            scalar_events = ea.Scalars(tag)
            
            print("Step -> Loss:")
            for i, event in enumerate(scalar_events):
                print(f"  {event.step}: {event.value:.6f}")
                if i >= 50:  # Limit to first 50 values
                    print("  ... (showing first 50 values)")
                    break
            return
    
    print("No loss data found! Available scalar tags:")
    for tag in ea.Tags()['scalars']:
        print(f"  - {tag}")

if __name__ == "__main__":
    log_dir = "/home/ken/workspace/TinyLLaVA_Factory/logs"
    extract_loss_from_logs(log_dir)
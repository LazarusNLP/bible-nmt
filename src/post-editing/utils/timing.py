"""
Timing utilities for performance analysis of post-editing pipeline components.
"""

import time
import json
import os
from typing import Dict, List, Optional
from collections import defaultdict
from pathlib import Path


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str, tracker: 'TimingTracker' = None):
        self.name = name
        self.tracker = tracker
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        if self.tracker:
            self.tracker.record_time(self.name, self.duration)


class TimingTracker:
    """Tracks timing information for different components of the pipeline."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.timings = defaultdict(list)  # component_name -> [duration1, duration2, ...]
        self.batch_timings = []  # List of dicts with timing info per batch
        self.total_start_time = None
        self.total_end_time = None
        
    def start_total_timing(self):
        """Start timing the entire pipeline."""
        self.total_start_time = time.time()
    
    def end_total_timing(self):
        """End timing the entire pipeline."""
        self.total_end_time = time.time()
    
    def record_time(self, component: str, duration: float):
        """Record timing for a component."""
        self.timings[component].append(duration)
    
    def record_batch_timing(self, batch_info: Dict):
        """Record timing information for a batch."""
        self.batch_timings.append(batch_info)
    
    def timer(self, name: str) -> Timer:
        """Create a timer context manager."""
        return Timer(name, self)
    
    def get_statistics(self) -> Dict:
        """Get timing statistics."""
        stats = {}
        
        # Calculate statistics for each component
        for component, times in self.timings.items():
            if times:
                stats[component] = {
                    'total_time': sum(times),
                    'average_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'call_count': len(times)
                }
        
        # Add total pipeline time
        if self.total_start_time and self.total_end_time:
            stats['total_pipeline_time'] = self.total_end_time - self.total_start_time
        
        # Add batch information
        stats['batch_timings'] = self.batch_timings
        
        return stats
    
    def print_summary(self):
        """Print timing summary to console."""
        stats = self.get_statistics()
        
        print("\n" + "="*80)
        print("⏱️  PERFORMANCE TIMING ANALYSIS")
        print("="*80)
        
        # Print component statistics
        print("\n📊 COMPONENT PERFORMANCE (Average per call):")
        print("-" * 60)
        
        # Sort by average time (descending)
        components = [(name, data) for name, data in stats.items() 
                     if isinstance(data, dict) and 'average_time' in data]
        components.sort(key=lambda x: x[1]['average_time'], reverse=True)
        
        for component, data in components:
            avg_time = data['average_time']
            total_time = data['total_time']
            call_count = data['call_count']
            
            print(f"{component:.<40} {avg_time:>8.2f}s avg ({total_time:>7.1f}s total, {call_count:>3d} calls)")
        
        # Print batch timing summary if available
        if stats.get('batch_timings'):
            print(f"\n📈 BATCH PROCESSING SUMMARY:")
            print("-" * 60)
            
            total_batches = len(stats['batch_timings'])
            if total_batches > 0:
                # Calculate batch averages
                batch_totals = [batch.get('total_batch_time', 0) for batch in stats['batch_timings']]
                avg_batch_time = sum(batch_totals) / len(batch_totals) if batch_totals else 0
                
                print(f"Total batches processed: {total_batches}")
                print(f"Average batch time: {avg_batch_time:.2f}s")
                print(f"Fastest batch: {min(batch_totals):.2f}s")
                print(f"Slowest batch: {max(batch_totals):.2f}s")
        
        # Print total pipeline time
        if 'total_pipeline_time' in stats:
            total_time = stats['total_pipeline_time']
            print(f"\n🏁 TOTAL PIPELINE TIME: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        
        print("="*80)
    
    def save_to_file(self, filename: str = "timing_analysis.txt"):
        """Save detailed timing analysis to file."""
        filepath = os.path.join(self.output_dir, filename)
        stats = self.get_statistics()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("⏱️  POST-EDITING PIPELINE PERFORMANCE ANALYSIS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Component performance
            f.write("📊 COMPONENT PERFORMANCE BREAKDOWN:\n")
            f.write("-" * 60 + "\n")
            
            # Sort by total time (descending) to show biggest bottlenecks
            components = [(name, data) for name, data in stats.items() 
                         if isinstance(data, dict) and 'total_time' in data]
            components.sort(key=lambda x: x[1]['total_time'], reverse=True)
            
            for component, data in components:
                f.write(f"\n{component.upper()}:\n")
                f.write(f"  Total Time: {data['total_time']:.2f}s\n")
                f.write(f"  Average Time: {data['average_time']:.2f}s\n")
                f.write(f"  Min Time: {data['min_time']:.2f}s\n")
                f.write(f"  Max Time: {data['max_time']:.2f}s\n")
                f.write(f"  Call Count: {data['call_count']}\n")
                
                # Calculate percentage of total time
                if 'total_pipeline_time' in stats:
                    percentage = (data['total_time'] / stats['total_pipeline_time']) * 100
                    f.write(f"  % of Total Pipeline: {percentage:.1f}%\n")
            
            # Batch details
            if stats.get('batch_timings'):
                f.write(f"\n📈 DETAILED BATCH TIMINGS:\n")
                f.write("-" * 60 + "\n")
                
                for i, batch in enumerate(stats['batch_timings'], 1):
                    f.write(f"\nBatch {i}:\n")
                    f.write(f"  Rows processed: {batch.get('rows_processed', 'N/A')}\n")
                    f.write(f"  Total batch time: {batch.get('total_batch_time', 0):.2f}s\n")
                    
                    # Write component times for this batch
                    for component, duration in batch.items():
                        if component not in ['rows_processed', 'total_batch_time', 'batch_id']:
                            f.write(f"  {component}: {duration:.2f}s\n")
            
            # Pipeline summary
            if 'total_pipeline_time' in stats:
                f.write(f"\n🏁 PIPELINE SUMMARY:\n")
                f.write("-" * 60 + "\n")
                f.write(f"Total execution time: {stats['total_pipeline_time']:.2f}s\n")
                f.write(f"Total execution time: {stats['total_pipeline_time']/60:.2f} minutes\n")
                
                # Add throughput information if available
                total_rows = sum(batch.get('rows_processed', 0) for batch in stats.get('batch_timings', []))
                if total_rows > 0:
                    throughput = total_rows / stats['total_pipeline_time']
                    f.write(f"Total rows processed: {total_rows}\n")
                    f.write(f"Throughput: {throughput:.2f} rows/second\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"📄 Detailed timing analysis saved to: {filepath}")
        
        # Also save as JSON for programmatic access
        json_filepath = filepath.replace('.txt', '.json')
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"📄 JSON timing data saved to: {json_filepath}")
        
        return filepath

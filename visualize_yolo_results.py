import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Read CSV
results = []
with open('results/results_augmented/distorted_yolo_results.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    results = list(reader)

# Calculate statistics
total_rows = len(results)
total_detections = sum(int(r['detections']) for r in results)
avg_detections = total_detections / total_rows if total_rows > 0 else 0

# Images with at least one detection
images_with_detections = sum(1 for r in results if int(r['detections']) > 0)
success_rate = (images_with_detections / total_rows * 100) if total_rows > 0 else 0

# Stats by distortion
dist_stats = defaultdict(lambda: {'count': 0, 'detections': 0, 'max': 0, 'with_detection': 0})
for row in results:
    dist = row['distortion']
    dets = int(row['detections'])
    dist_stats[dist]['count'] += 1
    dist_stats[dist]['detections'] += dets
    dist_stats[dist]['max'] = max(dist_stats[dist]['max'], dets)
    if dets > 0:
        dist_stats[dist]['with_detection'] += 1

# Print summary
print("\n" + "="*80)
print("📊 YOLO DETECTION RESULTS - COMPREHENSIVE ANALYSIS")
print("="*80)
print(f"\n📈 Overall Statistics:")
print(f"   Total images processed: {total_rows}")
print(f"   Total detections: {total_detections}")
print(f"   Average detections per image: {avg_detections:.2f}")
print(f"   ✅ Images with at least 1 detection: {images_with_detections}/{total_rows}")
print(f"   🎯 SUCCESS RATE: {success_rate:.1f}%")

print(f"\n📋 Detailed Breakdown by Distortion Method:\n")
print(f"{'Method':<20} {'Total':>6} {'Avg':>6} {'Max':>4} {'Success':>8} {'Rate':>7}")
print("-" * 60)

distortion_order = ['original', 'gaussian_noise', 'salt_and_pepper', 'gaussian_blur', 
                    'brightness_contrast', 'rotation', 'perspective_warp', 
                    'barrel_distortion', 'pixelation']

dist_names = []
dist_totals = []
dist_success_rates = []

for dist in distortion_order:
    if dist in dist_stats:
        stats = dist_stats[dist]
        avg = stats['detections'] / stats['count'] if stats['count'] > 0 else 0
        success = stats['with_detection']
        success_pct = (success / stats['count'] * 100) if stats['count'] > 0 else 0
        
        print(f"{dist:<20} {stats['detections']:>6} {avg:>6.2f} {stats['max']:>4} {success:>8}/{stats['count']} {success_pct:>6.1f}%")
        
        dist_names.append(dist)
        dist_totals.append(stats['detections'])
        dist_success_rates.append(success_pct)

print("=" * 80)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('YOLO Detection Analysis - Distortion Impact', fontsize=16, fontweight='bold')

# Plot 1: Total detections by method
ax1 = axes[0, 0]
bars1 = ax1.bar(range(len(dist_names)), dist_totals, color='steelblue', alpha=0.8)
ax1.set_xticks(range(len(dist_names)))
ax1.set_xticklabels(dist_names, rotation=45, ha='right')
ax1.set_ylabel('Total Detections', fontweight='bold')
ax1.set_title('Total Detections by Distortion Method')
ax1.grid(axis='y', alpha=0.3)
for i, v in enumerate(dist_totals):
    ax1.text(i, v + 1, str(v), ha='center', fontweight='bold')

# Plot 2: Success rate by method
ax2 = axes[0, 1]
bars2 = ax2.bar(range(len(dist_names)), dist_success_rates, color='forestgreen', alpha=0.8)
ax2.set_xticks(range(len(dist_names)))
ax2.set_xticklabels(dist_names, rotation=45, ha='right')
ax2.set_ylabel('Success Rate (%)', fontweight='bold')
ax2.set_title('Detection Success Rate by Distortion Method')
ax2.set_ylim([0, 100])
ax2.grid(axis='y', alpha=0.3)
for i, v in enumerate(dist_success_rates):
    ax2.text(i, v + 2, f'{v:.0f}%', ha='center', fontweight='bold', fontsize=9)

# Plot 3: Average detections per image
ax3 = axes[1, 0]
avg_detections_by_method = []
for dist in distortion_order:
    if dist in dist_stats:
        stats = dist_stats[dist]
        avg = stats['detections'] / stats['count'] if stats['count'] > 0 else 0
        avg_detections_by_method.append(avg)

bars3 = ax3.bar(range(len(dist_names)), avg_detections_by_method, color='coral', alpha=0.8)
ax3.set_xticks(range(len(dist_names)))
ax3.set_xticklabels(dist_names, rotation=45, ha='right')
ax3.set_ylabel('Avg Detections per Image', fontweight='bold')
ax3.set_title('Average Detections per Image by Distortion')
ax3.grid(axis='y', alpha=0.3)
for i, v in enumerate(avg_detections_by_method):
    ax3.text(i, v + 0.05, f'{v:.2f}', ha='center', fontweight='bold', fontsize=9)

# Plot 4: Summary statistics
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
SUMMARY STATISTICS

Total Images Processed: {total_rows}
Total Detections: {total_detections}
Overall Avg Detections: {avg_detections:.2f} per image

Overall Success Rate: {success_rate:.1f}%
({images_with_detections} out of {total_rows} images detected)

Best Method: {max(dist_success_rates) == max(dist_success_rates) and dist_names[dist_success_rates.index(max(dist_success_rates))] or 'N/A'}
  Success Rate: {max(dist_success_rates):.1f}%

Worst Method: {dist_names[dist_success_rates.index(min(dist_success_rates))]}
  Success Rate: {min(dist_success_rates):.1f}%

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('results/results_augmented/yolo_detection_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ Graph saved to: results/results_augmented/yolo_detection_analysis.png")

print("\n" + "="*80)
print("🎯 SUCCESS METRICS")
print("="*80)
print(f"\n✓ Overall Success Rate: {success_rate:.1f}%")
print(f"✓ {images_with_detections} out of {total_rows} images had detections")
print(f"✓ Average detections when found: {total_detections / images_with_detections:.2f}" if images_with_detections > 0 else "")
print("\n" + "="*80 + "\n")

plt.show()

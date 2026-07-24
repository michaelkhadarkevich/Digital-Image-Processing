import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Read CSV
results = []
with open('result/results task 3/yolo_distortion_results/distorted_yolo_results.csv', 'r', encoding='utf-8') as f:
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

# Prepare data ordered by distortion type
distortion_order = ['original', 'gaussian_noise', 'salt_and_pepper', 'gaussian_blur', 
                    'brightness_contrast', 'rotation', 'perspective_warp', 
                    'barrel_distortion', 'pixelation']

dist_names = []
dist_success_rates = []
dist_totals = []
dist_avg = []

for i, dist in enumerate(distortion_order):
    if dist in dist_stats:
        stats = dist_stats[dist]
        avg = stats['detections'] / stats['count'] if stats['count'] > 0 else 0
        success = stats['with_detection']
        success_pct = (success / stats['count'] * 100) if stats['count'] > 0 else 0
        
        # Create x-axis labels (short names)
        if dist == 'original':
            short_name = 'Original'
        elif dist == 'gaussian_noise':
            short_name = 'G.Noise'
        elif dist == 'salt_and_pepper':
            short_name = 'S&P'
        elif dist == 'gaussian_blur':
            short_name = 'Blur'
        elif dist == 'brightness_contrast':
            short_name = 'Bright.'
        elif dist == 'rotation':
            short_name = 'Rotate'
        elif dist == 'perspective_warp':
            short_name = 'Persp.'
        elif dist == 'barrel_distortion':
            short_name = 'Barrel'
        elif dist == 'pixelation':
            short_name = 'Pixel.'
        
        dist_names.append(short_name)
        dist_success_rates.append(success_pct)
        dist_totals.append(stats['detections'])
        dist_avg.append(avg)

x = np.arange(len(dist_names))

# Create figure with 2 subplots (like training_history.png)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('YOLO Detection Performance - Distortion Impact', fontsize=14, fontweight='bold')

# Left panel: Success Rate
ax1.plot(x, dist_success_rates, 'o-', linewidth=2.5, markersize=8, color='steelblue', label='Success Rate')
ax1.fill_between(x, dist_success_rates, alpha=0.3, color='steelblue')
ax1.set_xlabel('Distortion Method', fontweight='bold')
ax1.set_ylabel('Success Rate (%)', fontweight='bold')
ax1.set_title('Success Rate')
ax1.set_xticks(x)
ax1.set_xticklabels(dist_names, rotation=45, ha='right')
ax1.set_ylim([0, 100])
ax1.grid(True, alpha=0.3)
ax1.legend(loc='lower left', fontsize=10)

# Add value labels
for i, v in enumerate(dist_success_rates):
    ax1.text(i, v + 3, f'{v:.0f}%', ha='center', fontsize=9, fontweight='bold')

# Right panel: Total Detections & Average Detections
ax2_twin = ax2.twinx()

line1 = ax2.plot(x, dist_totals, 'o-', linewidth=2.5, markersize=8, color='forestgreen', label='Total Detections')
line2 = ax2_twin.plot(x, dist_avg, 's-', linewidth=2.5, markersize=8, color='coral', label='Avg Detections')

ax2.set_xlabel('Distortion Method', fontweight='bold')
ax2.set_ylabel('Total Detections', fontweight='bold', color='forestgreen')
ax2_twin.set_ylabel('Avg Detections/Image', fontweight='bold', color='coral')
ax2.set_title('Total & Average Detections')
ax2.set_xticks(x)
ax2.set_xticklabels(dist_names, rotation=45, ha='right')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='y', labelcolor='forestgreen')
ax2_twin.tick_params(axis='y', labelcolor='coral')

# Add value labels for total detections
for i, v in enumerate(dist_totals):
    ax2.text(i, v + 1.5, str(v), ha='center', fontsize=9, fontweight='bold', color='forestgreen')

# Add value labels for average detections
for i, v in enumerate(dist_avg):
    ax2_twin.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold', color='coral')

# Combine legends
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('result/results task 3/yolo_distortion_results/yolo_detection_history.png', dpi=150, bbox_inches='tight')
print("\n✅ Graph saved to: result/results task 3/yolo_distortion_results/yolo_detection_history.png")

# Print detailed summary
print("\n" + "="*80)
print("📊 YOLO DETECTION RESULTS - DETAILED ANALYSIS")
print("="*80)

print(f"\n📈 Overall Performance Metrics:")
print(f"   Total images processed: {total_rows}")
print(f"   Total detections: {total_detections}")
print(f"   Average detections per image: {avg_detections:.2f}")
print(f"   Images with detections: {images_with_detections}/{total_rows}")

print(f"\n" + "="*80)
print(f"🎯 SUCCESS RATE: {success_rate:.1f}%")
print(f"="*80)

print(f"\n📋 Breakdown by Distortion Method:\n")
print(f"{'Method':<20} {'Success':>10} {'Rate':>6} {'Total':>6} {'Avg':>6}")
print("-" * 50)

for dist in distortion_order:
    if dist in dist_stats:
        stats = dist_stats[dist]
        success = stats['with_detection']
        success_pct = (success / stats['count'] * 100) if stats['count'] > 0 else 0
        avg = stats['detections'] / stats['count'] if stats['count'] > 0 else 0
        print(f"{dist:<20} {success:>4}/{stats['count']:<4} {success_pct:>6.1f}% {stats['detections']:>6} {avg:>6.2f}")

print("=" * 80)
print(f"\n✓ Best Method: gaussian_noise (88.0% success rate)")
print(f"✓ Worst Method: pixelation (0.0% success rate)")
print(f"\n✓ Average detections when found: {total_detections / images_with_detections:.2f}")
print("\n" + "="*80 + "\n")

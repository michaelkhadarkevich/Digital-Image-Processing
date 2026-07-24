import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Read CSV
results = []
with open('result/results task 3/yolo_distortion_results/distorted_yolo_results.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    results = list(reader)

# Calculate cumulative statistics
cumulative_detections = []
cumulative_images_with_detection = []
cumulative_success_rate = []

total_detections = 0
total_images_with_detection = 0

for i, row in enumerate(results):
    dets = int(row['detections'])
    total_detections += dets
    if dets > 0:
        total_images_with_detection += 1
    
    cumulative_detections.append(total_detections)
    cumulative_images_with_detection.append(total_images_with_detection)
    success_rate = (total_images_with_detection / (i + 1) * 100) if (i + 1) > 0 else 0
    cumulative_success_rate.append(success_rate)

# Prepare x-axis (image progress)
image_indices = np.arange(1, len(results) + 1)

# Print summary
print("\n" + "="*80)
print("📊 YOLO DETECTION PROGRESS ANALYSIS")
print("="*80)

print(f"\n📈 Final Statistics:")
print(f"   Total images processed: {len(results)}")
print(f"   Total cumulative detections: {cumulative_detections[-1]}")
print(f"   Total images with detections: {cumulative_images_with_detection[-1]}")
print(f"   Final success rate: {cumulative_success_rate[-1]:.1f}%")

print("\n" + "="*80)
print("🎯 PROGRESS MILESTONES")
print("="*80)

milestones = [50, 100, 150, 200, 250, 300, 350, 400, 450]
for milestone in milestones:
    if milestone <= len(results):
        idx = milestone - 1
        print(f"After {milestone:3d} images: {cumulative_detections[idx]:3d} detections, {cumulative_success_rate[idx]:5.1f}% success rate")

print("\n" + "="*80 + "\n")

# Create figure with 2 subplots (like training_history.png)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('YOLO Detection Progress Over Time', fontsize=14, fontweight='bold')

# Left panel: Cumulative Detections & Images with Detection
ax1_twin = ax1.twinx()

line1 = ax1.plot(image_indices, cumulative_detections, 'o-', linewidth=2, markersize=4, 
                 color='steelblue', label='Cumulative Detections', alpha=0.8)
ax1.fill_between(image_indices, cumulative_detections, alpha=0.2, color='steelblue')

line2 = ax1_twin.plot(image_indices, cumulative_images_with_detection, 's-', linewidth=2, markersize=4, 
                      color='forestgreen', label='Images with Detection', alpha=0.8)
ax1_twin.fill_between(image_indices, cumulative_images_with_detection, alpha=0.2, color='forestgreen')

ax1.set_xlabel('Image Processing Progress', fontweight='bold')
ax1.set_ylabel('Cumulative Detections', fontweight='bold', color='steelblue')
ax1_twin.set_ylabel('Images with Detection', fontweight='bold', color='forestgreen')
ax1.set_title('Cumulative Detection Growth')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1_twin.tick_params(axis='y', labelcolor='forestgreen')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

# Right panel: Success Rate Progress
ax2.plot(image_indices, cumulative_success_rate, 'o-', linewidth=2.5, markersize=5, 
         color='coral', label='Cumulative Success Rate')
ax2.fill_between(image_indices, cumulative_success_rate, alpha=0.3, color='coral')
ax2.set_xlabel('Image Processing Progress', fontweight='bold')
ax2.set_ylabel('Cumulative Success Rate (%)', fontweight='bold', color='coral')
ax2.set_title('Success Rate Over Progress')
ax2.set_ylim([0, 100])
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='y', labelcolor='coral')
ax2.legend(loc='lower right', fontsize=10)

# Add milestone markers
milestones_to_mark = [100, 200, 300, 400]
for milestone in milestones_to_mark:
    if milestone <= len(results):
        idx = milestone - 1
        ax2.axvline(x=milestone, color='gray', linestyle='--', alpha=0.3)
        ax2.text(milestone, cumulative_success_rate[-1] + 2, f'{milestone}', 
                ha='center', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig('result/results task 3/yolo_distortion_results/yolo_detection_progress.png', dpi=150, bbox_inches='tight')
print("✅ Graph saved to: result/results task 3/yolo_distortion_results/yolo_detection_progress.png")

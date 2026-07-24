import csv
from collections import defaultdict

# Read CSV
results = []
with open('result/results task 3/yolo_distortion_results/distorted_yolo_results.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    results = list(reader)

# Calculate statistics
total_rows = len(results)
total_detections = sum(int(r['detections']) for r in results)
avg_detections = total_detections / total_rows if total_rows > 0 else 0

# Stats by distortion
dist_stats = defaultdict(lambda: {'count': 0, 'detections': 0, 'max': 0})
for row in results:
    dist = row['distortion']
    dets = int(row['detections'])
    dist_stats[dist]['count'] += 1
    dist_stats[dist]['detections'] += dets
    dist_stats[dist]['max'] = max(dist_stats[dist]['max'], dets)

print("="*70)
print("📊 SUMMARY - Distortions + YOLO Detection Results")
print("="*70)
print(f"\n📈 Overall Statistics:")
print(f"   Total rows: {total_rows}")
print(f"   Total detections: {total_detections}")
print(f"   Average detections per image: {avg_detections:.2f}")

print(f"\n📋 By Distortion Method:")
for dist in sorted(dist_stats.keys()):
    stats = dist_stats[dist]
    avg = stats['detections'] / stats['count']
    print(f"   {dist:20} {stats['detections']:4} total  |  {avg:5.2f} avg  |  {stats['max']:2} max")

print("\n" + "="*70)

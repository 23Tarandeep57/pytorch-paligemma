import subprocess
import time
import matplotlib.pyplot as plt
import os
import sys

print("Starting inference with memory monitoring...")

proc = subprocess.Popen(
    ['python', 'inference.py', 
     '--model_path', os.path.expanduser('~/Documents/paligemma-3b-pt-224'),
     '--prompt', 'describe this image',
     '--image_file_path', 'test_images/pizza.jpeg',
     '--max_tokens_to_generate', '20',
     '--only_cpu', 'True'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)


timestamps = []
memory_mb = []
output_lines = []
start_time = time.time()

print("Monitoring memory usage...")
print("-" * 70)

try:
    while True:
       
        poll_result = proc.poll()
        
        try:
            ps_output = subprocess.check_output(
                ['ps', '-p', str(proc.pid), '-o', 'rss='],
                text=True,
                stderr=subprocess.DEVNULL
            ).strip()
            
            if ps_output:
                mem_kb = int(ps_output)
                mem_mb_val = mem_kb / 1024
                elapsed = time.time() - start_time
                
                timestamps.append(elapsed)
                memory_mb.append(mem_mb_val)
                
                print(f"Time: {elapsed:6.1f}s | Memory: {mem_mb_val:8.1f} MB | {mem_mb_val/1024:5.2f} GB")
        except (subprocess.CalledProcessError, ValueError):
            pass
        
        
        try:
            import select
            if select.select([proc.stdout], [], [], 0)[0]:
                line = proc.stdout.readline()
                if line:
                    output_lines.append(line.rstrip())
        except:
            pass
        
        if poll_result is not None:
            break
        
        time.sleep(0.3)
    
    
    remaining_output = proc.stdout.read()
    if remaining_output:
        output_lines.extend(remaining_output.strip().split('\n'))
    
except KeyboardInterrupt:
    proc.kill()
    print("\nMonitoring interrupted")
    sys.exit(1)

print("-" * 70)
print("\nInference output:")
print("-" * 70)
for line in output_lines:
    if line.strip():
        print(line)
print("-" * 70)


if len(timestamps) > 0 and len(memory_mb) > 0:
    fig, ax = plt.subplots(figsize=(14, 8))
    
    
    ax.plot(timestamps, memory_mb, linewidth=2.5, color='#2E86AB', label='Memory Usage')
    ax.fill_between(timestamps, memory_mb, alpha=0.3, color='#2E86AB')
    
   
    max_mem = max(memory_mb)
    max_mem_idx = memory_mb.index(max_mem)
    max_time = timestamps[max_mem_idx]
    
    final_mem = sum(memory_mb[-min(5, len(memory_mb)):]) / min(5, len(memory_mb))
    
   
    ax.axhline(y=max_mem, color='red', linestyle='--', alpha=0.6, linewidth=1.5, label=f'Peak: {max_mem:.0f} MB')
    ax.axhline(y=final_mem, color='green', linestyle='--', alpha=0.6, linewidth=1.5, label=f'Final: {final_mem:.0f} MB')
    
    
    ax.scatter([max_time], [max_mem], color='red', s=150, zorder=5, edgecolors='darkred', linewidth=2)
    ax.annotate(f'Peak Memory\n{max_mem:.0f} MB ({max_mem/1024:.2f} GB)', 
                xy=(max_time, max_mem), 
                xytext=(max_time + max(timestamps)*0.1, max_mem + 400),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
   
    ax.set_xlabel('Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Memory Usage (MB)', fontsize=13, fontweight='bold')
    ax.set_title('PaliGemma 3B Model Loading - Memory Usage Over Time\n' + 
                 'Optimized with Float16 Precision & Direct Weight Loading', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    
    stats_text = (
        f'Memory Statistics\n'
        f'{"─" * 30}\n'
        f'Peak Memory:     {max_mem:7.0f} MB ({max_mem/1024:.2f} GB)\n'
        f'Final Memory:    {final_mem:7.0f} MB ({final_mem/1024:.2f} GB)\n'
        f'Memory Saved:    {max_mem - final_mem:7.0f} MB ({(max_mem - final_mem)/1024:.2f} GB)\n'
        f'Reduction:       {((max_mem - final_mem) / max_mem * 100):7.1f}%\n'
        f'{"─" * 30}\n'
        f'Total Time:      {max(timestamps):.1f} seconds'
    )
    
    ax.text(0.98, 0.02, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            family='monospace',
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.8, edgecolor='navy', linewidth=2))
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0, top=max_mem * 1.15)
    
    plt.tight_layout()
    plt.savefig('memory_usage_graph.png', dpi=300, bbox_inches='tight')
    print(f"\nGraph saved as 'memory_usage_graph.png'")
    
    print(f"\nMemory Statistics:")
    print(f"{'─' * 70}")
    print(f"  Peak Memory:      {max_mem:8.1f} MB  (~{max_mem/1024:.2f} GB)")
    print(f"  Final Memory:     {final_mem:8.1f} MB  (~{final_mem/1024:.2f} GB)")
    print(f"  Memory Saved:     {max_mem - final_mem:8.1f} MB  (~{(max_mem - final_mem)/1024:.2f} GB)")
    print(f"  Reduction:        {((max_mem - final_mem) / max_mem * 100):8.1f}%")
    print(f"  Total Time:       {max(timestamps):8.1f} seconds")
    print(f"{'─' * 70}")
else:
    print("No memory data collected")

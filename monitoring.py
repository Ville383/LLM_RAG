import threading
import time
import psutil
from pynvml import *
import matplotlib.pyplot as plt
from pathlib import Path

class HardwareMonitor(threading.Thread):
    def __init__(self, interval: float = 0.1):
        super().__init__()
        self.interval = interval
        self.stop_event = threading.Event()
        self.samples = []

        # Initialize NVML for NVIDIA monitoring
        try:
            nvmlInit()
            self.gpu_available = True
            self.handle = nvmlDeviceGetHandleByIndex(0) # Assumes the first GPU (index 0)
            self.gpu_name = nvmlDeviceGetName(self.handle)
        except NVMLError as e:
            print(f"Warning: NVIDIA GPU monitoring not available: {e}")
            self.gpu_available = False

    def run(self):
        while not self.stop_event.is_set():  
            sample = {
                'ram_gb': psutil.virtual_memory().used / 1024**3
            }

            if self.gpu_available:
                info = nvmlDeviceGetMemoryInfo(self.handle)
                util = nvmlDeviceGetUtilizationRates(self.handle)
                temp = nvmlDeviceGetTemperature(self.handle, NVML_TEMPERATURE_GPU)
                # Add wattage
                
                sample['vram_gb'] = info.used / 1024**3
                sample['gpu_load'] = util.gpu  # This is a percentage 0-100
                sample['gpu_temp'] = temp
            
            self.samples.append(sample)
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()
        if self.is_alive():
            self.join()
        if self.gpu_available:
            try:
                nvmlShutdown()
            except:
                pass

    def get_stats(self):
        if not self.samples:
            return {}
        
        ram_history = [s['ram_gb'] for s in self.samples]
        stats = {
            'peak_ram_gb': max(ram_history),
            'ram_history': ram_history
        }
        
        if self.gpu_available:
            vram_history = [s['vram_gb'] for s in self.samples]
            gpu_load_history = [s['gpu_load'] for s in self.samples]
            gpu_temp_history = [s['gpu_temp'] for s in self.samples]
            stats.update({
                'peak_vram_gb': max(vram_history),
                'peak_gpu_temp': max(gpu_temp_history),
                'peak_gpu_load': max(gpu_load_history),
                'vram_history': vram_history,
                'gpu_load_history': gpu_load_history,
                'gpu_temp_history': gpu_temp_history
            })
            
        return stats

    def save_plot(self, execution_provider: str):
        """Generates and saves the hardware profile with four subplots."""
        stats = self.get_stats()
        ram_history = stats.get('ram_history', [])
        
        if not ram_history:
            print("No data captured to plot.")
            return

        time_axis = [i * self.interval for i in range(len(self.samples))]
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 14))
        ax1.set_title(f"Execution provider: {execution_provider}")
        
        # --- RAM ---
        ax1.plot(time_axis, ram_history, label='System RAM', color='#1f77b4', linewidth=1.5)
        ax1.set_xlabel("Seconds (s)")
        ax1.set_ylabel("RAM (GB)")
        ax1.set_xlim([0, time_axis[-1]])
        ax1.grid(True, linestyle='--', alpha=0.5)

        # --- GPU ---
        if self.gpu_available:
            ax2.plot(time_axis, stats['vram_history'], label='GPU VRAM', color='#2ca02c', linewidth=1.5)
            ax2.set_xlabel("Seconds (s)")
            ax2.set_ylabel("VRAM (GB)")
            ax2.set_xlim([0, time_axis[-1]]) 
            ax2.grid(True, linestyle='--', alpha=0.5)

            ax3.plot(time_axis, stats['gpu_load_history'], label='GPU Load (%)', color='#ff7f0e', linewidth=1.2)
            ax3.fill_between(time_axis, stats['gpu_load_history'], color='#ff7f0e', alpha=0.1)
            ax3.set_xlabel("Seconds (s)")
            ax3.set_ylabel("GPU Load (%)")
            ax3.set_xlim([0, time_axis[-1]])
            ax3.set_ylim([0, 105])
            ax3.grid(True, linestyle='--', alpha=0.5)

            ax4.plot(time_axis, stats['gpu_temp_history'], label='GPU Temp (°C)', color='#d62728', linewidth=1.5)
            ax4.set_ylabel("GPU Temp (°C)")
            ax4.set_xlim([0, time_axis[-1]])
            ax4.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        save_file = Path("results", f"hardware_profile_{execution_provider}.pdf")
        save_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_file)
        plt.close()
        print(f"Hardware profile saved to: {save_file}")
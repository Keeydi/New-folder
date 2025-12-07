import spidev
import time
import lgpio
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

# ---------------- CONFIG ----------------
VREF = 3.3            # ADC reference voltage
SPI_BUS = 0
SPI_DEVICE = 0        # CE0
SPI_SPEED_HZ = 5000

DRDY_PIN = 25         # GPIO connected to DRDY (Pin 11)
MAX_POINTS = 500      # Number of points shown in live plot
PLOT_UPDATE = 5       # Update plot every 5 samples
SMOOTH_WINDOW = 5     # Number of samples for moving average
BASELINE_SAMPLES = 100  # Samples to collect for baseline calibration
BASELINE_SAMPLES = 200  # Samples to collect for baseline calibration (increased for better accuracy)
NOISE_THRESHOLD = 0.001  # Ignore signals below this voltage (1mV) - adjust as needed
MEDIAN_FILTER_SIZE = 5  # Size of median filter for noise rejection

# ADS1256 commands
CMD_RESET  = 0xFE
@@ -71,7 +73,8 @@
    # Configure DRATE register (data rate)
    # 0xF0 = 30,000 SPS, 0xE0 = 15,000 SPS, 0xD0 = 7,500 SPS
    # 0xC0 = 3,750 SPS, 0xB0 = 2,000 SPS, 0xA0 = 1,000 SPS
    write_register(REG_DRATE, 0xF0)  # 30,000 SPS for impact echo
    # Using slower rate (15,000 SPS) for better stability and less noise
    write_register(REG_DRATE, 0xE0)  # 15,000 SPS - good balance for impact echo

    # Configure IO register
    write_register(REG_IO, 0x00)  # All GPIO as inputs
@@ -97,8 +100,18 @@
    write_register(REG_MUX, mux_value)
    time.sleep(0.01)  # Allow MUX to settle

def wait_for_drdy(timeout=0.1):
    """Wait for DRDY pin to go low (data ready)"""
    start_time = time.time()
    while lgpio.gpio_read(h, DRDY_PIN) == 1:
        if time.time() - start_time > timeout:
            return False  # Timeout
        time.sleep(0.0001)  # Small delay to avoid busy waiting
    return True  # DRDY is low

def read_adc():
    """Read 24-bit value from ADS1256"""
    """Read 24-bit value from ADS1256 (assumes DRDY already checked)"""
    # Send read command
    spi.xfer2([CMD_RDATA])
    time.sleep(0.0005)  # Small delay for data ready
    raw = spi.readbytes(3)
@@ -122,32 +135,49 @@
    else:
        return sum(list(data_deque)[-window_size:])/window_size

def median_filter(data_list, window_size):
    """Apply median filter to reduce noise spikes"""
    if len(data_list) < window_size:
        return data_list[-1] if data_list else 0.0
    else:
        window = list(data_list)[-window_size:]
        return np.median(window)

def is_valid_signal(voltage, baseline, threshold):
    """Check if signal is above noise threshold"""
    return abs(voltage - baseline) >= threshold

def calibrate_baseline(num_samples=BASELINE_SAMPLES):
    """Calibrate baseline to remove DC offset and floating data"""
    print(f"Calibrating baseline with {num_samples} samples...")
    print("Please ensure device is stationary (no impacts) during calibration...")
    time.sleep(1)  # Give user time to read
    
    baseline_readings = []

    for i in range(num_samples):
        # Wait for DRDY
        while lgpio.gpio_read(h, DRDY_PIN) == 1:
            pass
        
        raw_val = read_adc()
        voltage = adc_to_voltage(raw_val)
        baseline_readings.append(voltage)
        if wait_for_drdy():
            raw_val = read_adc()
            voltage = adc_to_voltage(raw_val)
            baseline_readings.append(voltage)
        else:
            print(f"  Warning: DRDY timeout at sample {i+1}")

        if (i + 1) % 20 == 0:
        if (i + 1) % 50 == 0:
            print(f"  Collected {i + 1}/{num_samples} samples...")

    baseline = np.mean(baseline_readings)
    std_dev = np.std(baseline_readings)
    noise_level = std_dev * 2  # 2-sigma noise level

    print(f"Baseline calibration complete:")
    print(f"  Mean: {baseline:.6f} V")
    print(f"  Std Dev: {std_dev:.6f} V")
    print(f"  Estimated noise level: ±{noise_level:.6f} V")
    print(f"  Offset will be subtracted from all readings")
    print(f"  Signals below {NOISE_THRESHOLD:.6f} V will be ignored\n")

    return baseline
    return baseline, noise_level

# ---------------- INITIALIZE ----------------
print("Initializing ADS1256 ADC...")
@@ -157,9 +187,9 @@
time.sleep(0.3)  # Allow ADC to stabilize after configuration

# Calibrate baseline to fix floating data issue
baseline_offset = calibrate_baseline()
print(f"\nBaseline offset: {baseline_offset:.6f} V")
print("Starting data acquisition with baseline correction...\n")
baseline_offset, noise_level = calibrate_baseline()
print(f"Baseline offset: {baseline_offset:.6f} V")
print("Starting data acquisition with noise filtering...\n")

# ---------------- SETUP LIVE PLOT ----------------
plt.ion()
@@ -177,40 +207,69 @@

# ---------------- MAIN LOOP ----------------
sample_count = 0
raw_data_buffer = deque(maxlen=MEDIAN_FILTER_SIZE)  # Buffer for median filter
quiet_samples = 0  # Count of samples below threshold

try:
    while True:
        # Wait for DRDY low
        while lgpio.gpio_read(h, DRDY_PIN) == 1:
            pass

        # Wait for DRDY before reading
        if not wait_for_drdy():
            print("Warning: DRDY timeout, skipping sample")
            continue
            
        raw_val = read_adc()
        voltage = adc_to_voltage(raw_val)

        # Remove baseline offset to fix floating data
        voltage_corrected = voltage - baseline_offset
        
        # Add to median filter buffer
        raw_data_buffer.append(voltage_corrected)
        
        # Apply median filter to remove noise spikes
        if len(raw_data_buffer) >= MEDIAN_FILTER_SIZE:
            filtered_voltage = median_filter(raw_data_buffer, MEDIAN_FILTER_SIZE)
        else:
            filtered_voltage = voltage_corrected
        
        # Noise threshold filtering - only process signals above threshold
        if abs(filtered_voltage) < NOISE_THRESHOLD:
            # Signal is below threshold, treat as zero (quiet state)
            filtered_voltage = 0.0
            quiet_samples += 1
        else:
            quiet_samples = 0

        # Append corrected reading
        data.append(voltage_corrected)
        # Append filtered reading
        data.append(filtered_voltage)

        # Apply moving average for smoother plot
        smoothed_voltage = moving_average(data, SMOOTH_WINDOW)
        data[-1] = smoothed_voltage  # replace latest with smoothed value
        # Apply moving average for smoother plot (only if we have enough data)
        if len(data) >= SMOOTH_WINDOW:
            smoothed_voltage = moving_average(data, SMOOTH_WINDOW)
            data[-1] = smoothed_voltage  # replace latest with smoothed value

        sample_count += 1

        # Update plot every PLOT_UPDATE samples
        if sample_count % PLOT_UPDATE == 0:
            line.set_ydata(data)
            # Dynamic y-axis scaling with margin
            data_min, data_max = min(data), max(data)
            margin = max(abs(data_min), abs(data_max)) * 0.1 + 0.01
            ax.set_ylim(data_min - margin, data_max + margin)
            if len(data) > 0:
                data_min, data_max = min(data), max(data)
                # Only scale if there's actual signal variation
                if abs(data_max - data_min) > 0.0001:
                    margin = max(abs(data_min), abs(data_max)) * 0.1 + 0.01
                    ax.set_ylim(data_min - margin, data_max + margin)
                else:
                    # If all quiet, show small range around zero
                    ax.set_ylim(-NOISE_THRESHOLD * 2, NOISE_THRESHOLD * 2)
            fig.canvas.draw()
            fig.canvas.flush_events()

        # Optional: Print sample info periodically
        # Print sample info periodically
        if sample_count % 100 == 0:
            print(f"Sample {sample_count}: Raw={voltage:.6f}V, Corrected={voltage_corrected:.6f}V")
            status = "QUIET" if abs(filtered_voltage) < NOISE_THRESHOLD else "ACTIVE"
            print(f"Sample {sample_count}: Raw={voltage:.6f}V, Filtered={filtered_voltage:.6f}V [{status}]")

except KeyboardInterrupt:
    print("Live capture stopped by user")

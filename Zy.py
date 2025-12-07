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
BASELINE_SAMPLES = 200  # Samples to collect for baseline calibration (increased for better accuracy)
NOISE_THRESHOLD = 0.001  # Ignore signals below this voltage (1mV) - adjust as needed
MEDIAN_FILTER_SIZE = 5  # Size of median filter for noise rejection

# ADS1256 commands
CMD_RESET  = 0xFE
CMD_SYNC   = 0xFC
CMD_WAKEUP = 0x00
CMD_RDATA  = 0x01
CMD_WREG   = 0x50  # write register
CMD_RREG   = 0x10  # read register

# ADS1256 Register Addresses
REG_STATUS = 0x00
REG_MUX = 0x01
REG_ADCON = 0x02
REG_DRATE = 0x03
REG_IO = 0x04

# ---------------- SETUP SPI ----------------
spi = spidev.SpiDev()
spi.open(SPI_BUS, SPI_DEVICE)
spi.max_speed_hz = SPI_SPEED_HZ

# ---------------- SETUP DRDY ----------------
h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_input(h, DRDY_PIN)

# ---------------- FUNCTIONS ----------------
def adc_reset():
    spi.xfer2([CMD_RESET])
    time.sleep(0.1)
    spi.xfer2([CMD_WAKEUP])
    time.sleep(0.1)
    spi.xfer2([CMD_SYNC])
    time.sleep(0.1)

def write_register(reg, value):
    """Write to ADS1256 register"""
    spi.xfer2([CMD_WREG | reg, 0x00, value])
    time.sleep(0.01)

def read_register(reg):
    """Read from ADS1256 register"""
    # CMD_RREG format: [CMD_RREG | reg, number_of_bytes - 1]
    spi.xfer2([CMD_RREG | reg, 0x00])  # Read 1 byte (0x00 = 1-1)
    time.sleep(0.0001)
    return spi.readbytes(1)[0]

def configure_adc():
    """Configure ADS1256 for stable operation"""
    # Configure ADCON register
    # Bit 7: CLKOUT off, Bit 6-4: Sensor detect off, Bit 3-0: PGA gain = 1
    write_register(REG_ADCON, 0x00)  # PGA = 1, no sensor detect
    
    # Configure DRATE register (data rate)
    # 0xF0 = 30,000 SPS, 0xE0 = 15,000 SPS, 0xD0 = 7,500 SPS
    # 0xC0 = 3,750 SPS, 0xB0 = 2,000 SPS, 0xA0 = 1,000 SPS
    # Using slower rate (15,000 SPS) for better stability and less noise
    write_register(REG_DRATE, 0xE0)  # 15,000 SPS - good balance for impact echo
    
    # Configure IO register
    write_register(REG_IO, 0x00)  # All GPIO as inputs
    
    time.sleep(0.1)

def select_channel(channel=0):
    """Select ADC channel
    Channel 0: AIN0-AIN1 (differential)
    Channel 1: AIN2-AIN3 (differential)
    For single-ended with AINCOM: use (channel << 4) | 0x08
    """
    # For differential input: AIN0-AIN1 = 0x01, AIN2-AIN3 = 0x23, etc.
    # MUX format: [Positive Input (4 bits)] | Negative Input (4 bits)]
    if channel == 0:
        mux_value = 0x01  # AIN0 positive, AIN1 negative
    elif channel == 1:
        mux_value = 0x23  # AIN2 positive, AIN3 negative
    else:
        # For other channels, use single-ended with AINCOM
        mux_value = (channel << 4) | 0x08  # AINx positive, AINCOM negative
    
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
    """Read 24-bit value from ADS1256 (assumes DRDY already checked)"""
    # Send read command
    spi.xfer2([CMD_RDATA])
    time.sleep(0.0005)  # Small delay for data ready
    raw = spi.readbytes(3)
    if len(raw) != 3:
        raise ValueError(f"Expected 3 bytes, got {len(raw)}")
    value = (raw[0]<<16) | (raw[1]<<8) | raw[2]
    # Sign extend 24-bit to 32-bit
    if value & 0x800000:
        value -= 0x1000000
    return value

def adc_to_voltage(raw):
    """Convert 24-bit ADC reading to voltage"""
    # ADS1256 is 24-bit, full scale is ±0x7FFFFF
    return (raw / 0x7FFFFF) * VREF

def moving_average(data_deque, window_size):
    """Calculate moving average"""
    if len(data_deque) < window_size:
        return sum(data_deque)/len(data_deque)
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
        if wait_for_drdy():
            raw_val = read_adc()
            voltage = adc_to_voltage(raw_val)
            baseline_readings.append(voltage)
        else:
            print(f"  Warning: DRDY timeout at sample {i+1}")
        
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
    
    return baseline, noise_level

# ---------------- INITIALIZE ----------------
print("Initializing ADS1256 ADC...")
adc_reset()
configure_adc()
select_channel(0)
time.sleep(0.3)  # Allow ADC to stabilize after configuration

# Calibrate baseline to fix floating data issue
baseline_offset, noise_level = calibrate_baseline()
print(f"Baseline offset: {baseline_offset:.6f} V")
print("Starting data acquisition with noise filtering...\n")

# ---------------- SETUP LIVE PLOT ----------------
plt.ion()
fig, ax = plt.subplots()
data = deque([0.0]*MAX_POINTS, maxlen=MAX_POINTS)  # Initialize with zeros for baseline-corrected data
line, = ax.plot(data)
ax.set_ylim(-VREF/2, VREF/2)  # Adjusted for baseline-corrected data
ax.set_title("ADXL1002Z Live Vibration (Baseline Corrected)")
ax.set_xlabel("Sample")
ax.set_ylabel("Voltage (V) - Baseline Corrected")
ax.axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Zero Reference')
ax.legend()

print("Starting live vibration capture. Press Ctrl+C to stop.")

# ---------------- MAIN LOOP ----------------
sample_count = 0
raw_data_buffer = deque(maxlen=MEDIAN_FILTER_SIZE)  # Buffer for median filter
quiet_samples = 0  # Count of samples below threshold

try:
    while True:
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

        # Append filtered reading
        data.append(filtered_voltage)

        # Apply moving average for smoother plot (only if we have enough data)
        if len(data) >= SMOOTH_WINDOW:
            smoothed_voltage = moving_average(data, SMOOTH_WINDOW)
            data[-1] = smoothed_voltage  # replace latest with smoothed value

        sample_count += 1

        # Update plot every PLOT_UPDATE samples
        if sample_count % PLOT_UPDATE == 0:
            line.set_ydata(data)
            # Dynamic y-axis scaling with margin
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
            
        # Print sample info periodically
        if sample_count % 100 == 0:
            status = "QUIET" if abs(filtered_voltage) < NOISE_THRESHOLD else "ACTIVE"
            print(f"Sample {sample_count}: Raw={voltage:.6f}V, Filtered={filtered_voltage:.6f}V [{status}]")

except KeyboardInterrupt:
    print("Live capture stopped by user")

finally:
    spi.close()
    lgpio.gpiochip_close(h)
    plt.ioff()
    plt.show()
    print("Resources closed")
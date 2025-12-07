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
SMOOTH_WINDOW = 1     # Disable smoothing to preserve vibration details
BASELINE_SAMPLES = 200  # Samples for baseline calibration
NOISE_THRESHOLD = 0.0001  # Lower threshold for vibration detection (0.1mV)
HPF_ALPHA = 0.9995     # Much stronger high-pass filter (0.9995 = very aggressive DC removal)
DC_BLOCKER_ALPHA = 0.9999  # Running DC blocker coefficient (removes slow changes)
DC_BLOCKER_WINDOW = 500  # Window size for running average DC blocker
VIBRATION_THRESHOLD = 0.0005  # Threshold to detect actual vibration (0.5mV)

# ADS1256 commands
CMD_RESET  = 0xFE
CMD_SYNC   = 0xFC
CMD_WAKEUP = 0x00
CMD_RDATA  = 0x01
CMD_WREG   = 0x50  # write register
CMD_RREG   = 0x10  # read register

# ADS1256 registers
REG_STATUS = 0x00
REG_MUX    = 0x01
REG_ADCON  = 0x02
REG_DRATE  = 0x03
REG_IO     = 0x04

# ---------------- SETUP SPI ----------------
spi = spidev.SpiDev()
spi.open(SPI_BUS, SPI_DEVICE)
spi.max_speed_hz = SPI_SPEED_HZ

# ---------------- SETUP DRDY ----------------
h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_input(h, DRDY_PIN)

# ---------------- FUNCTIONS ----------------
def write_register(reg, value):
    """Write to ADS1256 register"""
    spi.xfer2([CMD_WREG | reg, 0x00, value])
    time.sleep(0.0001)

def read_register(reg):
    """Read from ADS1256 register"""
    spi.xfer2([CMD_RREG | reg, 0x00])
    time.sleep(0.0001)
    return spi.readbytes(1)[0]

def adc_reset():
    """Reset ADS1256"""
    spi.xfer2([CMD_RESET])
    time.sleep(0.1)
    spi.xfer2([CMD_WAKEUP])
    time.sleep(0.1)
    spi.xfer2([CMD_SYNC])
    time.sleep(0.1)

def configure_adc():
    """Properly configure ADS1256 to prevent floating data"""
    print("Configuring ADS1256 registers...")
    
    # Configure STATUS register
    # Bit 7: Reset (0), Bit 6-4: Reserved, Bit 3: ACAL (0=off), Bit 2: BUFEN (1=enable buffer)
    # Bit 1: Reserved, Bit 0: ORDER (0=MSB first)
    write_register(REG_STATUS, 0x04)  # Enable buffer, MSB first
    
    # Configure MUX register for differential input AIN0-AIN1
    # This prevents floating inputs - use differential mode for accelerometer
    # AINP=AIN0 (bits 7-4), AINN=AIN1 (bits 3-0)
    write_register(REG_MUX, 0x01)  # AIN0-AIN1 differential
    time.sleep(0.01)
    
    # Configure ADCON register
    # Bit 7-5: Reserved, Bit 4-2: CLK (000=oscillator), Bit 1-0: Sensor Detect (00=off)
    write_register(REG_ADCON, 0x20)  # Default settings
    
    # Configure DRATE register (data rate)
    # 0xF0 = 30,000 SPS, 0xE0 = 15,000 SPS, 0xD0 = 7,500 SPS
    # Higher rate for better vibration capture
    write_register(REG_DRATE, 0xF0)  # 30,000 SPS for impact echo vibrations
    
    # Configure IO register - set unused pins to avoid floating
    write_register(REG_IO, 0x00)  # All GPIO as inputs
    
    print("ADC configuration complete")

def select_channel(ainp=0, ainn=1):
    """Select differential channel (AINP positive, AINN negative)"""
    mux_value = (ainp << 4) | ainn
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
    spi.xfer2([CMD_RDATA])
    time.sleep(0.0005)
    raw = spi.readbytes(3)
    value = (raw[0]<<16) | (raw[1]<<8) | raw[2]
    if value & 0x800000:
        value -= 0x1000000
    return value

def adc_to_voltage(raw):
    """Convert 24-bit ADC value to voltage"""
    return (raw / 0x7FFFFF) * VREF

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
    print(f"  Offset will be subtracted from all readings\n")

    return baseline

def moving_average(data_deque, window_size):
    if len(data_deque) < window_size:
        return sum(data_deque)/len(data_deque)
    else:
        return sum(list(data_deque)[-window_size:])/window_size

def high_pass_filter(new_value, prev_output, prev_input, alpha):
    """
    Digital high-pass filter to remove DC offset and slow changes (tilt)
    while preserving high-frequency vibrations
    alpha: filter coefficient (0.99-0.999 typical, higher = stronger DC removal)
    """
    output = alpha * (prev_output + new_value - prev_input)
    return output, new_value

class DCBlocker:
    """Running DC blocker - continuously tracks and removes slow-moving average"""
    def __init__(self, alpha=0.9999, window_size=500):
        self.alpha = alpha
        self.window_size = window_size
        self.running_avg = 0.0
        self.buffer = deque(maxlen=window_size)
        
    def process(self, value):
        """Remove DC component using running average"""
        # Add to buffer
        self.buffer.append(value)
        
        # Calculate running average (exponential moving average for speed)
        if len(self.buffer) == 1:
            self.running_avg = value
        else:
            # Use exponential moving average
            self.running_avg = self.alpha * self.running_avg + (1 - self.alpha) * value
        
        # Subtract DC component
        return value - self.running_avg

# ---------------- INITIALIZE ----------------
print("Initializing ADS1256 ADC...")
adc_reset()
configure_adc()
select_channel(0, 1)  # Use differential input AIN0-AIN1
time.sleep(0.3)  # Allow ADC to stabilize after configuration

# Calibrate baseline to fix floating data issue
baseline_offset = calibrate_baseline()
print(f"Baseline offset: {baseline_offset:.6f} V")
print("Starting data acquisition with AC coupling (vibration only, tilt filtered)...\n")

# Initialize high-pass filter and DC blocker
print("Initializing filters (high-pass + DC blocker)...")
hpf_init_output = 0.0
hpf_init_input = 0.0
dc_blocker = DCBlocker(alpha=DC_BLOCKER_ALPHA, window_size=DC_BLOCKER_WINDOW)

# Stabilize both filters with initial samples
for _ in range(200):  # More samples for better stabilization
    if wait_for_drdy():
        raw_val = read_adc()
        voltage = adc_to_voltage(raw_val)
        voltage_dc = voltage - baseline_offset
        
        # Apply high-pass filter
        hpf_init_output, hpf_init_input = high_pass_filter(voltage_dc, hpf_init_output, hpf_init_input, HPF_ALPHA)
        
        # Apply DC blocker
        _ = dc_blocker.process(hpf_init_output)
        
print("Filters initialized and stabilized.\n")

# ---------------- SETUP LIVE PLOT ----------------
plt.ion()
fig, ax = plt.subplots()
data = deque([0.0]*MAX_POINTS, maxlen=MAX_POINTS)  # Start at 0 after baseline correction
line, = ax.plot(data)
ax.set_ylim(-0.01, 0.01)  # Tighter range for vibration-only signals
ax.set_title("ADXL1002Z Live Vibration (AC-Coupled, Tilt Filtered)")
ax.set_xlabel("Sample")
ax.set_ylabel("Voltage (V)")
ax.grid(True)

print("Starting live vibration capture. Press Ctrl+C to stop.")
print("Note: System will ignore slow tilt changes, only capture vibrations.\n")

# ---------------- MAIN LOOP ----------------
sample_count = 0
hpf_prev_output = hpf_init_output  # High-pass filter previous output
hpf_prev_input = hpf_init_input    # High-pass filter previous input
vibration_detected = False

try:
    while True:
        # Wait for DRDY before reading
        if not wait_for_drdy():
            print("Warning: DRDY timeout, skipping sample")
            continue
            
        raw_val = read_adc()
        voltage = adc_to_voltage(raw_val)

        # Remove static baseline offset
        voltage_dc_removed = voltage - baseline_offset
        
        # Step 1: Apply aggressive high-pass filter to remove DC and slow changes (tilt)
        voltage_hpf, hpf_prev_input = high_pass_filter(
            voltage_dc_removed, hpf_prev_output, hpf_prev_input, HPF_ALPHA
        )
        hpf_prev_output = voltage_hpf
        
        # Step 2: Apply DC blocker to remove any remaining slow-moving average
        # This double-filtering ensures tilt is completely removed
        voltage_filtered = dc_blocker.process(voltage_hpf)
        
        # Detect vibration (high-frequency signal)
        if abs(voltage_filtered) > VIBRATION_THRESHOLD:
            vibration_detected = True
        else:
            vibration_detected = False

        # Append filtered reading (vibration only, no tilt)
        # No smoothing - preserve all vibration details
        data.append(voltage_filtered)

        sample_count += 1

        # Update plot every PLOT_UPDATE samples
        if sample_count % PLOT_UPDATE == 0:
            line.set_ydata(data)
            if len(data) > 0:
                data_min, data_max = min(data), max(data)
                # Only scale if there's actual signal variation
                if abs(data_max - data_min) > 0.0001:
                    margin = max(abs(data_min), abs(data_max)) * 0.2 + 0.005
                    ax.set_ylim(data_min - margin, data_max + margin)
                else:
                    # If all quiet, show small range around zero
                    ax.set_ylim(-VIBRATION_THRESHOLD * 3, VIBRATION_THRESHOLD * 3)
            fig.canvas.draw()
            fig.canvas.flush_events()
        
        # Print sample info periodically
        if sample_count % 100 == 0:
            status = "VIBRATION!" if vibration_detected else "quiet"
            print(f"Sample {sample_count}: Raw={voltage:.6f}V, Filtered={voltage_filtered:.6f}V [{status}]")

except KeyboardInterrupt:
    print("Live capture stopped by user")

finally:
    spi.close()
    lgpio.gpiochip_close(h)
    plt.ioff()
    plt.show()
    print("Resources closed")
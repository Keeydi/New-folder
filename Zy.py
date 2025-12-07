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
SMOOTH_WINDOW = 5     # Number of samples for moving average (DISABLED for impact echo)
BASELINE_SAMPLES = 200  # Samples for baseline calibration (to remove DC/tilt)
HPF_ALPHA = 0.90      # High-pass filter coefficient (0.90 = ~20Hz cutoff - very strong filtering)
                      # Lower = stronger filtering (removes more DC/tilt)
                      # For impact echo: 0.85-0.95 range works well
USE_SMOOTHING = False  # DISABLE smoothing for impact echo - need raw vibration signals
IMPACT_THRESHOLD = 0.01  # Voltage threshold to detect impact events

# ADS1256 commands
CMD_RESET  = 0xFE
CMD_SYNC   = 0xFC
CMD_WAKEUP = 0x00
CMD_RDATA  = 0x01
CMD_WREG   = 0x50  # write register

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

def select_channel(channel=0):
    spi.xfer2([CMD_WREG | 0x01, 0x00, channel << 4])
    time.sleep(0.01)

def read_adc():
    spi.xfer2([CMD_RDATA])
    time.sleep(0.0005)
    raw = spi.readbytes(3)
    value = (raw[0]<<16) | (raw[1]<<8) | raw[2]
    if value & 0x800000:
        value -= 0x1000000
    return value

def adc_to_voltage(raw):
    return (raw / 0x7FFFFF) * VREF

def moving_average(data_deque, window_size):
    if len(data_deque) < window_size:
        return sum(data_deque)/len(data_deque)
    else:
        return sum(list(data_deque)[-window_size:])/window_size

def calibrate_baseline(num_samples=BASELINE_SAMPLES):
    """
    Calibrate ORIGINAL POSITION of accelerometer.
    As developer said: "yung kinukuha dapat natin yung layo nung galaw vs sa original position"
    (We need to get the distance of movement vs the original position)
    
    This measures the original position (includes tilt/gravity) so we can calculate
    the distance/change from this position when vibration occurs.
    """
    print(f"\n=== CALIBRATING ORIGINAL POSITION ===")
    print(f"Collecting {num_samples} samples to measure original position...")
    print("IMPORTANT: Keep device STATIONARY (no impacts) during calibration!")
    print("This establishes the 'original position' reference point.")
    print("All future readings will show distance/movement from this position.\n")
    time.sleep(2)  # Give user time to read
    
    baseline_readings = []
    
    for i in range(num_samples):
        # Wait for DRDY
        while lgpio.gpio_read(h, DRDY_PIN) == 1:
            pass
        
        raw_val = read_adc()
        voltage = adc_to_voltage(raw_val)
        baseline_readings.append(voltage)
        
        if (i + 1) % 50 == 0:
            print(f"  Collected {i + 1}/{num_samples} samples...")
    
    baseline = np.mean(baseline_readings)
    std_dev = np.std(baseline_readings)
    
    print(f"\nOriginal position calibration complete:")
    print(f"  Original Position (includes tilt/gravity): {baseline:.6f} V")
    print(f"  Noise level: ±{std_dev:.6f} V")
    print(f"  All readings will show: (current_position - original_position)")
    print(f"  This gives us the 'distance of movement' from original position\n")
    
    return baseline

class HighPassFilter:
    """
    Strong high-pass filter to remove DC drift and low-frequency tilt.
    For impact echo, we need to see oscillatory waves, not DC/tilt.
    Lower alpha = stronger filtering (removes more DC).
    """
    def __init__(self, alpha=HPF_ALPHA):
        self.alpha = alpha  # Filter coefficient (0.90 = strong filtering, removes DC/tilt)
        self.prev_output = 0.0
        self.prev_input = 0.0
        self.initialized = False
        self.warmup_samples = 50  # Warmup period to stabilize filter
    
    def filter(self, input_value):
        if not self.initialized:
            # Initialize with first value
            self.prev_input = input_value
            self.prev_output = 0.0
            self.initialized = True
            self.warmup_count = 0
            return 0.0
        
        # Warmup period - filter needs time to stabilize
        if self.warmup_count < self.warmup_samples:
            self.warmup_count += 1
            # High-pass filter: y[n] = alpha * (y[n-1] + x[n] - x[n-1])
            output = self.alpha * (self.prev_output + input_value - self.prev_input)
            self.prev_input = input_value
            self.prev_output = output
            return 0.0  # Return zero during warmup
        
        # High-pass filter: y[n] = alpha * (y[n-1] + x[n] - x[n-1])
        output = self.alpha * (self.prev_output + input_value - self.prev_input)
        self.prev_input = input_value
        self.prev_output = output
        return output

# ---------------- INITIALIZE ----------------
print("Initializing ADS1256 ADC...")
adc_reset()
select_channel(0)
time.sleep(0.1)

# Calibrate baseline to remove DC offset (tilt/gravity component)
# This fixes the "floating data" problem by removing static offset
baseline_offset = calibrate_baseline()

# Initialize high-pass filter to remove any remaining DC drift
hpf = HighPassFilter(alpha=HPF_ALPHA)

# ---------------- SETUP LIVE PLOT ----------------
plt.ion()
fig, ax = plt.subplots()
data = deque([0.0]*MAX_POINTS, maxlen=MAX_POINTS)  # Start at 0 (no DC offset)
line, = ax.plot(data)
ax.set_ylim(-0.1, 0.1)  # Show around zero for vibration signals
ax.set_title("ADXL1002Z - Raw Vibration Data (Distance from Original Position)")
ax.set_xlabel("Sample")
ax.set_ylabel("Voltage (V) - Movement from Original Position")
ax.axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Zero (baseline)')
ax.grid(True, alpha=0.3)
ax.legend()

print("\n=== RAW DATA COLLECTION MODE ===")
print("Measuring: Distance of movement vs original position")
print("High-pass filter active - removing slow tilt changes")
print("Showing RAW vibration data (no smoothing, no FFT yet)")
print("")
print("Purpose: Verify system is working correctly")
print("Hit with steel ball hammer - observe raw vibration response")
print("FFT analysis for defect detection will be added later")
print("Press Ctrl+C to stop.\n")

# ---------------- MAIN LOOP ----------------
sample_count = 0
try:
    while True:
        # Wait for DRDY low
        while lgpio.gpio_read(h, DRDY_PIN) == 1:
            pass

        raw_val = read_adc()
        voltage = adc_to_voltage(raw_val)

        # STEP 1: Calculate distance from original position
        # Developer: "yung kinukuha dapat natin yung layo nung galaw vs sa original position"
        # (We need to get the distance of movement vs the original position)
        # This measures how far the accelerometer has moved from its original position
        movement_from_original = voltage - baseline_offset

        # STEP 2: Apply high-pass filter to remove slow tilt changes
        # This ensures we only see dynamic vibration, not slow tilt drift
        # The filter removes any remaining DC/low-frequency components
        vibration_signal = hpf.filter(movement_from_original)

        # STEP 3: Store RAW vibration data (no smoothing)
        # Developer: "for raw data palang, di pa nalalagyan nung FFT"
        # (Just for raw data, FFT not added yet)
        # We need clean raw data first to verify system is working
        # NO smoothing - we need raw signals for later FFT analysis
        data.append(vibration_signal)  # Store raw vibration signal

        sample_count += 1

        # Update plot every PLOT_UPDATE samples
        if sample_count % PLOT_UPDATE == 0:
            line.set_ydata(data)
            # Dynamic scaling around zero for vibration signals
            if len(data) > 0:
                data_max = max(abs(min(data)), abs(max(data)))
                if data_max > 0.0001:  # Only scale if there's signal
                    margin = data_max * 0.3 + 0.005  # Slightly larger margin for echoes
                    ax.set_ylim(-margin, margin)
                else:
                    # If quiet, show small range
                    ax.set_ylim(-0.01, 0.01)
            fig.canvas.draw()
            fig.canvas.flush_events()

        # Detect impact events and print status
        if sample_count % 500 == 0:
            # Check if we're seeing vibration (not just noise)
            recent_data = list(data)[-100:] if len(data) >= 100 else list(data)
            if recent_data:
                signal_range = max(recent_data) - min(recent_data)
                if signal_range > IMPACT_THRESHOLD:
                    status = "IMPACT DETECTED - Look for oscillatory echoes!"
                else:
                    status = "Waiting for impact..."
            else:
                status = "Initializing..."
            
            print(f"Sample {sample_count}: Original={voltage:.6f}V, "
                  f"Movement={movement_from_original:.6f}V, "
                  f"Vibration={vibration_signal:.6f}V [{status}]")

except KeyboardInterrupt:
    print("Live capture stopped by user")

finally:
    spi.close()
    lgpio.gpiochip_close(h)
    plt.ioff()
    plt.show()
    print("Resources closed")
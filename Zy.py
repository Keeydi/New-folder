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
BASELINE_SAMPLES = 200  # Samples for baseline calibration (to remove DC/tilt)
HPF_ALPHA = 0.99      # High-pass filter coefficient (0.99 = ~1Hz cutoff at 15kSPS)

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
    Calibrate baseline to remove DC offset (tilt/gravity component).
    This removes the 'floating' data and isolates vibration signals.
    """
    print(f"\n=== BASELINE CALIBRATION ===")
    print(f"Collecting {num_samples} samples to measure DC offset...")
    print("IMPORTANT: Keep device STATIONARY (no impacts) during calibration!")
    print("This will remove tilt/gravity component and show only vibration.\n")
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
    
    print(f"\nBaseline calibration complete:")
    print(f"  DC Offset (tilt/gravity): {baseline:.6f} V")
    print(f"  Noise level: ±{std_dev:.6f} V")
    print(f"  This offset will be subtracted to show only vibration signals\n")
    
    return baseline

class HighPassFilter:
    """
    Simple high-pass filter to remove DC drift and low-frequency components.
    This ensures we only see AC vibration signals, not slow tilt changes.
    """
    def __init__(self, alpha=HPF_ALPHA):
        self.alpha = alpha  # Filter coefficient
        self.prev_output = 0.0
        self.prev_input = 0.0
        self.initialized = False
    
    def filter(self, input_value):
        if not self.initialized:
            # Initialize with first value
            self.prev_input = input_value
            self.prev_output = 0.0
            self.initialized = True
            return 0.0
        
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
ax.set_title("ADXL1002Z Live Vibration (AC Component Only)")
ax.set_xlabel("Sample")
ax.set_ylabel("Voltage (V) - Vibration Signal")
ax.axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Zero (baseline)')
ax.legend()

print("\n=== STARTING VIBRATION CAPTURE ===")
print("DC offset removed - showing only vibration signals (AC component)")
print("Hit with steel ball hammer to see impact echo response!")
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

        # STEP 1: Remove DC baseline offset (tilt/gravity component)
        # This removes the "floating" static offset
        voltage_ac = voltage - baseline_offset

        # STEP 2: Apply high-pass filter to remove any remaining DC drift
        # This ensures we only see dynamic vibration, not slow tilt changes
        vibration_signal = hpf.filter(voltage_ac)

        # Append vibration signal (AC component only)
        data.append(vibration_signal)

        # Apply moving average for smoother plot (optional)
        if len(data) >= SMOOTH_WINDOW:
            smoothed_voltage = moving_average(data, SMOOTH_WINDOW)
            data[-1] = smoothed_voltage  # replace latest with smoothed value

        sample_count += 1

        # Update plot every PLOT_UPDATE samples
        if sample_count % PLOT_UPDATE == 0:
            line.set_ydata(data)
            # Dynamic scaling around zero for vibration signals
            if len(data) > 0:
                data_max = max(abs(min(data)), abs(max(data)))
                if data_max > 0.0001:  # Only scale if there's signal
                    margin = data_max * 0.2 + 0.01
                    ax.set_ylim(-margin, margin)
                else:
                    # If quiet, show small range
                    ax.set_ylim(-0.01, 0.01)
            fig.canvas.draw()
            fig.canvas.flush_events()

        # Print status periodically
        if sample_count % 500 == 0:
            print(f"Sample {sample_count}: Raw={voltage:.6f}V, "
                  f"DC-removed={voltage_ac:.6f}V, "
                  f"Vibration={vibration_signal:.6f}V")

except KeyboardInterrupt:
    print("Live capture stopped by user")

finally:
    spi.close()
    lgpio.gpiochip_close(h)
    plt.ioff()
    plt.show()
    print("Resources closed")
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
HPF_ALPHA = 0.95      # High-pass filter coefficient (0.95 = less aggressive - preserves vibration)
                      # Lower = stronger filtering (removes more DC/tilt)
                      # 0.95 = good balance - removes slow tilt but keeps fast vibration
USE_SMOOTHING = False  # DISABLE smoothing for impact echo - need raw vibration signals
IMPACT_THRESHOLD = 0.01  # Voltage threshold to detect impact events
NOISE_THRESHOLD = 0.0005  # Very low threshold (0.5mV) - don't filter out real vibration
                         # Vibration from smashing should be much larger than this
TILT_FILTER_WINDOW = 100  # Larger window - only removes very slow tilt, preserves vibration
DEBUG_MODE = True  # Show raw signals before filtering for diagnosis

# Spike detection parameters
SPIKE_DETECTION_WINDOW = 50  # Samples to analyze for spike pattern
DIRECT_HIT_THRESHOLD = 0.05  # Very large amplitude = likely direct sensor hit (50mV)
OSCILLATION_MIN_CYCLES = 3  # Minimum oscillation cycles to be considered real vibration
QUIET_PERIOD_FOR_VALIDATION = 200  # Need quiet period before accepting spike as valid
MIN_OSCILLATION_AMPLITUDE = 0.002  # Minimum oscillation amplitude after spike (2mV)
FALSE_POSITIVE_THRESHOLD = 0.03  # If spike > this and no oscillation = likely false positive

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
    noise_level = std_dev * 3  # 3-sigma noise level
    
    print(f"\nOriginal position calibration complete:")
    print(f"  Original Position (includes tilt/gravity): {baseline:.6f} V")
    print(f"  Noise level: ±{std_dev:.6f} V")
    print(f"  Estimated noise threshold: ±{noise_level:.6f} V")
    print(f"  Signals below {NOISE_THRESHOLD:.6f} V will be filtered as noise")
    print(f"  All readings will show: (current_position - original_position)")
    print(f"  This gives us the 'distance of movement' from original position\n")
    
    return baseline, noise_level

class HighPassFilter:
    """
    High-pass filter to remove slow tilt while preserving fast vibration.
    For impact echo, we need to see oscillatory waves, not slow tilt.
    Lower alpha = stronger filtering (removes more DC/tilt).
    """
    def __init__(self, alpha=HPF_ALPHA):
        self.alpha = alpha  # Filter coefficient (0.95 = less aggressive, preserves vibration)
        self.prev_output = 0.0
        self.prev_input = 0.0
        self.initialized = False
        self.warmup_samples = 20  # Shorter warmup to respond faster
    
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

class TiltRemover:
    """
    Removes slow tilt by tracking slow-moving average and subtracting it.
    This preserves fast vibration while removing slow tilt movement.
    Uses exponential moving average for faster response to vibration.
    """
    def __init__(self, window_size=TILT_FILTER_WINDOW):
        self.window_size = window_size
        self.signal_history = deque(maxlen=window_size)
        self.initialized = False
        self.ema_alpha = 0.05  # Exponential moving average - very slow (only tracks tilt)
        self.ema_value = 0.0
    
    def remove_tilt(self, signal):
        self.signal_history.append(signal)
        
        if not self.initialized:
            if len(self.signal_history) < 10:  # Need minimum samples
                return signal  # Return as-is until we have enough samples
            # Initialize EMA with current average
            self.ema_value = np.mean(list(self.signal_history))
            self.initialized = True
        
        # Update exponential moving average (very slow - only tracks tilt)
        # EMA responds slowly, so fast vibration passes through
        self.ema_value = self.ema_alpha * signal + (1 - self.ema_alpha) * self.ema_value
        
        # Subtract slow tilt (EMA) to get vibration only
        vibration_only = signal - self.ema_value
        
        return vibration_only

class SpikeDetector:
    """
    Detects and validates spikes to distinguish:
    - Direct sensor hits: Very large, immediate, no oscillation pattern
    - Hammer strikes: Initial spike + oscillatory echoes (returning vibration)
    
    Based on images: We want to see initial spike followed by oscillatory waves
    """
    def __init__(self):
        self.signal_history = deque(maxlen=SPIKE_DETECTION_WINDOW)
        self.quiet_period_count = 0
        self.last_spike_sample = -9999
        self.spike_detected = False
        self.current_is_spike = False
        self.current_is_direct_hit = False
        self.current_has_oscillation = False
        
    def detect_oscillation(self, window_size=30):
        """
        Check if there's an oscillatory pattern after a spike.
        Real vibrations from hammer strikes show oscillatory waves.
        Direct sensor hits show no oscillation.
        """
        if len(self.signal_history) < window_size:
            return False, 0
        
        recent = list(self.signal_history)[-window_size:]
        
        # Count zero crossings (oscillation indicator)
        zero_crossings = 0
        for i in range(1, len(recent)):
            if (recent[i-1] >= 0 and recent[i] < 0) or (recent[i-1] < 0 and recent[i] >= 0):
                zero_crossings += 1
        
        # Check for oscillatory pattern: multiple zero crossings + amplitude variation
        amplitude_variation = max(recent) - min(recent)
        has_oscillation = (zero_crossings >= OSCILLATION_MIN_CYCLES * 2 and 
                          amplitude_variation >= MIN_OSCILLATION_AMPLITUDE)
        
        return has_oscillation, zero_crossings
    
    def is_direct_hit(self, current_signal, sample_num):
        """
        Detect if this is a direct sensor hit (not a hammer strike).
        Direct hits: Very large amplitude, immediate, no oscillation pattern.
        """
        if abs(current_signal) < DIRECT_HIT_THRESHOLD:
            return False
        
        # Check if we had a quiet period before this spike
        # Real hammer strikes usually come after quiet periods
        if self.quiet_period_count < 50:  # Not enough quiet period
            # Could be a direct hit or noise spike
            if abs(current_signal) > FALSE_POSITIVE_THRESHOLD:
                # Very large spike without quiet period = likely direct hit
                return True
        
        # Check for oscillation pattern
        has_oscillation, zero_crosses = self.detect_oscillation()
        
        # If spike is very large but no oscillation pattern = likely direct hit
        if abs(current_signal) > DIRECT_HIT_THRESHOLD and not has_oscillation:
            # Wait a bit to see if oscillation develops
            if sample_num - self.last_spike_sample > 20:  # Give time for oscillation
                if not has_oscillation:
                    return True  # No oscillation after waiting = direct hit
        
        return False
    
    def update(self, signal, sample_num):
        """
        Update spike detector and return validation result.
        Returns: (is_valid_spike, is_direct_hit, has_oscillation)
        """
        self.signal_history.append(signal)
        
        # Track quiet periods
        if abs(signal) < NOISE_THRESHOLD * 2:
            self.quiet_period_count += 1
        else:
            self.quiet_period_count = 0
        
        # Detect spike
        is_spike = abs(signal) > IMPACT_THRESHOLD
        
        if is_spike:
            self.last_spike_sample = sample_num
            self.spike_detected = True
            self.current_is_spike = True
            
            # Check if direct hit
            is_direct = self.is_direct_hit(signal, sample_num)
            self.current_is_direct_hit = is_direct
            
            # Check for oscillation (might not be present immediately)
            has_oscillation, zero_crosses = self.detect_oscillation()
            self.current_has_oscillation = has_oscillation
            
            return True, is_direct, has_oscillation
        else:
            # Check oscillation in recent history (after spike)
            if self.spike_detected and (sample_num - self.last_spike_sample) < SPIKE_DETECTION_WINDOW:
                has_oscillation, zero_crosses = self.detect_oscillation()
                self.current_has_oscillation = has_oscillation
                # Update direct hit status - if oscillation develops, it's not a direct hit
                if has_oscillation:
                    self.current_is_direct_hit = False
                return self.current_is_spike, self.current_is_direct_hit, has_oscillation
            else:
                # No recent spike
                self.current_is_spike = False
                self.current_is_direct_hit = False
                self.current_has_oscillation = False
        
        return False, False, False

# ---------------- INITIALIZE ----------------
print("Initializing ADS1256 ADC...")
adc_reset()
select_channel(0)
time.sleep(0.1)

# Calibrate baseline to remove DC offset (tilt/gravity component)
# This fixes the "floating data" problem by removing static offset
baseline_offset, measured_noise = calibrate_baseline()

# Adjust noise threshold based on measured noise if it's higher than default
if measured_noise > NOISE_THRESHOLD:
    NOISE_THRESHOLD = measured_noise * 1.5  # Use 1.5x measured noise as threshold
    print(f"Auto-adjusted noise threshold to {NOISE_THRESHOLD:.6f} V based on calibration")

# Initialize high-pass filter to remove any remaining DC drift
hpf = HighPassFilter(alpha=HPF_ALPHA)

# Initialize tilt remover to subtract slow tilt from signal
tilt_remover = TiltRemover(window_size=TILT_FILTER_WINDOW)

# Initialize spike detector to distinguish direct hits from hammer strikes
spike_detector = SpikeDetector()

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
print("Tilt removal active - subtracting slow tilt, preserving fast vibration")
print("High-pass filter active - removing remaining slow drift")
print("Noise threshold: VERY LOW (0.5mV) - real vibration will pass through")
print("DEBUG MODE: Showing diagnostic information")
print("Showing RAW vibration data (no smoothing, no FFT yet)")
print("")
print("Purpose: Verify system is working correctly")
print("IMPORTANT: Place sensor DIRECTLY on impact point or very close to it")
print("Smash beside sensor - vibration should travel through concrete/rock")
print("If no signal detected, check:")
print("  - Sensor contact with surface (must be firmly attached)")
print("  - Distance from impact point (closer = stronger signal)")
print("  - Impact force (harder hit = stronger vibration)")
print("")
print("FFT analysis for defect detection will be added later")
print("Press Ctrl+C to stop.\n")

# ---------------- MAIN LOOP ----------------
sample_count = 0
raw_voltage_history = deque(maxlen=100)  # Track raw voltage for floating data detection
quiet_period_samples = 0  # Count consecutive quiet samples
raw_movement_history = deque(maxlen=50)  # Track raw movement for debug
signal_history_for_spike = deque(maxlen=SPIKE_DETECTION_WINDOW)  # For spike detection

try:
    # Initialize spike detection variables
    current_is_spike = False
    current_is_direct_hit = False
    current_has_oscillation = False
    
    while True:
        # Wait for DRDY low
        while lgpio.gpio_read(h, DRDY_PIN) == 1:
            pass

        raw_val = read_adc()
        voltage = adc_to_voltage(raw_val)
        raw_voltage_history.append(voltage)  # Track for floating data detection

        # STEP 1: Calculate distance from original position
        # Developer: "yung kinukuha dapat natin yung layo nung galaw vs sa original position"
        # (We need to get the distance of movement vs the original position)
        # This measures how far the accelerometer has moved from its original position
        movement_from_original = voltage - baseline_offset
        raw_movement_history.append(movement_from_original)  # For debug

        # STEP 2: Remove slow tilt using exponential moving average
        # This preserves fast vibration (from smashing) while removing slow tilt
        # Tilt = slow movement, Vibration = fast movement
        movement_no_tilt = tilt_remover.remove_tilt(movement_from_original)

        # STEP 3: Apply high-pass filter to remove any remaining slow drift
        # This is a secondary filter to catch any remaining DC/low-frequency components
        vibration_signal = hpf.filter(movement_no_tilt)
        
        # DEBUG: Also track raw movement (before all filtering) to see if signal exists
        if DEBUG_MODE:
            raw_signal_for_debug = movement_from_original

        # STEP 4: Detect FLOATING DATA (not tilt, not vibration)
        # Developer: "parang floating data, di sa tilt or sa vibration eh"
        # Floating data = slow drift in raw voltage even when stationary
        floating_data_detected = False
        if len(raw_voltage_history) >= 50:
            # Check for slow drift in raw voltage (floating data)
            voltage_trend = np.polyfit(range(len(raw_voltage_history)), list(raw_voltage_history), 1)[0]
            voltage_std = np.std(list(raw_voltage_history))
            
            # Floating data: slow drift (>0.0001 V/sample) with low variation
            if abs(voltage_trend) > 0.0001 and voltage_std < 0.01:
                floating_data_detected = True

        # STEP 5: Apply noise threshold to filter electrical noise
        # IMPORTANT: Keep threshold VERY LOW so real vibration from smashing passes through
        # The amplifier and ADC can pick up noise even when stationary
        # Only filter very small signals (electrical noise), not real vibration
        # For impact echo, even small vibrations are important
        vibration_before_threshold = vibration_signal
        
        # CRITICAL FIX: Don't filter if we detect significant raw movement
        # If raw movement is large, it's real vibration, not noise
        raw_movement_magnitude = abs(movement_from_original)
        
        # Only apply noise threshold if raw movement is also small
        # If raw movement is large (>10mV), it's definitely real vibration
        if raw_movement_magnitude > 0.01:  # 10mV threshold for raw movement
            # Large raw movement = real vibration, don't filter!
            quiet_period_samples = 0
        elif abs(vibration_signal) < NOISE_THRESHOLD:
            # Signal is below noise threshold AND raw movement is small - treat as zero
            vibration_signal = 0.0
            quiet_period_samples += 1
        else:
            quiet_period_samples = 0

        # STEP 6: Spike Detection and Validation
        # Developer: "nasabayn siya sa spike" - system should align with spike
        # "yung kasama sa raw data is yung vibration every spike.. tapos yung bumlik na vibration"
        # (Raw data should include vibration every spike and the returning vibration)
        # Problem: "minsan kpag nakastay lng siya is biglang may mga unnecessary n signals na mlalki na ndedetect"
        # (Sometimes when stationary, unnecessary large signals are detected)
        # Problem: "nagreresponse yung graph kapag yung mismong adxl yung pinupukpok sa concrete"
        # (Graph responds when ADXL itself is hit on concrete, not when hammer hits)
        
        # Use movement_no_tilt for spike detection (preserves vibration pattern)
        signal_for_spike_detection = movement_no_tilt
        signal_history_for_spike.append(signal_for_spike_detection)
        
        # Update spike detector
        is_spike, is_direct_hit, has_oscillation = spike_detector.update(signal_for_spike_detection, sample_count)
        
        # Store for status reporting
        current_is_spike = is_spike
        current_is_direct_hit = is_direct_hit
        current_has_oscillation = has_oscillation
        
        # STEP 7: Store RAW vibration data (no smoothing)
        # Developer: "for raw data palang, di pa nalalagyan nung FFT"
        # (Just for raw data, FFT not added yet)
        # We need clean raw data first to verify system is working
        # NO smoothing - we need raw signals for later FFT analysis
        
        # CRITICAL FIX: Filter out direct sensor hits and false positives
        raw_movement_magnitude = abs(movement_from_original)
        
        # Reject direct sensor hits - these are NOT what we want
        # We want vibrations from hammer strikes, not direct hits on ADXL
        if is_direct_hit:
            # Direct sensor hit detected - reject it (set to zero or very small)
            # This addresses: "nagreresponse yung graph kapag yung mismong adxl yung pinupukpok"
            display_signal = 0.0
            if DEBUG_MODE and sample_count % 100 == 0:
                print(f"  🚫 DIRECT SENSOR HIT REJECTED (no oscillation pattern)")
        elif raw_movement_magnitude > 0.01:  # If raw movement > 10mV
            # Large raw movement detected - check if it's a valid hammer strike
            if is_spike and has_oscillation:
                # Valid hammer strike with oscillation - this is what we want!
                # Use the signal after HPF (before noise threshold) to preserve it
                if abs(vibration_before_threshold) > 0.0001:
                    display_signal = vibration_before_threshold
                else:
                    display_signal = movement_no_tilt
            elif is_spike and not has_oscillation:
                # Spike detected but no oscillation yet - might be developing
                # Wait a bit, but still show the signal (it might develop oscillation)
                if abs(vibration_before_threshold) > 0.0001:
                    display_signal = vibration_before_threshold
                else:
                    display_signal = movement_no_tilt
            else:
                # Large movement but not a clear spike - could be valid vibration
                if abs(vibration_before_threshold) > 0.0001:
                    display_signal = vibration_before_threshold
                else:
                    display_signal = movement_no_tilt
        else:
            # Small raw movement - apply stricter filtering to prevent false positives
            # This addresses: "minsan kpag nakastay lng siya is biglang may mga unnecessary n signals"
            if quiet_period_samples < QUIET_PERIOD_FOR_VALIDATION:
                # Not enough quiet period - might be false positive
                if abs(vibration_signal) > FALSE_POSITIVE_THRESHOLD:
                    # Large signal without quiet period = likely false positive
                    display_signal = 0.0
                else:
                    display_signal = vibration_signal
            else:
                # Had quiet period - more likely to be real
                display_signal = vibration_signal
        
        data.append(display_signal)  # Store signal for display

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

        # Detect impact events and print status with floating data detection
        if sample_count % 500 == 0:
            # Check if we're seeing vibration (not just noise)
            recent_data = list(data)[-100:] if len(data) >= 100 else list(data)
            
            # Determine signal type with spike detection info
            if floating_data_detected:
                status = "⚠️ FLOATING DATA DETECTED (not tilt/vibration) - ADC drift/noise"
                signal_type = "FLOATING"
            elif recent_data:
                signal_range = max(recent_data) - min(recent_data)
                if signal_range > IMPACT_THRESHOLD:
                    # Check spike detector status
                    if current_is_direct_hit:
                        status = "🚫 DIRECT SENSOR HIT (rejected) - Hit concrete, not sensor!"
                        signal_type = "DIRECT_HIT"
                    elif current_has_oscillation:
                        status = "✅ HAMMER STRIKE DETECTED - Oscillatory echoes present!"
                        signal_type = "VIBRATION"
                    else:
                        status = "✅ IMPACT DETECTED - Waiting for oscillation..."
                        signal_type = "SPIKE"
                elif quiet_period_samples > QUIET_PERIOD_FOR_VALIDATION:
                    status = "🔇 QUIET - System idle (ready for impact)"
                    signal_type = "QUIET"
                else:
                    status = "⏳ Waiting for impact..."
                    signal_type = "WAITING"
            else:
                status = "🔄 Initializing..."
                signal_type = "INIT"
            
            # Calculate statistics for diagnosis
            if len(raw_voltage_history) >= 50:
                voltage_drift = np.polyfit(range(len(raw_voltage_history)), list(raw_voltage_history), 1)[0]
                voltage_variation = np.std(list(raw_voltage_history))
            else:
                voltage_drift = 0.0
                voltage_variation = 0.0
            
            print(f"Sample {sample_count}: [{signal_type}]")
            print(f"  Raw Voltage: {voltage:.6f}V")
            print(f"  Movement (before filters): {movement_from_original:.6f}V")
            if DEBUG_MODE and len(raw_movement_history) >= 10:
                raw_range = max(raw_movement_history) - min(raw_movement_history)
                print(f"  Raw Movement Range (last 50): {raw_range:.6f}V")
            print(f"  After Tilt Removal: {movement_no_tilt:.6f}V")
            print(f"  After HPF: {vibration_before_threshold:.6f}V")
            print(f"  Final Vibration: {vibration_signal:.6f}V")
            if len(raw_voltage_history) >= 50:
                print(f"  Voltage Drift: {voltage_drift*1000:.3f}mV/sample | Variation: {voltage_variation*1000:.3f}mV")
            print(f"  Spike Detection: Spike={current_is_spike}, DirectHit={current_is_direct_hit}, Oscillation={current_has_oscillation}")
            print(f"  Quiet Period: {quiet_period_samples} samples")
            print(f"  Status: {status}\n")
            
            # DEBUG: Alert if raw movement shows signal but final is zero
            if DEBUG_MODE and len(raw_movement_history) >= 10:
                raw_max = max(abs(min(raw_movement_history)), abs(max(raw_movement_history)))
                if raw_max > 0.01 and abs(vibration_signal) < 0.001:
                    print(f"  ⚠️ WARNING: Raw movement shows {raw_max:.6f}V but filtered to {vibration_signal:.6f}V")
                    print(f"     Signal might be filtered out! Check filter settings.\n")

except KeyboardInterrupt:
    print("Live capture stopped by user")

finally:
    spi.close()
    lgpio.gpiochip_close(h)
    plt.ioff()
    plt.show()
    print("Resources closed")
import spidev
import time
import lgpio
import matplotlib.pyplot as plt
from collections import deque

# ---------------- CONFIG ----------------
VREF = 3.3            # ADC reference voltage
SPI_BUS = 0
SPI_DEVICE = 0        # CE0
SPI_SPEED_HZ = 5000

DRDY_PIN = 25         # GPIO connected to DRDY (Pin 11)
MAX_POINTS = 500      # Number of points shown in live plot
PLOT_UPDATE = 5       # Update plot every 5 samples
SMOOTH_WINDOW = 5     # Number of samples for moving average

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

# ---------------- INITIALIZE ----------------
adc_reset()
select_channel(0)
time.sleep(0.1)

# ---------------- SETUP LIVE PLOT ----------------
plt.ion()
fig, ax = plt.subplots()
data = deque([VREF/2]*MAX_POINTS, maxlen=MAX_POINTS)
line, = ax.plot(data)
ax.set_ylim(0, VREF)
ax.set_title("ADXL1002Z Live Vibration (Smoothed)")
ax.set_xlabel("Sample")
ax.set_ylabel("Voltage (V)")

print("Starting live vibration capture. Press Ctrl+C to stop.")

# ---------------- MAIN LOOP ----------------
sample_count = 0
try:
    while True:
        # Wait for DRDY low
        while lgpio.gpio_read(h, DRDY_PIN) == 1:
            pass

        raw_val = read_adc()
        voltage = adc_to_voltage(raw_val)

        # Append new reading
        data.append(voltage)

        # Apply moving average for smoother plot
        smoothed_voltage = moving_average(data, SMOOTH_WINDOW)
        data[-1] = smoothed_voltage  # replace latest with smoothed value

        sample_count += 1

        # Update plot every PLOT_UPDATE samples
        if sample_count % PLOT_UPDATE == 0:
            line.set_ydata(data)
            ax.set_ylim(min(data)-0.05, max(data)+0.05)
            fig.canvas.draw()
            fig.canvas.flush_events()

except KeyboardInterrupt:
    print("Live capture stopped by user")

finally:
    spi.close()
    lgpio.gpiochip_close(h)
    plt.ioff()
    plt.show()
    print("Resources closed")
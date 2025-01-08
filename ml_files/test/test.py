import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data= pd.read_csv(r'D:\dev\biomedical-signal-research\datasets\csv_files\datasets_subject_01_to_10_scidata\GDN0001\GDN0001_1_Resting.csv')


radar_i = data['radar_i'].values
radar_q = data['radar_q'].values

# Apply FFT
fft_values_i = np.fft.fft(radar_i)
fft_frequencies_i = np.fft.fftfreq(len(radar_i))

fft_values_q = np.fft.fft(radar_q)
fft_frequencies_q = np.fft.fftfreq(len(radar_q))

# Plot positive frequencies for radar_i
plt.figure(figsize=(12, 6))
plt.plot(fft_frequencies_i[:len(radar_i)//2], np.abs(fft_values_i[:len(radar_i)//2]), label='radar_i')
plt.plot(fft_frequencies_q[:len(radar_q)//2], np.abs(fft_values_q[:len(radar_q)//2]), label='radar_q', linestyle='--')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Frequency Spectrum of radar_i and radar_q')
plt.legend()
plt.show()


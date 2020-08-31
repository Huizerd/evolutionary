import numpy as np

decimal = -117.23

binstring = bin(np.float16(decimal).view("H"))[2:].zfill(16)
print(binstring)
spike_array = np.array(list(binstring)).astype(float)
print(spike_array)

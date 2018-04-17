# Schottky Spectroscopy Data Monitor
The display of the acquired Schottky data in real time is of importance to monitor the experimental status, and possibly, change the machine condition accordingly whenever necessary.
It allows for an extremely useful feedback to optimize the beam-time utilizing efficiency to the maximum.
A good data monitoring system must preferably provide with a graphical user interface to display the Schottky spectra on screen.
Besides, it should promptly respond to the ambient changes, whether a new file has arrived at the server or the end user has clicked a button, dragged a figure, and so on.

This program watches a pre-assigned directory where all the data files reside in for any changes.
If a new file has come, the program extracts meta-info relevant to the data acquisition from the header, and Fourier transforms the time signal into the frequency spectrogram.
The generated results are visualized with 1D and 2D figures.

## Prerequisites
 - `Python 3`
 - `scipy`, `numpy`, `matplotlib`
 - `pyfftw` (_fast compute Fourier transform_)

## Usage
```python
python3 monitor.py
```

## License
This repository is licensed under the **GNU GPLv3**.

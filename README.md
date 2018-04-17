# Schottky Spectroscopy Data Monitor
The display of the acquired Schottky data in real time is of importance to monitor the experimental status, and possibly, change the machine condition accordingly whenever necessary.
It allows for an extremely useful feedback to optimize the beam-time utilizing efficiency to the maximum.
A good data monitoring system must preferably provide with a graphical user interface (GUI) to display the Schottky spectra on screen.
Besides, it should promptly respond to the ambient changes, whether a new file has arrived at the server or the end user has clicked a button, dragged a figure, and so on.

This program watches a pre-assigned directory where all the data files reside in for any changes.
If a new file has come, the program extracts meta-info relevant to the data acquisition from the header, and Fourier transforms the time signal into the frequency spectrogram.
The generated results are visualized with 1D and 2D figures.

## Prerequisites
 - `Python 3`
 - `scipy`, `numpy`, `matplotlib`
 - `pyfftw` (_fast compute Fourier transform_)
 - `pyqt5`, `pyqtgraph` (_fast render graphics_)
 - `Roboto Condensed` (_optional font, freely accessible at [Google Fonts](https://fonts.google.com/specimen/Roboto+Condensed)_)

## Inventory
 1. `monitor.py`: main script to be executed; `monitor.ui`: generated by `Qt creator` to describe the GUI layout
 2. `multithread.py`: multithreading the workflow to keep the GUI responsive even when time-consuming tasks are going on behind the scene
 3. **`dpss.py`**, **`preprocessing.py`**, **`processing.py`**: to be detailed in another repo

## Usage
```python
python3 monitor.py
```

A `Qt` interface will pop out when the program is launched.
The interface itself is quite intuitive to play with, where a list of stored data files are shown on the right and the Schottky spectra of the currently selected file are displayed on the left.

By default the file pointer points to the latest one on the list, and it automatically refreshes once a new file has been saved.
To disable it, toggle the refresh mode from `auto` to `manual`, shown at the top right corner of the GUI.
This can sometimes become convenient if one wants to browse the files at his/her interest.

## License
This repository is licensed under the **GNU GPLv3**.

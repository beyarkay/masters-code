from serial.tools.list_ports import comports
import serial
from time import perf_counter
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

app = pg.mkQApp()

label_fps = QtWidgets.QLabel()
layout = pg.LayoutWidget()

plots = []
datas = []
layout.addWidget(label_fps, row=0, col=0)
predictions_plot = pg.plot()
layout.addWidget(predictions_plot, row=0, col=1)
for col in range(2):
    plots.append([])
    datas.append([])
    for row in range(5):
        datas[col].append([[], [], []])
        plots[col].append(pg.plot())
        plots[col][row].setYRange(200, 1000)
        layout.addWidget(plots[col][row], row=row + 1, col=col)

layout.resize(800, 1000)
layout.show()

lastUpdate = perf_counter()
avgFps = 0.0

baudrate = 19_200
port = "/dev/cu.usbmodem1401"

serial_port = serial.Serial(port=port, baudrate=baudrate, timeout=1)


def update():
    global label_fps, lastUpdate, avgFps
    if not serial_port.isOpen():
        return
    values = serial_port.readline().decode("utf-8").strip().split(",")
    values = ["0"] * (33 - len(values)) + values
    values = np.array([int(v) for v in values[2:-1]])

    for col in range(2):
        for row in range(5):
            for dim in range(3):
                datas[col][row][dim].append(values.reshape((2, 5, 3))[col, row, dim])
                plots[col][row].plot(
                    datas[col][row][dim][-50:],
                    clear=(dim == 0),
                    skipFiniteCheck=True,
                    pen=(dim, 3),
                )
    now = perf_counter()
    fps = 1.0 / (now - lastUpdate)
    lastUpdate = now
    avgFps = avgFps * 0.8 + fps * 0.2
    label_fps.setText(f"Frames per second: {avgFps:0.1f}")


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

if __name__ == "__main__":
    pg.exec()
    print("Done")
    if serial_port is not None:
        serial_port.close()

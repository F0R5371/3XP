from pyo import *

server = Server(nchnls=2, ichnls=1)
server.deactivateMidi()
server.boot()
server.start()

a = Sine(mul = 0.01).out()

input("Stop: ")

server.stop()
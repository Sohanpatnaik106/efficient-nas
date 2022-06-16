import os
import sys

class Logger(object):
    def __init__(self, name):
        self.terminal = sys.stdout
        self.log = open(name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()

# NOTE: Write the lines in main.py file to log out
# log_out = args.log_dir + '/output.log'
# sys.stdout = Logger(log_out)


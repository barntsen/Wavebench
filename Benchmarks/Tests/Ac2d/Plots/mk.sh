#!/bin/sh
python3 solvertime-rtx4070.py
python3 solvertime-a100.py
python3 solvertime-gh200.py
python3 solvertime-eps-gh200-a100-rtx4070.py
python3 speedup-single-gh200.py
python3 speedup-multi-gh200.py
python3 solvertime-mi250x.py


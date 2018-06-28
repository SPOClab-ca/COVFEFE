import os
import subprocess

def shell_run(params, silent=True):
    if silent:
        fnull = open(os.devnull, "w")

    res = subprocess.call(params, stdout=fnull, stderr=fnull)
    fnull.close()

    return res
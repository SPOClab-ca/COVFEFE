import os
import subprocess

def shell_run(cmd, stdout=None, stdin=None, stderr=None):
    if stderr is None:
        stderr = open(os.devnull, "w")
    elif isinstance(stderr, str):
        stderr = open(stderr, "w")

    if stdout is None:
        stdout = open(os.devnull, "w")
    elif isinstance(stdout, str):
        stdout = open(stdout, "w")

    res = subprocess.call(cmd, stdout=stdout, stderr=stderr, stdin=stdin)

    stderr.close()
    stdout.close()

    return res
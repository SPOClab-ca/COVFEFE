
from abc import ABC, abstractmethod
import os
import logging

from nodes.helper_nodes import FileOutputNode
from utils import file_utils
from utils.shell_run import shell_run
from config import OPENSMILE_HOME

class Mp3ToWav(FileOutputNode):
    def run(self, mp3_file):
        self.log(logging.INFO, "Starting %s" % (mp3_file))

        if not mp3_file.endswith(".mp3"):
            self.log(logging.ERROR,"Failed %s. Not mp3 file" % (mp3_file))
            return

        wav_file = self.derive_new_file_path(mp3_file, "wav")

        if file_utils.should_run(mp3_file, wav_file):
            res = shell_run(["lame", "--decode", mp3_file, wav_file])

            if res != 0:
                self.log(logging.ERROR,"Failed %s -> %s with lame error code %i" % (mp3_file, wav_file, res))
                return

            self.log(logging.INFO, "Done %s -> %s" % (mp3_file, wav_file))

        self.emit(wav_file)


class ResampleWav(FileOutputNode):
    def setup(self, new_sr):
        self.new_sr = new_sr

    def run(self, wav_file):
        self.log(logging.INFO, "Starting %s" % (wav_file))

        if not wav_file.endswith(".wav"):
            self.log(logging.ERROR,"Failed %s. Not wav file" % (wav_file))
            return

        new_wav_file = self.derive_new_file_path(wav_file, "wav")

        if file_utils.should_run(wav_file, new_wav_file):
            res = shell_run(["sox", wav_file, "--rate", str(self.new_sr), new_wav_file])

            if res != 0:
                self.log(logging.ERROR,"Failed %s -> %s with lame error code %i" % (wav_file, new_wav_file, res))
                return

            self.log(logging.INFO, "Done %s -> %s" % (wav_file, new_wav_file))

        self.emit(new_wav_file)


class ShellCommand(FileOutputNode):
    """
        Take as input a format string representing a shell command that can accept an in_file and out_file.
        For example "someCommand -i {in_file} -o {out_file}"
        ext: Extension of the output file, ex. "wav", "csv"
    """
    def setup(self, command, ext):
        self.command = command
        self.ext = ext

    def run(self, in_file):
        self.log(logging.INFO, "Starting %s" % (in_file))

        out_file = self.derive_new_file_path(in_file, self.ext)

        if file_utils.should_run(in_file, out_file):
            cmd = self.command.format(in_file=in_file, out_file=out_file)
            res = shell_run(cmd.split(" "))

            if res != 0:
                self.log(logging.ERROR,"Failed %s -> %s with error code %i. cmd: %s" % (in_file, out_file, res, cmd))
                return

            self.log(logging.INFO, "Done %s -> %s" % (in_file, out_file))

        self.emit(out_file)


class OpenSmileRunner(FileOutputNode):
    """
        conf_file: Either absolute path to an opensmile conf file or the name of a config file in opensmile's config folder
        out_flag: Flag to use for the output file.
        extra_flags: A string of extra flags to pass to SMILExtract.
        out_ext: Extension of the output file
    """

    def setup(self, conf_file, out_flag="-csvoutput", extra_flags="-nologfile -noconsoleoutput -appendcsv 0", out_ext="csv"):
        self.conf_file = file_utils.locate_file(conf_file, [os.path.join(OPENSMILE_HOME, "config")])
        self.extra_flags = extra_flags.split(" ")
        self.out_flag = out_flag
        self.out_ext = out_ext

        self.opensmile_exec = file_utils.locate_file("SMILExtract", [OPENSMILE_HOME])


    def run(self, in_file):
        self.log(logging.INFO, "Starting %s" % (in_file))

        out_file = self.derive_new_file_path(in_file, self.out_ext)

        if file_utils.should_run(in_file, out_file):
            cmd = [self.opensmile_exec, "-C", self.conf_file, "-I", in_file, self.out_flag, out_file] + self.extra_flags
            res = shell_run(cmd)

            if res != 0:
                self.log(logging.ERROR,"Failed %s -> %s with SmileExtract error code %i. cmd: %s" % (in_file, out_file, res, " ".join(cmd)))
                return

            self.log(logging.INFO, "Done %s -> %s" % (in_file, out_file))

        self.emit([out_file])



class IS10_Paraling(OpenSmileRunner):

    def get_conf_name(self):
        return "IS10_paraling.conf"

    def get_command(self, wav_file, out_file):
        return [self.os_exec, "-C", self.conf_file, "-I", wav_file, "-csvoutput", out_file, "-nologfile", "-noconsoleoutput", "-appendcsv", "0"]


class IS10_Paraling_lld(OpenSmileRunner):

    def get_conf_name(self):
        return "IS10_paraling.conf"

    def get_command(self, wav_file, out_file):
        return [self.os_exec, "-C", self.conf_file, "-I", wav_file, "-lldcsvoutput", out_file, "-nologfile", "-noconsoleoutput", "-appendcsv", "0"]

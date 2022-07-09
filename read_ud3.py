#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import os

# class for reading ud3 Files to 3d rti matrices

# Author: Matthias Clahsen (clahsen@iap-kborn.de)
#      @: Leibnitz Institut für Athmosphärenphysik, Kühlungsborn

# used python version: 2.7.6
# used numpy version : 1.9.1

class UD3_Reader(object):

    def __init__(self, file_name, t_mess=60.0):
        super(UD3_Reader, self).__init__()
        self.t_mess = t_mess

        self.chunk = 0  # The chunk you've actually read

#        if not ".ud3" in file_name:
 #           raise IOError("Given file is not a ud3 file!")
        self.in_file = open("%s" % (file_name), 'r',encoding='iso-8859-1')

        # Read Parameters and jump to begin of data
        params = []
        l = ""
        while not "DATA" in l:
            l = self.in_file.readline()[:-1]
            print(l)
            params.append(l)
        params.pop(0)  # remove comment on firstline

        # Create a dictionary with parameters and delete the list
        self.param_dict = {l.split(None, 1)[0]:
                           (l.split(None, 1)[1] if len(
                               l.split(None, 1)) > 1 else '')
                           for l in params}
        del(params)

        # Calculate n_chunks out of t_mess and ACQ_TIME
        # rounds to next integer
        self.n_chunks = int(numpy.round(
                            int(self.param_dict["ACQ_TIME"]) * 1.0 / t_mess)
                            )
        print("Beginne mit Datei: %s" % self.file_name)
        print("Time duration of File: " + str(self.param_dict["ACQ_TIME"]))
        print("Number of chunks: " + str(self.n_chunks))
        print("Time duration of chunk: " + str(
            int(self.param_dict["ACQ_TIME"]) *
            1.0 / self.n_chunks))
    # properties of the filereader to provide some parameters
    # out of param_dict, these are readonly
    ##########################################################################

    def PRF():
        doc = "The PRF property."

        def fget(self):
            return int(self.param_dict['PRF'])
        return locals()
    PRF = property(**PRF())

    def time():
        doc = "The time property."

        def fget(self):
            return int(self.param_dict['DATA'])
        return locals()
    time = property(**time())

    def n_ranges():
        doc = "The n_ranges property."

        def fget(self):
            return int(self.param_dict['GATES'])
        return locals()
    n_ranges = property(**n_ranges())
    
    def n_integrations():
        doc = "The n_integrations property."

        def fget(self):
            return int(self.param_dict['INTEGRATIONS'])
        return locals()
    
    n_integrations = property(**n_integrations())

    def time_duration():
        doc = "The time_duration property."

        def fget(self):
            return int(self.param_dict['ACQ_TIME'])
        return locals()
    time_duration = property(**time_duration())

    def n_chan():
        doc = "The n_chan property."

        def fget(self):
            return int(int(self.param_dict['CHANNELS']) / 2)
        return locals()
    n_chan = property(**n_chan())
    

    def n_points_chunk():
        doc = "The n_points_chunk property."

        def fget(self):
            return int(
                1.0 * self.time_duration * self.PRF / self.n_chunks / self.n_integrations
            ) * self.n_ranges * self.n_chan
        return locals()
    n_points_chunk = property(**n_points_chunk())

    def deltaR():
        doc = "The deltaR property."

        def fget(self):
            return 1.0 * int(self.param_dict['RESOLUTION']) / 1000
        return locals()
    deltaR = property(**deltaR())

    def phase_offsets():
        doc = "The phase_offsets property."

        def fget(self):
            return numpy.array(
                [float(ph) for ph in
                 self.param_dict['PHASE_OFFSETS'].split()]
            )
        return locals()
    phase_offsets = property(**phase_offsets())

    def frequency():
        doc = "The frequency property."

        def fget(self):
            return int(self.param_dict['FREQUENCY']) * 1.0e-6
        return locals()
    frequency = property(**frequency())

    def range_offset():
        doc = "The range_offset property."

        def fget(self):
            return int(self.param_dict['RANGE']) * 1.0e-3
        return locals()
    range_offset = property(**range_offset())

    def antenna_coords():
        doc = "The antenna_coords property."
        
        def fget(self):
            print(self.param_dict['ANTENNA_COORDS'].split())
            print(self.n_chan)
            return numpy.array(
                [float(s) for s in
                 self.param_dict['ANTENNA_COORDS'].split()[0:2 * self.n_chan]]
            ).reshape(self.n_chan, 2)
        return locals()
    antenna_coords = property(**antenna_coords())

    def IPP():
        doc = "The IPP property."

        def fget(self):
            return 1.0 / self.PRF
        return locals()
    IPP = property(**IPP())

    def file_name():
        doc = "The file_name property."

        def fget(self):
            return os.path.basename(self.in_file.name)
        return locals()
    file_name = property(**file_name())

    def code():
        doc = "The code property."

        def fget(self):
            return numpy.array([float(c) for c in
                                self.param_dict['PULSE_CODE']],
                               dtype=numpy.float32) * 2.0 - 1
        return locals()
    code = property(**code())

    def time_vector():
        doc = "The time_vector property."

        def fget(self):
            time_len = self.n_points_chunk / (self.n_ranges * self.n_chan)
            return (numpy.arange(time_len) * 1.0 +
                    self.get_chunk_nr() * time_len
                    ) * self.IPP
        return locals()
    time_vector = property(**time_vector())
    ##########################################################################

    def _read_chunk(self):
        """reads one chunk of data out of file and return a 3d matrix with
        [chan, range, time]"""
        temp = numpy.fromfile(
            self.in_file,
            dtype="<i2",
            count=self.n_points_chunk * 2,
            sep='')
        if len(temp) != self.n_points_chunk*2:
            return(None)
        data = numpy.zeros(int(len(temp) / 2), dtype=numpy.complex64)
        data = temp[0::2] + 1j * temp[1::2]
        del(temp)
        return data.reshape(int(self.n_points_chunk / (self.n_ranges *
                                                   self.n_chan)),
                            self.n_ranges,
                            self.n_chan).transpose(2, 1, 0)

    def get_next_chunk(self):
        """get data of the next chunk"""
        if self.chunk == self.n_chunks:
            return None
        data3d = self._read_chunk()
        self.chunk += 1
        return data3d

    def get_chunk_nr(self):
        """returns the number of the last chunk read"""
        return self.chunk - 1

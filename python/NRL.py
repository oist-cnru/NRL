#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 BSD 3-Clause License

  Copyright (c) 2020 Okinawa Institute of Science and Technology (OIST).
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     disclaimer in the documentation and/or other materials provided
     with the distribution.
   * Neither the name of Willow Garage, Inc. nor the names of its
     contributors may be used to endorse or promote products derived
     from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.

 Author: Hendry F. Chame <hendryfchame@gmail.com>

 Publication:

   "Towards hybrid primary intersubjectivity: a neural robotics 
   library for human science"

   Hendry F. Chame, Ahmadreza Ahmadi, Jun Tani

   Okinawa Institute of Science and Technology Graduate University (OIST)
   Cognitive Neurorobotics Research Unit (CNRU)
   1919-1, Tancha, Onna, Kunigami District, Okinawa 904-0495, Japan

"""


import os
from sys import platform as _platform
from ctypes import *
import numpy as np

class NRL(object):    
    
    def __init__(self):
        
        libFolder = 'lib'
        libName = 'libNRL'
        if _platform == "linux" or _platform == "linux2": # linux
            libName = libName + '.so'            
        elif _platform == "darwin": # MAC OS X
            libName = libName + '.dylib'            
        elif _platform == "win32": # Windows
            libName = libName + '.dll'            
        elif _platform == "win64": # Windows 64-bit
            libName = libName + '.dll'            

        self.lib = cdll.LoadLibrary(os.path.dirname(__file__) + os.sep + libFolder + os.sep + libName)
        self.lib.getInstance.restype = POINTER(c_void_p)
        
        self.parser = None
        self.obj = self.lib.getInstance()
            
    def newModel(self, _propPath):
        
        self.lib.newModel(self.obj, _propPath)
        
    def setStateParser(self, _parser):
        
        self.parser = _parser
        
    def load(self):        
        
        self.lib.load(self.obj)         
            
    def t_init(self, _stdoutLog):        
        
        self.lib.t_init(self.obj, _stdoutLog)
    
    def t_loop(self, _trainOut, _nLoop):        
        
        self.lib.t_loop(self.obj, _trainOut, _nLoop);            
    
    def t_end(self):        
        
        self.lib.t_end(self.obj)
    
    def t_background(self):        
        
        self.lib.t_background(self.obj)
                                              
    def e_enable(self, _pId, _winSize, _w, _expTime, _epoch, _alpha, _beta1, _beta2, _storeStates=False, _storeER=False):
        
        self.lib.e_enable(self.obj, _pId, _winSize, _w, _expTime, _epoch, _alpha, _beta1, _beta2, _storeStates, _storeER)
        
    def e_postdict(self, _pos_win, _elbo, _showLog):
        
        return self.lib.e_postdict(self.obj, _pos_win, _elbo, _showLog)
    
    def e_generate(self, _tgt_pos):
        
        self.lib.e_generate(self.obj, _tgt_pos)

    def e_save(self, _modelPath):
        
        self.lib.e_save(self.obj, _modelPath)

    def getStateBufferSize(self):
        
        return self.lib.getStateDim(self.obj)
                       
    def e_getState(self,_v):    
        
        self.lib.e_getState(self.obj, _v)
        
    def getNDof(self):        
        
        return self.lib.getNDof(self.obj)
    
    def getNLayers(self):
        
        return self.lib.getNLayers(self.obj)
    
    def a_predict(self, _path, _n, _init_state):
        
        self.lib.a_predict(self.obj, _path, _n, _init_state)	

    def a_feedForwardOutputFromContext(self, _context, _X):
        
        self.lib.a_feedForwardOutputFromContext(self.obj, _context, _X);	

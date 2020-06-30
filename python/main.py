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

import sys
import ctypes
from collections import deque
import time
from NRL import NRL
import numpy as np

"""
 Demonstration on how to train a model 
"""
class DemonstrateTraining(object):
    def __init__(self, nrl):
        print ("training demonstration begin")
        nrl.t_background();
        print ("training demonstration end")

"""
 Demonstration on the computation of on-line post-diction
"""

class DemonstratePostdiction(object):
    
    def __init__(self, nrl):        
        print("simulation demonstration begin")                
        nrl.load();
        nDof = nrl.getNDof();                    
        if nDof > 0:
            winSize = 15;
            winBufferSize = winSize * nDof;
            winBuffdata = deque(maxlen=winSize) # circular buffer
            primId = 0;

            # The e_w parameters set bellow assume the network has two layers
            # as in the original distribution of the sources
            # in case more layers are set by changing the properties.d file,
            # the same dimension for e_w must be considered
            e_w = [0.025,0.025];

            expTimeSteps = 15;
            postdiction_epochs = 15;
            alpha = 0.1;
            beta1 = 0.9;
            beta2 = 0.999;
            storeStates = False;
            storeER = False;
            showERLog = False;
            nrl.e_enable(primId,\
                         winSize,
                         (ctypes.c_float * len(e_w))(*e_w),
                         expTimeSteps,
                         postdiction_epochs,
                         (ctypes.c_float)(alpha),
                         (ctypes.c_float)(beta1),
                         (ctypes.c_float)(beta2), 
                         storeStates, 
                         storeER)

            # ctype input/output buffers to NRL
            tgt_pos_buffer = np.zeros((nDof,), dtype=float)
            dataOut = (ctypes.c_float * nDof)(*tgt_pos_buffer)		
                                                             
            elbo_buffer = np.zeros((3,), dtype=float);
            elboOut = (ctypes.c_float * 3)(*elbo_buffer)        
            
            stateBufferSize = nrl.getStateBufferSize()
            m_state = np.zeros((stateBufferSize,), dtype=float);
            m_stateOut = (ctypes.c_float * stateBufferSize)(*m_state)
                    
            t = 0

            endExperiment = False
            
            while not endExperiment:
                
                # get time in ms
                mst1 = int(round(time.time() * 1000))

                # < --- Here you should read the robot's joint state
                # Since this is a dummy example, the current posture 
                # 'cur_pos' is set to zero 
                cur_pos = np.zeros((nDof,), dtype=float)


                # The target posture is generated by the RNN
                nrl.e_generate(dataOut)                    
                tgt_pos = np.frombuffer(dataOut, np.float32)
                              
                # < --- Here you should call asynchronously the robot driver 
                # and send it tgt_pos to move the robot

                # store the current posture in the buffer
                winBuffdata.append(cur_pos)
            
                if len(winBuffdata) == winSize:
                    
                    t += 1   
                    posWinBufferArray = np.hstack(winBuffdata)                
                    nrl.e_postdict((ctypes.c_float * winBufferSize)(*posWinBufferArray), elboOut, showERLog);

                    # Optional: Information on free energy minimization     
                    # can be obtained and analyzed on-line
                    opt_elbo = np.frombuffer(elboOut, np.float32).tolist()                                             
                    
                    # Optional: The latent state of the network can be obtained 
                    # analyzed on-line or stored for future analysis
                    nrl.e_getState(m_stateOut)
                    st_data = np.frombuffer(m_stateOut, np.float32)
                        
             
                # get time in ms
                mst2 = int(round(time.time() * 1000))                
                detla_t = mst2 - mst1
                
                if t > 0:
                    print("Step: {} in {} ms".format(t, detla_t ))

                # For real applications you should set the program to sleep according to the
                # desired loop period

                period = 0.0 # loop period in ms   
                sleepTime = period - detla_t  
                if sleepTime > 0:
                    time.sleep(sleepTime/1000)

                if t == expTimeSteps:
                    endExperiment = True
                          
        else:
            print("Hint: the model path may be incorrect, or perhaps it requires to be trained !")
            
        print("simulation demonstration end")
       



	
print("-----------------------------")
print("Neural Robotics Libray (NRL)")
print("-----------------------------")
print("This is a demonstration program for stand alone application ")
print("**** Instructions **** ")
print("To run: NRL_SA [PATH] [train|sim]")
print("Arguments")
print("PATH:  Full path to the property file distributed in 'src/standalone/data/config/properties.d'" )
print("train: trains a model for a generic 16 degrees of freedom robot with dumb data")
print("       the parameters can be selected by editing the file 'properties.d'")
print("sim:   simulates on-line interaction with the robot during 50 time steps")
print("       the loop time in milliseconds is shown in the standard output")
print("******************** ")

nrl = NRL() 
# verifying the arguments

path = '';

argv = sys.argv
argc = len(argv)

if (argc > 1):
    for i in range(argc):        
        if i == 1:
            path = argv[i];        
            print("Loading the properties file from: [{}]".format(path));                  
            nrl.newModel(path.encode('ascii'));
            
            continue
        arg_s = argv[i].lower()
        if arg_s == "train":
            DemonstrateTraining(nrl)
        elif arg_s == "sim":            
            DemonstratePostdiction(nrl)
        else:
            print("Please indicate a valid argument [train,sim] !")				
print("Program end")


/*<!--

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

-->*/

#ifndef SRC_VBMTRNN_INCLUDES_H_
#define SRC_VBMTRNN_INCLUDES_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <vector>
#include <random>
#include <exception>
#include <chrono>
#include <math.h>
#include <map>
#include <type_traits>
#include <Eigen/Dense>

#include "utils/Exception.h"

#define NON_ZERO 1.0e-20f
using namespace std;
using namespace Eigen;

typedef vector<VectorXf> vectorXf1DContainer;
typedef vector<vectorXf1DContainer> vectorXf2DContainer;
typedef vector<vectorXf2DContainer> vectorXf3DContainer;
typedef vector<vectorXf3DContainer> vectorXf4DContainer;

typedef vector<ArrayXf> arrayXf1DContainer;
typedef vector<arrayXf1DContainer> arrayXf2DContainer;
typedef vector<arrayXf2DContainer> arrayXf3DContainer;
typedef vector<arrayXf3DContainer> arrayXf4DContainer;

typedef vector<ArrayXf> matrixXf1DContainer;
typedef vector<matrixXf1DContainer> matrixXf2DContainer;
typedef vector<matrixXf2DContainer> matrixXf3DContainer;
typedef vector<matrixXf3DContainer> matrixXf4DContainer;

typedef vector<bool> bool1DContainer;

typedef vector<int> int1DContainer;
typedef vector<int1DContainer> int2DContainer;
typedef vector<int2DContainer> int3DContainer;

typedef vector<float> float1DContainer;
typedef vector<float1DContainer> float2DContainer;
typedef vector<float2DContainer> float3DContainer;
typedef vector<float3DContainer> float4DContainer;


#endif /* SRC_VBMTRNN_INCLUDES_H_ */

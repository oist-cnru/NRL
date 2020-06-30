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


#ifndef ROBOT_IROBOT_H_
#define ROBOT_IROBOT_H_

#include "../includes.h"

namespace oist {

/**
 * Abstract class (Interface) for robot implementations
 * */
class IRobot {
public:

	/**
	 * Gets the number of degrees of freedom (DoF) of the robot
	 * @return Integer number of DoF
	 * */
	virtual int getDOF(void) = 0;

	/**
	 * Gets the inferior limit of the robot's joints
	 * @param output Output float vector container
	 * */
	virtual void getJointLowLimit(float1DContainer& output) = 0;

	/**
	 * Gets the superior limit of the robot's joints
	 * @param output Output float vector container
	 * */
	virtual void getJointHighLimit(float1DContainer& output) = 0;

	/**
	 * Gets the interval range of the robot's joints
	 * @param output Output float vector container
	 * */
	virtual void getJointRange(float1DContainer& output) = 0;


	/**
	 * Destructor
	 * */
	virtual ~IRobot(){}
};

} /* namespace oist */

#endif /* ROBOT_IROBOT_H_ */

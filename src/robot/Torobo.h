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

   Chame, H. F., Ahmadi, A., & Tani, J. (2020).
   A hybrid human-neurorobotics approach to primary intersubjectivity via
   active inference. Frontiers in psychology, 11.

   Okinawa Institute of Science and Technology Graduate University (OIST)
   Cognitive Neurorobotics Research Unit (CNRU)
   1919-1, Tancha, Onna, Kunigami District, Okinawa 904-0495, Japan

-->*/

#ifndef TOROBO_H_
#define TOROBO_H_

#include "IRobot.h"

namespace oist {

/**
 * This class implements a 16 degrees of freedom Torobo robot, form a singleton design pattern.
 * */
class Torobo: public IRobot {

	const int fullDof;
	float jmin[16] = {-65.0, -40.0, -155.0, -45.0, -155.0, -100.0, -65.0, -40.0, -155.0, -45.0, -155.0, -100.0, -75.0, -20.0, -85.0, -40.0};
	float jmax[16] = {245.0, 100.0,  155.0, 110.0,  155.0,  100.0, 245.0, 100.0,  155.0, 110.0,  155.0,  100.0, 75.0,  50.0,  85.0,  40.0};
	float jrange[16]; // joint interval in degree
	bool active[16] = {false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false};
	int nDof;

	static IRobot* instance;

	/**
	 * Constructor
	 * @param activeJoints Boolean vector indicating active joints
	 * */
	Torobo(vector<bool>& activeJoints);

	/**
	 * Destructor
	 * */
	virtual ~Torobo();

public:

	/**
	 * Gets the singleton instance
	 * */
	static IRobot* getInstance(vector<bool>& activeJoints);

	// ------- IRobot interface methods -------

	int  getDOF(void);
	void getJointLowLimit(float1DContainer&);
	void getJointHighLimit(float1DContainer&);
	void getJointRange(float1DContainer&);

};

} /* namespace oist */

#endif /* TOROBO_H_ */

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


#include "Cartesian.h"

namespace oist {

IRobot* Cartesian::instance = nullptr;

IRobot* Cartesian::getInstance(vector<bool>& _activeJoints){

	if(instance == nullptr)
		instance = new Cartesian(_activeJoints);

	return instance;
}

Cartesian::Cartesian(vector<bool>& _activeJoints) : fullDof(3){

	if (_activeJoints.size() != fullDof){
		stringstream stream;
		stream << "The activity of " << fullDof << " joints must be specified for Cartesian robot";
		throw  Exception(stream.str());
	}
	nDof = 0;

	for (unsigned int j = 0 ; j < _activeJoints.size(); j++){
		bool act_j = _activeJoints[j];
		active[j] = act_j;
		if (act_j)
			nDof ++;
	}

	for (int j = 0; j < fullDof; j++){
		jrange[j] = jmax[j] - jmin[j];
	}

	cout << endl << "Cartesian robot model: the number of active joints is " << nDof << endl;

}

int Cartesian::getDOF(void){

	return nDof;
}

void Cartesian::getJointLowLimit(float1DContainer& _v){

	_v.clear();
	for (int j = 0; j < fullDof; j++){
		if (active[j]){
			_v.push_back(jmin[j]);
		}
	}

}

void Cartesian::getJointHighLimit(float1DContainer& _v){

	_v.clear();
	for (int j = 0; j < fullDof; j++){
		if (active[j]){
			_v.push_back(jmax[j]);
		}
	}

}

void Cartesian::getJointRange(float1DContainer& _v){

	_v.clear();
	for (int j = 0; j < fullDof; j++){
		if (active[j]){
			_v.push_back(jrange[j]);
		}
	}

}


Cartesian::~Cartesian() {

	cout << "Cartesian Robot deallocated" << endl;
}

} /* namespace oist */

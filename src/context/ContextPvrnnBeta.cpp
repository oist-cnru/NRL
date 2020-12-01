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


#include "ContextPvrnnBeta.h"

namespace oist {

ContextPvrnnBeta::ContextPvrnnBeta(){

}

ContextPvrnnBeta::~ContextPvrnnBeta(){

}

string ContextPvrnnBeta::getClassName(){
	return string("ContextPvrnn");
}

void ContextPvrnnBeta::print(){

	cout << "dp_top: " << dp_top.transpose() << endl;
	cout << "dq_top: " << dq_top.transpose() << endl;

	cout << "g_h_next: " << g_h_next.transpose() << endl;
	cout << "g_hq_top: " << g_hq_top.transpose() << endl;
	cout << "g_dqloss: " << g_dqloss.transpose() << endl;

	cout << "dp_gen: " << dp_gen.transpose() << endl;

	for (int i = 0 ; i < t_dp.size() ; i++)
		for (int j = 0 ; j < t_dp[i].size() ; j++)
			cout << "t_dp[" << i << "][" << j << "] " <<  t_dp[i][j].transpose() << endl;

	for (int i = 0 ; i < t_dq.size() ; i++)
			for (int j = 0 ; j < t_dq[i].size() ; j++)
				cout << "t_dq[" << i << "][" << j << "] " <<  t_dq[i][j].transpose() << endl;

	for (int i = 0 ; i < t_kld.size() ; i++)
				for (int j = 0 ; j < t_kld[i].size() ; j++)
					cout << "t_kld[" << i << "][" << j << "] " <<  t_kld[i][j] << endl;

	for (int j = 0 ; j < e_dp.size() ; j++)
		cout << "e_dp[" << j << "] " <<  e_dp[j].transpose() << endl;

	for (int j = 0 ; j < e_dq.size() ; j++)
			cout << "e_dq[" << j << "] " <<  e_dq[j].transpose() << endl;

	for (int j = 0 ; j < e_kld.size() ; j++)
			cout << "e_kld[" << j << "] " <<  e_kld[j]<< endl;

}


} /* namespace oist */

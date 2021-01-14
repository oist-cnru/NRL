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


#ifndef SRC_PVRNN_NETWORK_BETA_H_
#define SRC_PVRNN_NETWORK_BETA_H_

#include "../includes.h"
#include "INetwork.h"

#include "../dataset/Dataset.h"
#include "../layer/ILayer.h"
#include "../layer/LayerPvrnnBeta.h"
#include "../context/ContextPvrnnBeta.h"

namespace oist {

/**
 * This class implements a network type PV-RNN
 * */
class NetworkPvrnnBeta : public INetwork {

	Dataset* dataset;

	int1DContainer d_num;
	int1DContainer z_num;
	int1DContainer o_num;
	int1DContainer tau;
	float1DContainer w1;
	float1DContainer w;

	vector<ILayer*> layers;
	ILayer* l0;
	ContextPvrnnBeta* l0_context;
	Utils* ut;

	vector<MatrixXf> Wdo;
	vector<MatrixXf> g_Wdo;
	vector<MatrixXf> m_Wdo;
	vector<MatrixXf> v_Wdo;
	vector<MatrixXf> Wdo_transpose;

	vector<VectorXf> Bo;
	vector<VectorXf> g_Bo;
	vector<VectorXf> m_Bo;
	vector<VectorXf> v_Bo;

	int prim_num;
	int prim_len;
	int layer_num;
	int state_dim;
	int o_dim;
	int l0_d_num;
	float rec_coef;
	float reg_coef;

	// Experiment mode

	int e_window_size;
	int e_num_times;
	int	e_cur_time;
	int e_prim_id;
	bool e_store_gen;
	bool e_store_inference;

public:


	/**
	 * Constructor
	 * @param paramMap Input map with layer information containers (number of d and z units, time constants, and the meta-parameters W)
	 * @param dataset Pointer to a data-set object
	 * */
	NetworkPvrnnBeta(map<string,float1DContainer>& paramMap, Dataset* dataset);
	~NetworkPvrnnBeta();

	int getNLayers();
	int getStateDim();

	void load(string);
	void save(string);
	void print();

	// ------------------------- training mode methods -------------------------

	void t_generate(int, int, vectorXf2DContainer&);
	void t_forward(int, int, vectorXf2DContainer&);
	void t_backward(int, vectorXf2DContainer&, vectorXf3DContainer&, float&, float&, float&);
	void t_optAdam(int, float, float, float);
	float getRecError(vectorXf2DContainer&, vectorXf3DContainer&);

	// ------------------------- Analysis mode methods -------------------------

	//void a_predict(int, float*, float*, string);
	void a_predict(int, float*, string);
	void a_feedForwardOutputFromContext(float*, float*);

	// ------------------------- Experiment mode methods -------------------------

	void e_enable(int, int, float*, int, bool, bool);
	void e_generate(float*);
	bool e_initForward();
	void e_forward(vectorXf2DContainer&);
	void e_backward(vectorXf2DContainer&, vectorXf2DContainer&, float&, float&, float&);
	void e_copyParam();
	void e_overwriteParam();
	void e_optAdam(int, float, float, float);
	void e_getState(float*);
	void e_save(string);

};

} /* namespace oist */
#endif /* SRC_PVRNN_NETWORK_BETA_H_ */

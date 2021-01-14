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


#ifndef SRC_LAYER_PVRNNBETA_H_
#define SRC_LAYER_PVRNNBETA_H_

#include "../context/ContextPvrnnBeta.h"
#include "../includes.h"
#include "../utils/Utils.h"
#include "../layer/ILayer.h"

namespace oist {

/**
 * This class implements a layer type PV-RNN
 * */
class LayerPvrnnBeta : public ILayer{

private:

	ContextPvrnnBeta* c;
	Utils* ut;

	int id;
	int d_num;
	int d_num_top;
	int z_num;
	int z_sum;
	int tau;
	int stateDim;
	bool top;
	bool bottom;
	float eps;
	float eps_top;
	float one_sub_eps;
	float w1_div_z_sum;
	float w_div_z_sum;
	int prim_num;
	int prim_len;
	float w1;
	float w;
	int gen_time_thres;

	// --- Parameters

	MatrixXf Wdh;
	MatrixXf Wzh;
	MatrixXf Wdh_top;
	MatrixXf Wdup;
	MatrixXf Wdlp;
	MatrixXf Wduq;
	MatrixXf Wdlq;
	VectorXf Bh;
	VectorXf Bup;
	VectorXf Blp;
	VectorXf Buq;
	VectorXf Blq;
	vectorXf2DContainer t_au;
	vectorXf2DContainer t_al;

	// auxiliary variables
	MatrixXf Wdh_top_transpose;


	//ArrayXf dp_gen; // declared in the context class
	ArrayXf hp_gen;
	ArrayXf up_gen;
	ArrayXf lp_gen;
	ArrayXf sp_gen;
	ArrayXf np_gen;
	ArrayXf zp_gen;

	// --- Gradients
	MatrixXf g_Wdh;
	MatrixXf g_Wzh;
//	MatrixXf g_Wdh_bottom;
	MatrixXf g_Wdh_top;
	MatrixXf g_Wdup;
	MatrixXf g_Wdlp;
	MatrixXf g_Wduq;
	MatrixXf g_Wdlq;
	VectorXf g_Bh;
	VectorXf g_Bup;
	VectorXf g_Blp;
	VectorXf g_Buq;
	VectorXf g_Blq;

	vectorXf1DContainer g_au;
	vectorXf1DContainer g_al;

	RowVectorXf g_uq_next_transpose;
	RowVectorXf g_lq_next_transpose;
	RowVectorXf g_up_next_transpose;
	RowVectorXf g_lp_next_transpose;

	// --- ADAM optimization

	MatrixXf m_Wdh;
	MatrixXf m_Wzh;
	MatrixXf m_Wdh_top;
	MatrixXf m_Wdup;
	MatrixXf m_Wdlp;
	MatrixXf m_Wduq;
	MatrixXf m_Wdlq;
	VectorXf m_Bh;
	VectorXf m_Bup;
	VectorXf m_Blp;
	VectorXf m_Buq;
	VectorXf m_Blq;

	vectorXf1DContainer m_au;
	vectorXf1DContainer m_al;

	MatrixXf v_Wdh;
	MatrixXf v_Wzh;
	MatrixXf v_Wdh_top;
	MatrixXf v_Wdup;
	MatrixXf v_Wdlp;
	MatrixXf v_Wduq;
	MatrixXf v_Wdlq;
	VectorXf v_Bh;
	VectorXf v_Bup;
	VectorXf v_Blp;
	VectorXf v_Buq;
	VectorXf v_Blq;

	vectorXf1DContainer v_au;
	vectorXf1DContainer v_al;

	ArrayXf up_bw_next;
	ArrayXf uq_bw_next;

	// iterators

	arrayXf1DContainer::reverse_iterator up_bw_next_i;
	arrayXf1DContainer::reverse_iterator uq_bw_next_i;

	arrayXf1DContainer::reverse_iterator up_bw_i;
	arrayXf1DContainer::reverse_iterator sp_bw_i;
	arrayXf1DContainer::reverse_iterator sq_bw_i;
	arrayXf1DContainer::reverse_iterator uq_bw_i;
	arrayXf1DContainer::reverse_iterator nq_bw_i;
	arrayXf1DContainer::reverse_iterator dq_bw_i;

	vectorXf1DContainer::reverse_iterator g_au_bw_i;

	// ------------ training mode data structures --------------------

	vectorXf1DContainer::reverse_iterator g_al_bw_i;

	//float2DContainer t_kld; // declared in the context class

	vectorXf2DContainer t_g_au;
	vectorXf2DContainer t_g_al;
	vectorXf2DContainer t_m_au;
	vectorXf2DContainer t_m_al;
	vectorXf2DContainer t_v_au;
	vectorXf2DContainer t_v_al;


	//arrayXf2DContainer t_dp; // declared in the context class
	arrayXf2DContainer t_hp;
	arrayXf2DContainer t_up;
	arrayXf2DContainer t_lp;
	arrayXf2DContainer t_sp;
	arrayXf2DContainer t_np;
	arrayXf2DContainer t_zp;

	//arrayXf2DContainer t_dq;  // declared in the context class
	arrayXf2DContainer t_hq;
	arrayXf2DContainer t_uq;
	arrayXf2DContainer t_lq;
	arrayXf2DContainer t_sq;
	arrayXf2DContainer t_nq;
	arrayXf2DContainer t_zq;

	// ------------ Experiment mode data structures  --------------------

	int e_window_size;
	int e_gen_time;
	int e_prim_id;
	int e_num_time;
	bool e_store_gen;
	bool e_store_inference;

	ArrayXf e_dq_opt;
	ArrayXf e_hq_opt;

	ArrayXf e_dq_tzero;
	ArrayXf e_hq_tzero;

	//arrayXf1DContainer e_dp; // declared in the context class
	arrayXf1DContainer e_hp;
	arrayXf1DContainer e_up;
	arrayXf1DContainer e_lp;
	arrayXf1DContainer e_sp;
	arrayXf1DContainer e_np;
	arrayXf1DContainer e_zp;

	//arrayXf1DContainer e_dq; // declared in the context class
	arrayXf1DContainer e_hq;
	arrayXf1DContainer e_uq;
	arrayXf1DContainer e_lq;
	arrayXf1DContainer e_sq;
	arrayXf1DContainer e_nq;
	arrayXf1DContainer e_zq;
	//float1DContainer e_kld; // declared in the context class

	vectorXf1DContainer e_au;
	vectorXf1DContainer e_al;
	vectorXf1DContainer::iterator e_au_i;
	vectorXf1DContainer::iterator e_al_i;

	//storages

	arrayXf1DContainer::iterator e_hp_gen_store_i;
	arrayXf1DContainer::iterator e_dp_gen_store_i;
	arrayXf1DContainer::iterator e_up_gen_store_i;
	arrayXf1DContainer::iterator e_lp_gen_store_i;
	arrayXf1DContainer::iterator e_sp_gen_store_i;
	arrayXf1DContainer::iterator e_np_gen_store_i;
	arrayXf1DContainer::iterator e_zp_gen_store_i;

	arrayXf1DContainer hp_gen_store;
	arrayXf1DContainer dp_gen_store;
	arrayXf1DContainer up_gen_store;
	arrayXf1DContainer lp_gen_store;
	arrayXf1DContainer sp_gen_store;
	arrayXf1DContainer np_gen_store;
	arrayXf1DContainer zp_gen_store;

	arrayXf1DContainer e_hp_store;
	arrayXf1DContainer e_dp_store;
	arrayXf1DContainer e_up_store;
	arrayXf1DContainer e_lp_store;
	arrayXf1DContainer e_sp_store;
	arrayXf1DContainer e_np_store;
	arrayXf1DContainer e_zp_store;

	arrayXf1DContainer e_hq_store;
	arrayXf1DContainer e_dq_store;
	arrayXf1DContainer e_uq_store;
	arrayXf1DContainer e_lq_store;
	arrayXf1DContainer e_sq_store;
	arrayXf1DContainer e_nq_store;
	arrayXf1DContainer e_zq_store;
	float1DContainer e_kld_store;

	vectorXf1DContainer e_au_store;
	vectorXf1DContainer e_al_store;

	// optimization parameters

	vectorXf1DContainer e_au_copy;
	vectorXf1DContainer e_al_copy;



	float get_kld(ArrayXf& _mp, ArrayXf& _sp, ArrayXf& _mq, ArrayXf& _sq);

	void free_memory();

public:

	/**
	 * Constructor
	 * @param id Layer id
	 * @param d_num Number Number of d units
	 * @param d_num_top Number of d units (top level)
	 * @param z_num Number of z units
	 * @param z_sum Sum of z units for all layers in the network
	 * @param tau Time constant
	 * @param tau_top Time constant (top level)
	 * @param prim_num Number of primitives
	 * @param prim_len Length of primitives
	 * @param w1 Meta-parameter w (t=1)
	 * @param w Meta-parameter w
	 * */
	LayerPvrnnBeta(int id, int d_num, int d_num_top, int z_num, int z_sum, int tau, int tau_top, int prim_num, int prim_len, float w1, float w);
	/**
	 * Destructor
	 * */
	~LayerPvrnnBeta();


	int getStateDim();

    // ------------------------- context methods

    void initContext(int);
    IContext* getContext();

    // ------------------------- training methods

    void t_generate(int, int);
	void t_forward(int, int);
	void t_initBackward();
	void t_backward(int, int);
	void t_optAdam(int, float, float, float);
	void load(string);
	void save(string);

	// ------------------------- Analysis methods

	void a_init(float*);
	void a_predict();
	void a_save(string);

	// ------------------------ Experiment methods

	void e_enable(int, int, float, int, bool, bool);
	void e_generate();
	void e_initForward();
	void e_forward();
	void e_initBackward();
	void e_backward(int);
	void e_copyParam();
	void e_overwriteParam();
	void e_optAdam(int, float, float, float);
	float* e_getState(float*);
	void e_save(string);

	// ------------------------- Debug methods

	void print();


};

} /* namespace oist */

#endif /* SRC_LAYER_PVRNNBETA_H_ */

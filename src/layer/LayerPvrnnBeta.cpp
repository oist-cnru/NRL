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

#include "../utils/Utils.h"
#include "LayerPvrnnBeta.h"

namespace oist {

LayerPvrnnBeta::LayerPvrnnBeta(int _id, int _d_num, int _d_num_top, int _z_num, int _z_sum, int _tau, int _tau_top, int _prim_num, int _prim_len, float _w1, float _w){

		ut = Utils::getInstance();
    	id = _id;
		d_num = _d_num;
		z_num = _z_num;
		d_num_top = _d_num_top;
		z_sum = _z_sum;
		tau = _tau;
		prim_num = _prim_num;
		prim_len = _prim_len;
		w1 = _w1;
		w = _w;

		w1_div_z_sum = w1/((float)z_sum*1.0);
		w_div_z_sum = w/((float)z_sum*1.0);

		stateDim = (d_num*2 + z_num*5)*2; // ((h,d) + (u, l, s, n, z))*(p,q)

		// time constant
		eps = 1.0/(tau*1.0);
		eps_top = (_tau_top > 0.0 ? 1.0/(_tau_top*1.0) : 0.0);
		one_sub_eps = 1.0 - eps;

		top = (d_num_top == 0);
		//bottom = (d_num_bottom == 0);
		bottom = (id == 0);

		c = new ContextPvrnnBeta();

		// initializing weight matrixes
		Wdh = ut->kaiming_uniform_initialization(d_num,d_num, Utils::nonlinearity::Linear);
		g_Wdh = MatrixXf::Zero(d_num,d_num);
		m_Wdh = MatrixXf::Zero(d_num,d_num);
		v_Wdh = MatrixXf::Zero(d_num,d_num);

		Bh  = ut->kaiming_uniform_initialization(d_num);
		g_Bh  = VectorXf::Zero(d_num);
		m_Bh = VectorXf::Zero(d_num);
		v_Bh = VectorXf::Zero(d_num);

		Wzh = ut->kaiming_uniform_initialization(d_num,z_num, Utils::nonlinearity::Linear);
		g_Wzh = MatrixXf::Zero(d_num,z_num);
		m_Wzh = MatrixXf::Zero(d_num,z_num);
		v_Wzh = MatrixXf::Zero(d_num,z_num);

		if (!top){
			Wdh_top = ut->kaiming_uniform_initialization(d_num,d_num_top, Utils::nonlinearity::Linear);
			g_Wdh_top = MatrixXf::Zero(d_num,d_num_top);
			m_Wdh_top = MatrixXf::Zero(d_num,d_num_top);
			v_Wdh_top = MatrixXf::Zero(d_num,d_num_top);
			c->dp_top = VectorXf::Zero(d_num_top);
			c->dq_top = VectorXf::Zero(d_num_top);
			Wdh_top_transpose = Wdh_top.transpose();
		}

		Wdup = ut->kaiming_uniform_initialization(z_num,d_num, Utils::nonlinearity::Tanh);
		g_Wdup = MatrixXf::Zero(z_num,d_num);
		m_Wdup = MatrixXf::Zero(z_num,d_num);
		v_Wdup = MatrixXf::Zero(z_num,d_num);

		Wdlp = ut->kaiming_uniform_initialization(z_num,d_num, Utils::nonlinearity::Linear);
		g_Wdlp = MatrixXf::Zero(z_num,d_num);
		m_Wdlp = MatrixXf::Zero(z_num,d_num);
		v_Wdlp = MatrixXf::Zero(z_num,d_num);

		Wduq = ut->kaiming_uniform_initialization(z_num,d_num, Utils::nonlinearity::Tanh);
		g_Wduq = MatrixXf::Zero(z_num,d_num);
		m_Wduq = MatrixXf::Zero(z_num,d_num);
		v_Wduq = MatrixXf::Zero(z_num,d_num);

		Wdlq = ut->kaiming_uniform_initialization(z_num,d_num, Utils::nonlinearity::Linear);
		g_Wdlq = MatrixXf::Zero(z_num,d_num);
		m_Wdlq = MatrixXf::Zero(z_num,d_num);
		v_Wdlq = MatrixXf::Zero(z_num,d_num);

		Bup = ut->kaiming_uniform_initialization(z_num);
		g_Bup = VectorXf::Zero(z_num);
		m_Bup = VectorXf::Zero(z_num);
		v_Bup = VectorXf::Zero(z_num);

		Blp = ut->kaiming_uniform_initialization(z_num);
		g_Blp = VectorXf::Zero(z_num);
		m_Blp = VectorXf::Zero(z_num);
		v_Blp = VectorXf::Zero(z_num);

		Buq = ut->kaiming_uniform_initialization(z_num);
		g_Buq = VectorXf::Zero(z_num);
		m_Buq = VectorXf::Zero(z_num);
		v_Buq = VectorXf::Zero(z_num);

		Blq = ut->kaiming_uniform_initialization(z_num);
		g_Blq = VectorXf::Zero(z_num);
		m_Blq = VectorXf::Zero(z_num);
		v_Blq = VectorXf::Zero(z_num);


		for (int i = 0; i < _prim_num ; i++){

			vectorXf1DContainer au, g_au, m_au, v_au, al, g_al, m_al, v_al;

			for (int j = 0; j < _prim_len ; j++){
				au.push_back(ut->kaiming_uniform_initialization(z_num));
				g_au.push_back(VectorXf::Zero(z_num));
				m_au.push_back(VectorXf::Zero(z_num));
				v_au.push_back(VectorXf::Zero(z_num));
				al.push_back(ut->kaiming_uniform_initialization(z_num));
				g_al.push_back(VectorXf::Zero(z_num));
				m_al.push_back(VectorXf::Zero(z_num));
				v_al.push_back(VectorXf::Zero(z_num));

			}
			t_au.push_back(au); t_g_au.push_back(g_au); t_m_au.push_back(m_au); t_v_au.push_back(v_au);
			t_al.push_back(al); t_g_al.push_back(g_al); t_m_al.push_back(m_al); t_v_al.push_back(v_al);
		}

		gen_time_thres = 3;
		e_window_size = 0;
		e_gen_time = 0;
		e_prim_id = 0;
		e_num_time = 0;

		c->g_hq_top = VectorXf::Zero(d_num_top);
		c->g_dqloss = VectorXf::Zero(d_num);

		c->dp_top = VectorXf::Zero(d_num_top);
		c->dq_top = VectorXf::Zero(d_num_top);

		c->g_h_next = VectorXf::Zero(d_num);
		g_up_next_transpose = RowVectorXf::Zero(z_num);
		g_lp_next_transpose = RowVectorXf::Zero(z_num);
		g_uq_next_transpose = RowVectorXf::Zero(z_num);
		g_lq_next_transpose = RowVectorXf::Zero(z_num);

		e_dq_opt = ArrayXf::Zero(d_num);
		e_hq_opt = ArrayXf::Zero(d_num);
		e_dq_tzero = ArrayXf::Zero(d_num);
		e_hq_tzero = ArrayXf::Zero(d_num);
		e_store_gen = false;
		e_store_inference = false;

		for (int p = 0; p < prim_num; p++){

			arrayXf1DContainer hp, dp, up, lp, sp, np, zp;

			hp.push_back(ArrayXf::Zero(d_num));
			dp.push_back(ArrayXf::Zero(d_num));
			up.push_back(ArrayXf::Zero(z_num));
			lp.push_back(ArrayXf::Zero(z_num));
			sp.push_back(ArrayXf::Zero(z_num));
			np.push_back(ArrayXf::Zero(z_num));
			zp.push_back(ArrayXf::Zero(z_num));

			t_hp.push_back(hp);
			c->t_dp.push_back(dp);
			t_up.push_back(up);
			t_lp.push_back(lp);
			t_sp.push_back(sp);
			t_np.push_back(np);
			t_zp.push_back(zp);

			arrayXf1DContainer hq, dq, uq, lq, sq, nq, zq;

			hq.push_back(ArrayXf::Zero(d_num));
			dq.push_back(ArrayXf::Zero(d_num));
			uq.push_back(ArrayXf::Zero(z_num));
			lq.push_back(ArrayXf::Zero(z_num));
			sq.push_back(ArrayXf::Zero(z_num));
			nq.push_back(ArrayXf::Zero(z_num));
			zq.push_back(ArrayXf::Zero(z_num));

			t_hq.push_back(hq);
			c->t_dq.push_back(dq);
			t_uq.push_back(uq);
			t_lq.push_back(lq);
			t_sq.push_back(sq);
			t_nq.push_back(nq);
			t_zq.push_back(zq);

			float1DContainer kld;
			kld.push_back(0.0);
			c->t_kld.push_back(kld);
		}

	}

	int LayerPvrnnBeta::getStateDim(){
		return stateDim;
	}

	IContext* LayerPvrnnBeta::getContext(){
		return c;
	}

	void LayerPvrnnBeta::initContext(int _prim_id){


		t_hp[_prim_id].clear();
		c->t_dp[_prim_id].clear();
		t_up[_prim_id].clear();
		t_lp[_prim_id].clear();
		t_sp[_prim_id].clear();
		t_np[_prim_id].clear();
		t_zp[_prim_id].clear();

		t_hq[_prim_id].clear();
		c->t_dq[_prim_id].clear();
		t_uq[_prim_id].clear();
		t_lq[_prim_id].clear();
		t_sq[_prim_id].clear();
		t_nq[_prim_id].clear();
		t_zq[_prim_id].clear();

		c->t_kld[_prim_id].clear();

		t_hp[_prim_id].push_back(ArrayXf::Zero(d_num));
		c->t_dp[_prim_id].push_back(ArrayXf::Zero(d_num));
		t_up[_prim_id].push_back(ArrayXf::Zero(z_num));
		t_lp[_prim_id].push_back(ArrayXf::Zero(z_num));
		t_sp[_prim_id].push_back(ArrayXf::Zero(z_num));
		t_np[_prim_id].push_back(ArrayXf::Zero(z_num));
		t_zp[_prim_id].push_back(ArrayXf::Zero(z_num));

		t_hq[_prim_id].push_back(ArrayXf::Zero(d_num));
		c->t_dq[_prim_id].push_back(ArrayXf::Zero(d_num));
		t_uq[_prim_id].push_back(ArrayXf::Zero(z_num));
		t_lq[_prim_id].push_back(ArrayXf::Zero(z_num));
		t_sq[_prim_id].push_back(ArrayXf::Zero(z_num));
		t_nq[_prim_id].push_back(ArrayXf::Zero(z_num));
		t_zq[_prim_id].push_back(ArrayXf::Zero(z_num));

		c->t_kld[_prim_id].push_back(0.0);

		}


	void LayerPvrnnBeta::t_generate(int _time, int _prim_id){

		 //generating from the prior distribution
		 VectorXf hp = t_hp[_prim_id].back();
		 VectorXf dp = c->t_dp[_prim_id].back();

		 ArrayXf up;
		 ArrayXf lp;

		 if (_time < gen_time_thres) {
			 up = Wduq*dp + Buq + t_au[_prim_id][_time];
			 lp = Wdlq*dp + Blq + t_al[_prim_id][_time];
		 }
		 else{
			 up = Wdup*dp + Bup;
			 lp = Wdlp*dp + Blp;
		 }

		 ut->tanH<ArrayXf>(&up);
		 ArrayXf sp = lp.exp();
		 ArrayXf np = ArrayXf::Zero(z_num);
		 ut->randN<ArrayXf>(&np);
		 VectorXf zp = up + sp*np;

		 hp = one_sub_eps*hp + eps*(Wdh*dp + Wzh*zp + Bh);

		 if (!top)
			 hp += eps*(Wdh_top*c->dp_top);

		 dp = hp;
		 ut->tanH<VectorXf>(&dp);

		 t_hp[_prim_id].push_back(hp);
		 c->t_dp[_prim_id].push_back(dp);
		 t_up[_prim_id].push_back(up);
		 t_lp[_prim_id].push_back(lp);
		 t_sp[_prim_id].push_back(sp);
		 t_np[_prim_id].push_back(np);
		 t_zp[_prim_id].push_back(zp);
	}

	inline float LayerPvrnnBeta::get_kld(ArrayXf& _up, ArrayXf& _sp, ArrayXf& _uq, ArrayXf& _sq){

		auto up = _up.data();
		auto sp = _sp.data();
		auto uq = _uq.data();
		auto sq = _sq.data();

		float kld = 0.0;
		for (int i = 0 ; i < z_num; i++, up++, sp++, uq++, sq++){
			kld += -log(*sq/ *sp) + (((*uq-*up)*(*uq-*up)) + (*sq)*(*sq))/(2.0* ((*sp)*(*sp))) - 0.5 ;
		}

		return kld;
	}

	void LayerPvrnnBeta::t_forward(int _time, int _prim_id){

		// --------------- generation from the prior distribution ---------------

		VectorXf hp = t_hp[_prim_id].back();
		VectorXf dp = c->t_dp[_prim_id].back();

		ArrayXf up;
		ArrayXf lp;
		if (_time == 0){
			// unit Gaussian distribution
			up = ArrayXf::Zero(z_num);
			lp = ArrayXf::Zero(z_num);
		}else{
			up = Wdup*dp + Bup;
			ut->tanH<ArrayXf>(&up);
			lp = Wdlp*dp + Blp;
		}
		ArrayXf sp = lp.exp();
		ArrayXf np = ArrayXf::Zero(z_num);
		ut->randN<ArrayXf>(&np);
		VectorXf zp = up + sp*np;

		hp = one_sub_eps*hp + eps*(Wdh*dp + Wzh*zp + Bh);

		// --------------- generation from the posterior distribution ---------------

		VectorXf hq = t_hq[_prim_id].back();
		VectorXf dq = c->t_dq[_prim_id].back();

		ArrayXf uq = Wduq*dq + Buq  + t_au[_prim_id][_time];
		ut->tanH<ArrayXf>(&uq);
		ArrayXf lq = Wdlq*dq + Blq + t_al[_prim_id][_time];
		ArrayXf sq = lq.exp();
		ArrayXf nq = ArrayXf::Zero(z_num);

		ut->randN<ArrayXf>(&nq);
		VectorXf zq = uq + sq*nq;

		hq = one_sub_eps*hq + eps*(Wdh*dq + Wzh*zq + Bh);

		if (!top){
			hp += eps*Wdh_top*c->dp_top;
			hq += eps*Wdh_top*c->dq_top;
		}

		dp = hp;
		ut->tanH<VectorXf>(&dp);

		dq = hq;
		ut->tanH<VectorXf>(&dq);

		float kld = get_kld(up, sp, uq, sq);

		t_hp[_prim_id].push_back(hp);
		c->t_dp[_prim_id].push_back(dp);
		t_up[_prim_id].push_back(up);
		t_lp[_prim_id].push_back(lp);
		t_sp[_prim_id].push_back(sp);
		t_np[_prim_id].push_back(np);
		t_zp[_prim_id].push_back(zp);

		t_hq[_prim_id].push_back(hq);
		c->t_dq[_prim_id].push_back(dq);
		t_uq[_prim_id].push_back(uq);
		t_lq[_prim_id].push_back(lq);
		t_sq[_prim_id].push_back(sq);
		t_nq[_prim_id].push_back(nq);
		t_zq[_prim_id].push_back(zq);

		c->t_kld[_prim_id].push_back(kld);


	 }

	 void LayerPvrnnBeta::t_initBackward(){

		 // Clearing state gradients

		ut->zero<VectorXf>(&c->g_h_next);
		ut->zero<VectorXf>(&c->g_hq_top);
		ut->zero<RowVectorXf>(&g_up_next_transpose);
		ut->zero<RowVectorXf>(&g_lp_next_transpose);
		ut->zero<RowVectorXf>(&g_uq_next_transpose);
		ut->zero<RowVectorXf>(&g_lq_next_transpose);
	 }

	 void LayerPvrnnBeta::t_backward(int _time, int _prim_id){

		ArrayXf up_next = ArrayXf::Zero(z_num);
		ArrayXf uq_next = ArrayXf::Zero(z_num);
		if (_time < prim_len){
			up_next = t_up[_prim_id][_time+1];
			uq_next = t_uq[_prim_id][_time+1];
		}

		ArrayXf up = t_up[_prim_id][_time];
		ArrayXf sp = t_sp[_prim_id][_time];
		ArrayXf sq = t_sq[_prim_id][_time];
		ArrayXf uq = t_uq[_prim_id][_time];
		ArrayXf nq = t_nq[_prim_id][_time];
		ArrayXf dq =  c->t_dq[_prim_id][_time];
		RowVectorXf zq = t_zq[_prim_id][_time];
		RowVectorXf dp_prev_transpose = c->t_dp[_prim_id][_time-1];
		RowVectorXf dq_prev_transpose = c->t_dq[_prim_id][_time-1];

		ArrayXf up_pow_2 = up.pow(2.0);
		ArrayXf sp_pow_2 = sp.pow(2.0)  +  NON_ZERO;
		ArrayXf uq_pow_2 = uq.pow(2.0);
		ArrayXf sq_pow_2 = sq.pow(2.0);

		VectorXf g_d = eps*c->g_h_next.transpose()*Wdh;

		if (bottom){
			g_d += c->g_dqloss;
		}

		if (!top){
			g_d += ((VectorXf)(eps_top*c->g_hq_top.transpose()*Wdh_top_transpose)).transpose();
		}

		 RowVectorXf g_uptanh_transpose = g_up_next_transpose.array()*(1.0 - up_next.pow(2.0).transpose());
		 RowVectorXf g_uqtanh_transpose = g_uq_next_transpose.array()*(1.0 - uq_next.pow(2.0).transpose());

		 g_d += g_uptanh_transpose*Wdup;
		 g_d += g_uqtanh_transpose*Wduq;
		 g_d += g_lp_next_transpose*Wdlp;
		 g_d += g_lq_next_transpose*Wdlq;


		 VectorXf g_h = g_d.array()*(1.0 - dq.pow(2.0)) + (one_sub_eps * c->g_h_next.array());
		 ArrayXf g_z = eps*g_h.transpose()*Wzh;

		 RowVectorXf g_up;
		 RowVectorXf g_lp;
		 RowVectorXf g_uq;
		 RowVectorXf g_lq;

		 float wFactor = w_div_z_sum;

		 if (_time == 1)
			 wFactor = w1_div_z_sum;

		 g_up = (wFactor*((up - uq)/sp_pow_2));
		 g_lp = wFactor*(1.0 - (((ArrayXf)(uq-up)).pow(2.0) + sq_pow_2)/sp_pow_2);
		 g_uq = (g_z + wFactor*((uq - up)/sp_pow_2));
		 g_lq = g_z*sq*nq + wFactor*(-1.0 + (sq_pow_2/sp_pow_2));

		 // Parameter gradients

		 g_Wdh += eps*g_h*dq_prev_transpose;
		 g_Bh += eps*g_h;


		 if (!top){
			 g_Wdh_top += eps * g_h * c->dq_top.transpose();
		 }

		 g_Wzh += eps* g_h* zq;

		 VectorXf g_uqtanh = g_uq.array().transpose()*(1.0 - uq_pow_2);

		 g_Wduq += g_uqtanh*dq_prev_transpose;
		 g_Buq += g_uqtanh;
		 g_Wdlq += g_lq.transpose()*dq_prev_transpose;
		 g_Blq += g_lq;

		 VectorXf g_uptanh = g_up.array().transpose()*(1.0 - up_pow_2);

		 g_Wdup += g_uptanh*dp_prev_transpose;
		 g_Bup += g_uptanh;
		 g_Wdlp += g_lp.transpose()*dp_prev_transpose;
		 g_Blp += g_lp;

		 t_g_au[_prim_id][_time-1] = g_uqtanh;
		 t_g_al[_prim_id][_time-1] = g_lq;

		 c->g_h_next = g_h;
		 g_up_next_transpose = g_up;
		 g_uq_next_transpose = g_uq;
		 g_lp_next_transpose = g_lp;
		 g_lq_next_transpose = g_lq;

	 }

	 void LayerPvrnnBeta::t_optAdam(int _epoch, float _alpha, float _beta1, float _beta2){

		ut->adam<MatrixXf>(&Wdh, &g_Wdh, &m_Wdh, &v_Wdh, _epoch, _alpha, _beta1, _beta2 );
		ut->adam<MatrixXf>(&Wzh, &g_Wzh, &m_Wzh, &v_Wzh, _epoch, _alpha, _beta1, _beta2 );

		ut->adam<MatrixXf>(&Wdup, &g_Wdup, &m_Wdup, &v_Wdup, _epoch, _alpha, _beta1, _beta2 );
		ut->adam<MatrixXf>(&Wdlp, &g_Wdlp, &m_Wdlp, &v_Wdlp, _epoch, _alpha, _beta1, _beta2 );
		ut->adam<MatrixXf>(&Wduq, &g_Wduq, &m_Wduq, &v_Wduq, _epoch, _alpha, _beta1, _beta2 );
		ut->adam<MatrixXf>(&Wdlq, &g_Wdlq, &m_Wdlq, &v_Wdlq, _epoch, _alpha, _beta1, _beta2 );

		ut->adam<VectorXf>(&Bh,   &g_Bh,   &m_Bh,   &v_Bh,   _epoch, _alpha, _beta1, _beta2 );
		ut->adam<VectorXf>(&Bup, &g_Bup, &m_Bup, &v_Bup, _epoch, _alpha, _beta1, _beta2 );
		ut->adam<VectorXf>(&Blp, &g_Blp, &m_Blp, &v_Blp, _epoch, _alpha, _beta1, _beta2 );
		ut->adam<VectorXf>(&Buq, &g_Buq, &m_Buq, &v_Buq, _epoch, _alpha, _beta1, _beta2 );
		ut->adam<VectorXf>(&Blq, &g_Blq, &m_Blq, &v_Blq, _epoch, _alpha, _beta1, _beta2 );

		// Clearing parameter gradients

		ut->zero<MatrixXf>(&g_Wdh);
		ut->zero<MatrixXf>(&g_Wzh);

		ut->zero<MatrixXf>(&g_Wdup);
		ut->zero<MatrixXf>(&g_Wdlp);
		ut->zero<MatrixXf>(&g_Wduq);
		ut->zero<MatrixXf>(&g_Wdlq);

		ut->zero<VectorXf>(&g_Bh);
		ut->zero<VectorXf>(&g_Bup);
		ut->zero<VectorXf>(&g_Blp);
		ut->zero<VectorXf>(&g_Buq);
		ut->zero<VectorXf>(&g_Blq);


		if (! top){
			ut->adam<MatrixXf>(&Wdh_top,  &g_Wdh_top,  &m_Wdh_top,  &v_Wdh_top,  _epoch, _alpha, _beta1, _beta2 );
			ut->zero<MatrixXf>(&g_Wdh_top);
			Wdh_top_transpose = Wdh_top.transpose();
		}

		for (int s = 0; s < prim_num ; s++){
			vectorXf1DContainer::iterator 	 au_i = t_au[s].begin();
			vectorXf1DContainer::iterator  g_au_i = t_g_au[s].begin();
			vectorXf1DContainer::iterator  m_au_i = t_m_au[s].begin();
			vectorXf1DContainer::iterator  v_au_i = t_v_au[s].begin();

			vectorXf1DContainer::iterator    al_i = t_al[s].begin();
			vectorXf1DContainer::iterator  g_al_i = t_g_al[s].begin();
			vectorXf1DContainer::iterator  m_al_i = t_m_al[s].begin();
			vectorXf1DContainer::iterator  v_al_i = t_v_al[s].begin();

			for (int t = 0; t < prim_len ; t++, au_i++, g_au_i++, m_au_i++, v_au_i++, al_i++, g_al_i++, m_al_i++, v_al_i++){
				ut->adam<VectorXf>(au_i.base(),   g_au_i.base(),   m_au_i.base(),   v_au_i.base(),   _epoch, _alpha, _beta1, _beta2 );
				ut->adam<VectorXf>(al_i.base(),   g_al_i.base(),   m_al_i.base(),   v_al_i.base(),   _epoch, _alpha, _beta1, _beta2 );
				ut->zero<VectorXf>(g_au_i.base());
				ut->zero<VectorXf>(g_al_i.base());

			}
		}
	 }

	 void LayerPvrnnBeta::print(){

		 vector<string> w_names;
		 vector<string> b_names;
		 vector<MatrixXf*> w_p;
		 vector<VectorXf*> b_p;

		 w_names.push_back(string("Wdh"));
		 w_names.push_back(string("Wzh"));
		 w_names.push_back(string("Wdup"));
		 w_names.push_back(string("Wdlp"));
		 w_names.push_back(string("Wduq"));
		 w_names.push_back(string("Wdlq"));

		 b_names.push_back(string("Bh"));
		 b_names.push_back(string("Bup"));
		 b_names.push_back(string("Blp"));
		 b_names.push_back(string("Buq"));
		 b_names.push_back(string("Blq"));

		 w_p.push_back(&Wdh);
		 w_p.push_back(&Wzh);
		 w_p.push_back(&Wdup);
		 w_p.push_back(&Wdlp);
		 w_p.push_back(&Wduq);
		 w_p.push_back(&Wdlp);

		 if (! top){
			w_p.push_back(&Wdh_top);
			w_names.push_back(string("Wdh_top"));
		 }

		 b_p.push_back(&Bh);
		 b_p.push_back(&Bup);
		 b_p.push_back(&Blp);
		 b_p.push_back(&Buq);
		 b_p.push_back(&Blq);

		 cout << "Layer " << id << endl;
		 for (int i = 0 ; i < (int)w_p.size() ; i++){
			cout << w_names[i] << endl;
			cout << "data: ";
			auto d = w_p[i]->data();
			for (int j = 0; j < w_p[i]->size(); j++, d++){
				cout << *d << " ";
			}
			cout << endl;
			cout << "size: [" << w_p[i]->rows() << "," << w_p[i]->cols() << "]"<< endl;

		 }

		 for (int i = 0 ; i < (int)b_p.size() ; i++){
			cout << b_names[i] << endl;
			cout << "data: ";
			auto d = b_p[i]->data();
			for (int j = 0; j < b_p[i]->size(); j++, d++){
				cout << *d << " ";
			}
			cout << endl;
			cout << "size: [" << b_p[i]->rows() << "," << b_p[i]->cols() << "]"<< endl;
		 }

		// au and al vectors
		for (int s = 0 ; s < prim_num; s++){
			vectorXf1DContainer au = t_au[s];
			for (int t = 0 ; t < prim_len; t++){
				cout << "au_" << s << "_" << t << endl;
				cout << "data: ";
				auto d = au[t].data();
				for (int j = 0; j < au[t].size(); j++, d++){
					cout << *d << " ";
				}
				cout << endl;
				cout << "size: [" << au[t].rows() << "," << au[t].cols() << "]"<< endl;

			}
			vectorXf1DContainer al = t_al[s];
			for (int t = 0 ; t < prim_len; t++){
				cout << "al_" << s << "_" << t << endl;
				cout << "data: ";
				auto d = al[t].data();
				for (int j = 0; j < al[t].size(); j++, d++){
					cout << *d << " ";
				}
				cout << endl;
				cout << "size: [" << al[t].rows() << "," << al[t].cols() << "]"<< endl;

			}

		}

	}

	void LayerPvrnnBeta::load(string _path){

	 	vector<MatrixXf*> w_p, w_m, w_v;
	 	vector<VectorXf*> b_p, b_m, b_v;

	 	std::string delimiter = ut->getDelimiter();

	 	w_p.push_back(&Wdh);	w_p.push_back(&Wzh);	w_p.push_back(&Wdup);	w_p.push_back(&Wdlp);	w_p.push_back(&Wduq);	w_p.push_back(&Wdlq);
		w_m.push_back(&m_Wdh);	w_m.push_back(&m_Wzh);	w_m.push_back(&m_Wdup); w_m.push_back(&m_Wdlp); w_m.push_back(&m_Wduq);	w_m.push_back(&m_Wdlq);
		w_v.push_back(&v_Wdh);	w_v.push_back(&v_Wzh);	w_v.push_back(&v_Wdup); w_v.push_back(&v_Wdlp); w_v.push_back(&v_Wduq);	w_v.push_back(&v_Wdlq);
		b_p.push_back(&Bh);	b_p.push_back(&Bup);	b_p.push_back(&Blp);	b_p.push_back(&Buq);	b_p.push_back(&Blq);
		b_m.push_back(&m_Bh);	b_m.push_back(&m_Bup);	b_m.push_back(&m_Blp);	b_m.push_back(&m_Buq);	b_m.push_back(&m_Blq);
		b_v.push_back(&v_Bh);	b_v.push_back(&v_Bup);	b_v.push_back(&v_Blp);	b_v.push_back(&v_Buq);	b_v.push_back(&v_Blq);


		if (! top){
			w_p.push_back(&Wdh_top);  	w_m.push_back(&m_Wdh_top);		w_v.push_back(&v_Wdh_top);
		}

	 	stringstream strmWp, strmWm, strmWv;

	 	strmWp << _path << "/L" << id << "_w_p.d";
	 	strmWm << _path << "/L" << id << "_w_m.d";
	 	strmWv << _path << "/L" << id << "_w_v.d";

	 	ifstream wFile(strmWp.str()); ifstream m_wFile(strmWm.str());	ifstream v_wFile(strmWv.str());

	 	stringstream strmBp, strmBm, strmBv;

	 	strmBp << _path << "/L" << id << "_b_p.d";
	 	strmBm << _path << "/L" << id << "_b_m.d";
	 	strmBv << _path << "/L" << id << "_b_v.d";

	 	ifstream bFile(strmBp.str()); ifstream m_bFile(strmBm.str());	ifstream v_bFile(strmBv.str());

	 	stringstream strmAup, strmAum, strmAuv;

	 	strmAup << _path << "/L" << id << "_au_p.d";
		strmAum << _path << "/L" << id << "_au_m.d";
		strmAuv << _path << "/L" << id << "_au_v.d";

		ifstream AuFile(strmAup.str()); ifstream m_AuFile(strmAum.str());	ifstream v_AuFile(strmAuv.str());

		stringstream strmAlp, strmAlm, strmAlv;

		strmAlp << _path << "/L" << id << "_al_p.d";
		strmAlm << _path << "/L" << id << "_al_m.d";
		strmAlv << _path << "/L" << id << "_al_v.d";

		ifstream AlFile(strmAlp.str()); ifstream m_AlFile(strmAlm.str());	ifstream v_AlFile(strmAlv.str());

	 	ifstream* buff_wFile[] =   {&wFile,   &m_wFile,   &v_wFile};
	 	ifstream* buff_bFile[] =   {&bFile,   &m_bFile,   &v_bFile};

	 	if (wFile.is_open()   && m_wFile.is_open()   && v_wFile.is_open() 	&&
	 		bFile.is_open()   && m_bFile.is_open()   && v_bFile.is_open() 	&&
			AuFile.is_open() && m_AuFile.is_open() && v_AuFile.is_open() &&
			AlFile.is_open() && m_AlFile.is_open() && v_AlFile.is_open() ){

			// loading the weight matrices
			for (int i = 0 ; i < (int)w_p.size() ; i++){

				MatrixXf* buff [] = {w_p[i], w_m[i], w_v[i]};
				for (int k = 0 ; k < 3 ; k++){
					try{
						ut->loadEigen<MatrixXf>(buff_wFile[k], buff[k], delimiter);
					}catch(oist::Exception& _e){
						stringstream stream;
						stream << "Unsuccessful loading of L" << id << " Weight matrices, msg[" << _e.what() << "]" << endl;
						throw oist::Exception(stream.str());
					}
				}
			}

			// loading the bias vectors
			for (int i = 0 ; i < (int)b_p.size() ; i++){

				VectorXf* buff [] = {b_p[i], b_m[i], b_v[i]};
				for (int k = 0 ; k < 3 ; k++){
					try{
						ut->loadEigen<VectorXf>(buff_bFile[k], buff[k], delimiter);
					}catch(oist::Exception& _e){
						stringstream stream;
						stream << "Unsuccessful loading of L" << id << " Bias vectors, msg[" << _e.what() << "]" << endl;
						throw oist::Exception(stream.str());
					}
				}
			}

			// loading the au and al vectors
			for (int s = 0 ; s < prim_num; s++){
				for (int t = 0 ; t < prim_len; t++){

					try{
						ut->loadEigen<VectorXf>(&AuFile, &t_au[s][t], delimiter);
						ut->loadEigen<VectorXf>(&AlFile, &t_al[s][t], delimiter);
						ut->loadEigen<VectorXf>(&m_AuFile, &t_m_au[s][t], delimiter);
						ut->loadEigen<VectorXf>(&m_AlFile, &t_m_al[s][t], delimiter);
						ut->loadEigen<VectorXf>(&v_AuFile, &t_v_au[s][t], delimiter);
						ut->loadEigen<VectorXf>(&v_AlFile, &t_v_al[s][t], delimiter);
					}
					catch(oist::Exception& _e){
						stringstream stream;
						stream << "Unsuccessful loading of L" << id << " A vectors, msg[" << _e.what() << "]" << endl;
						throw oist::Exception(stream.str());
					}
				}
			}

			// loading auxiliary matrices
			if (! top){
				Wdh_top_transpose = Wdh_top.transpose();
			}

			wFile.close();		m_wFile.close();	v_wFile.close();
			bFile.close();		m_bFile.close();	v_bFile.close();
			AuFile.close(); 	m_AuFile.close(); 	v_AuFile.close();
			AlFile.close(); 	m_AlFile.close();   v_AlFile.close();
	 	}
	 	else{
	 		throw oist::Exception("Error while loading the parameters");
	 	}

	 }

	 void LayerPvrnnBeta::save(string _path){

		 	vector<MatrixXf*> w_p, w_m, w_v;
			vector<VectorXf*> b_p, b_m, b_v;

			std::string delimiter = ut->getDelimiter();

			w_p.push_back(&Wdh);	w_p.push_back(&Wzh);	w_p.push_back(&Wdup);	w_p.push_back(&Wdlp);	w_p.push_back(&Wduq);	w_p.push_back(&Wdlq);
			w_m.push_back(&m_Wdh);	w_m.push_back(&m_Wzh);	w_m.push_back(&m_Wdup); w_m.push_back(&m_Wdlp); w_m.push_back(&m_Wduq);	w_m.push_back(&m_Wdlq);
			w_v.push_back(&v_Wdh);	w_v.push_back(&v_Wzh);	w_v.push_back(&v_Wdup); w_v.push_back(&v_Wdlp); w_v.push_back(&v_Wduq);	w_v.push_back(&v_Wdlq);
			b_p.push_back(&Bh);	b_p.push_back(&Bup);	b_p.push_back(&Blp);	b_p.push_back(&Buq);	b_p.push_back(&Blq);
			b_m.push_back(&m_Bh);	b_m.push_back(&m_Bup);	b_m.push_back(&m_Blp);	b_m.push_back(&m_Buq);	b_m.push_back(&m_Blq);
			b_v.push_back(&v_Bh);	b_v.push_back(&v_Bup);	b_v.push_back(&v_Blp);	b_v.push_back(&v_Buq);	b_v.push_back(&v_Blq);

			if (!top){
				w_p.push_back(&Wdh_top); w_m.push_back(&m_Wdh_top); w_v.push_back(&v_Wdh_top);
			}

		 	stringstream strmWp, strmWm, strmWv;

		 	strmWp << _path << "/L" << id << "_w_p.d";
			strmWm << _path << "/L" << id << "_w_m.d";
			strmWv << _path << "/L" << id << "_w_v.d";

			ofstream wFile(strmWp.str(),std::ofstream::out); ofstream m_wFile(strmWm.str(),std::ofstream::out);	ofstream v_wFile(strmWv.str(),std::ofstream::out);

			stringstream strmBp, strmBm, strmBv;

			strmBp << _path << "/L" << id << "_b_p.d";
			strmBm << _path << "/L" << id << "_b_m.d";
			strmBv << _path << "/L" << id << "_b_v.d";

			ofstream bFile(strmBp.str(),std::ofstream::out); ofstream m_bFile(strmBm.str(),std::ofstream::out);	ofstream v_bFile(strmBv.str(),std::ofstream::out);

			stringstream strmAup, strmAum, strmAuv;

			strmAup << _path << "/L" << id << "_au_p.d";
			strmAum << _path << "/L" << id << "_au_m.d";
			strmAuv << _path << "/L" << id << "_au_v.d";

			ofstream AuFile(strmAup.str(),std::ofstream::out); ofstream m_AuFile(strmAum.str(),std::ofstream::out);	ofstream v_AuFile(strmAuv.str(),std::ofstream::out);

			stringstream strmAlp, strmAlm, strmAlv;

			strmAlp << _path << "/L" << id << "_al_p.d";
			strmAlm << _path << "/L" << id << "_al_m.d";
			strmAlv << _path << "/L" << id << "_al_v.d";

			ofstream AlFile(strmAlp.str(),std::ofstream::out); ofstream m_AlFile(strmAlm.str(),std::ofstream::out);	ofstream v_AlFile(strmAlv.str(),std::ofstream::out);

	 		ofstream* buff_wFile[] = 	{&wFile,   &m_wFile,   &v_wFile};
	 		ofstream* buff_bFile[] = 	{&bFile,   &m_bFile,   &v_bFile};

	 		if (wFile.is_open()   && m_wFile.is_open()   && v_wFile.is_open()   &&
	 			bFile.is_open()   && m_bFile.is_open()   && v_bFile.is_open()   &&
				AuFile.is_open() && m_AuFile.is_open() && v_AuFile.is_open() &&
				AlFile.is_open() && m_AlFile.is_open() && v_AlFile.is_open() ){

	 			// storing the weight matrices
	 			for (int i = 0 ; i < (int)w_p.size() ; i++){

	 				MatrixXf* buff [] = {w_p[i], w_m[i], w_v[i]};
	 				for (int k = 0 ; k < 3 ; k++){
	 					ut->saveEigen<MatrixXf>(buff_wFile[k], buff[k], delimiter);
	 				}
	 			}

	 			// storing the bias vectors
	 			for (int i = 0 ; i < (int)b_p.size() ; i++){

	 				VectorXf* buff [] = {b_p[i], b_m[i], b_v[i]};
	 				for (int k = 0 ; k < 3 ; k++){
	 					ut->saveEigen<VectorXf>(buff_bFile[k], buff[k], delimiter);
	 				}
	 			}

	 			// saving the AMu and ALs vectors
	 			for (int s = 0 ; s < prim_num; s++){
					for (int t = 0 ; t < prim_len; t++){

						try{
							ut->saveEigen<VectorXf>(&AuFile, &t_au[s][t], delimiter);
							ut->saveEigen<VectorXf>(&AlFile, &t_al[s][t], delimiter);
							ut->saveEigen<VectorXf>(&m_AuFile, &t_m_au[s][t], delimiter);
							ut->saveEigen<VectorXf>(&m_AlFile, &t_m_al[s][t], delimiter);
							ut->saveEigen<VectorXf>(&v_AuFile, &t_v_au[s][t], delimiter);
							ut->saveEigen<VectorXf>(&v_AlFile, &t_v_al[s][t], delimiter);
						}
						catch(oist::Exception& _e){
							stringstream stream;
							stream << "Unsuccessful loading of L" << id << " A vectors, msg[" << _e.what() << "]" << endl;
							throw oist::Exception(stream.str());
						}
					}
				}

	 			wFile.close();	 m_wFile.close(); v_wFile.close();
	 			bFile.close();	 m_bFile.close(); v_bFile.close();
	 			AuFile.close(); m_AuFile.close(); v_AuFile.close();
	 			AlFile.close(); m_AlFile.close(); v_AlFile.close();
	 			}

	 		else{
	 			throw oist::Exception("Error while saving the parameters");
	 		}
	 }

	 void LayerPvrnnBeta::free_memory(){

		// memory from training

		if (t_hp.size()>0){

			 for (int s = 0; s < prim_num ; s++){
				t_hp[s].clear();
				c->t_dp[s].clear();
				t_up[s].clear();
				t_lp[s].clear();
				t_sp[s].clear();
				t_np[s].clear();
				t_zp[s].clear();

				t_hq[s].clear();
				c->t_dq[s].clear();
				t_uq[s].clear();
				t_lq[s].clear();
				t_sq[s].clear();
				t_nq[s].clear();
				t_zq[s].clear();

				c->t_kld[s].clear();
			}
		}

		// memory for post-diction

		e_hp.clear();
		c->e_dp.clear();
		e_up.clear();
		e_lp.clear();
		e_sp.clear();
		e_np.clear();
		e_zp.clear();

		e_hq.clear();
		c->e_dq.clear();
		e_uq.clear();
		e_lq.clear();
		e_sq.clear();
		e_nq.clear();
		e_zq.clear();
		c->e_kld.clear();

		e_au.clear();
		e_al.clear();

		// memory for ADAM optimization

		e_au_copy.clear();
		g_au.clear();
		m_au.clear();
		v_au.clear();

		e_al_copy.clear();
		g_al.clear();
		m_al.clear();
		v_al.clear();

		// memory for storing

		hp_gen_store.clear();
		dp_gen_store.clear();
		up_gen_store.clear();
		lp_gen_store.clear();
		sp_gen_store.clear();
		np_gen_store.clear();
		zp_gen_store.clear();
		e_hp_store.clear();
		e_dp_store.clear();
		e_up_store.clear();
		e_lp_store.clear();
		e_sp_store.clear();
		e_np_store.clear();
		e_zp_store.clear();

		e_hq_store.clear();
		e_dq_store.clear();
		e_uq_store.clear();
		e_lq_store.clear();
		e_sq_store.clear();
		e_nq_store.clear();
		e_zq_store.clear();
		e_kld_store.clear();

		e_au_store.clear();
		e_al_store.clear();

	 }

	 // ------------------------- Experiment mode methods -------------------------

	 void LayerPvrnnBeta::e_enable(int _prim_id, int _e_window_size, float _w, int _e_num_times, bool _store_gen, bool _store_inf){

		e_prim_id = _prim_id;
		e_window_size = _e_window_size;
		gen_time_thres = 3;
		e_gen_time = 0;
		e_num_time = _e_num_times;
		w = _w;
		e_store_gen = _store_gen;
		e_store_inference = _store_inf;

		w_div_z_sum = w/((float)z_sum*1.0);

		free_memory();

		try{
			// allocate memory for error regression

			c->dp_gen = ArrayXf::Zero(d_num);
			hp_gen = ArrayXf::Zero(d_num);
			up_gen = ArrayXf::Zero(z_num);
			lp_gen = ArrayXf::Zero(z_num);
			sp_gen = ArrayXf::Zero(z_num);
			np_gen = ArrayXf::Zero(z_num);
			zp_gen = ArrayXf::Zero(z_num);

			if (!top){
				c->dp_top = VectorXf::Zero(d_num_top);
				c->dq_top = VectorXf::Zero(d_num_top);
			}

			int nTimes = e_num_time;

			if (e_store_gen){

				for (int t = 0; t < nTimes; t++){

					hp_gen_store.push_back(ArrayXf::Zero(d_num));
					dp_gen_store.push_back(ArrayXf::Zero(d_num));
					up_gen_store.push_back(ArrayXf::Zero(z_num));
					lp_gen_store.push_back(ArrayXf::Zero(z_num));
					sp_gen_store.push_back(ArrayXf::Zero(z_num));
					np_gen_store.push_back(ArrayXf::Zero(z_num));
					zp_gen_store.push_back(ArrayXf::Zero(z_num));
				}

				e_hp_gen_store_i = hp_gen_store.begin();
				e_dp_gen_store_i = dp_gen_store.begin();
				e_up_gen_store_i = up_gen_store.begin();
				e_lp_gen_store_i = lp_gen_store.begin();
				e_sp_gen_store_i = sp_gen_store.begin();
				e_np_gen_store_i = np_gen_store.begin();
				e_zp_gen_store_i = zp_gen_store.begin();
			}

			for (int i = 0; i < e_window_size ; i++){

				e_au.push_back(VectorXf::Zero(z_num)); 		e_al.push_back(VectorXf::Zero(z_num));
				e_au_copy.push_back(VectorXf::Zero(z_num)); e_al_copy.push_back(VectorXf::Zero(z_num));
				g_au.push_back(VectorXf::Zero(z_num)); 		g_al.push_back(VectorXf::Zero(z_num));
				m_au.push_back(VectorXf::Zero(z_num)); 		m_al.push_back(VectorXf::Zero(z_num));
				v_au.push_back(VectorXf::Zero(z_num)); 		v_al.push_back(VectorXf::Zero(z_num));

			}

		}catch(bad_alloc& _e){
			cout << "Error: " << _e.what();
			stringstream stream;
			stream << "State vector could not be allocated for experiment time: " << e_num_time << endl;
			throw Exception(stream.str());
		}catch(...){
			stringstream stream;
			stream << "Unknown exception. State vector could not be allocated for experiment time: " << e_num_time << endl;
			throw Exception(stream.str());
		}

	}

	void LayerPvrnnBeta::e_initForward(){

		ArrayXf h0 = e_hq_tzero;
		ArrayXf d0 = e_dq_tzero;

		e_hp.clear();
		c->e_dp.clear();
		e_up.clear();
		e_lp.clear();
		e_sp.clear();
		e_np.clear();
		e_zp.clear();
		e_hq.clear();
		c->e_dq.clear();
		e_uq.clear();
		e_lq.clear();
		e_sq.clear();
		e_nq.clear();
		e_zq.clear();
		c->e_kld.clear();

		e_hp.push_back(h0);
		c->e_dp.push_back(d0);

		e_up.push_back(ArrayXf::Zero(z_num));
		e_lp.push_back(ArrayXf::Zero(z_num));
		e_sp.push_back(ArrayXf::Zero(z_num));
		e_np.push_back(ArrayXf::Zero(z_num));
		e_zp.push_back(ArrayXf::Zero(z_num));

		e_hq.push_back(h0);
		c->e_dq.push_back(d0);

		e_uq.push_back(ArrayXf::Zero(z_num));
		e_lq.push_back(ArrayXf::Zero(z_num));
		e_sq.push_back(ArrayXf::Zero(z_num));
		e_nq.push_back(ArrayXf::Zero(z_num));
		e_zq.push_back(ArrayXf::Zero(z_num));

		c->e_kld.push_back(0.0);

		e_au_i = e_au.begin();
		e_al_i = e_al.begin();


	}

	void LayerPvrnnBeta::e_initBackward(){
		t_initBackward();

		up_bw_next = ArrayXf::Zero(z_num);
		uq_bw_next = ArrayXf::Zero(z_num);

		up_bw_next_i = e_up.rbegin();
		uq_bw_next_i = e_uq.rbegin();

		up_bw_i = e_up.rbegin();
		sp_bw_i = e_sp.rbegin();
		sq_bw_i = e_sq.rbegin();
		uq_bw_i = e_uq.rbegin();
		nq_bw_i = e_nq.rbegin();
		dq_bw_i = c->e_dq.rbegin();

		g_au_bw_i = g_au.rbegin();
		g_al_bw_i = g_al.rbegin();

	}

	void LayerPvrnnBeta::e_generate(){

		 //generating from the prior distribution
		 VectorXf hp = hp_gen;
		 VectorXf dp = c->dp_gen;

		 ArrayXf up;
		 ArrayXf lp;

		 if (e_gen_time < gen_time_thres ) {
			 up = Wduq*dp + Buq + t_au[e_prim_id][e_gen_time];
			 lp = Wdlq*dp + Blq + t_al[e_prim_id][e_gen_time];
		 }
		 else{
			 up = Wdup*dp + Bup;
			 lp = Wdlp*dp + Blp;
		 }
		 e_gen_time += 1;

		 ut->tanH<ArrayXf>(&up);
		 ArrayXf sp = lp.exp();
		 ArrayXf np = ArrayXf::Zero(z_num);
		 ut->randN<ArrayXf>(&np);
		 VectorXf zp = up + sp*np;

		 hp = one_sub_eps*hp + eps*(Wdh*dp + Wzh*zp + Bh);

		 if (!top)
			 hp += eps*(Wdh_top*c->dp_top);

		 dp = hp;
		 ut->tanH<VectorXf>(&dp);

		 if (e_store_gen == true){
			 *(e_hp_gen_store_i++) = hp;
			 *(e_dp_gen_store_i++) = dp;
			 *(e_up_gen_store_i++) = up;
			 *(e_lp_gen_store_i++) = lp;
			 *(e_sp_gen_store_i++) = sp;
			 *(e_np_gen_store_i++) = np;
			 *(e_zp_gen_store_i++) = zp;
		 }

		 hp_gen = hp;
		 c->dp_gen = dp;
		 up_gen = up;
		 lp_gen = lp;
		 sp_gen = sp;
		 np_gen = np;
		 zp_gen = zp;

	}

	void LayerPvrnnBeta::e_forward(){

		// --------------- generating the prior distribution ---------------
		VectorXf hp = e_hp.back();
		VectorXf dp = c->e_dp.back();

		ArrayXf up = Wdup*dp + Bup;
		ut->tanH<ArrayXf>(&up);
		ArrayXf lp = Wdlp*dp + Blp;
		ArrayXf sp = lp.exp();
		ArrayXf np = ArrayXf::Zero(z_num);
		ut->randN<ArrayXf>(&np);
		VectorXf zp = up + sp*np;

		hp = one_sub_eps*hp + eps*(Wdh*dp + Wzh*zp + Bh);

		// --------------- generating the posterior distribution ---------------
		VectorXf hq = e_hq.back();
		VectorXf dq = c->e_dq.back();

		ArrayXf uq = Wduq*dq + Buq  + *(e_au_i++);
		ut->tanH<ArrayXf>(&uq);
		ArrayXf lq = Wdlq*dq + Blq + *(e_al_i++);
		ArrayXf sq = lq.exp();
		ArrayXf nq = ArrayXf::Zero(z_num);

		ut->randN<ArrayXf>(&nq);
		VectorXf zq = uq + sq*nq;

		hq = one_sub_eps*hq + eps*(Wdh*dq + Wzh*zq + Bh);

		if (!top){
			hp += eps*Wdh_top*c->dp_top;
			hq += eps*Wdh_top*c->dq_top;
		}

		dp = hp;
		ut->tanH<VectorXf>(&dp);

		dq = hq;
		ut->tanH<VectorXf>(&dq);

		float kld = get_kld(up, sp, uq, sq);

		e_hp.push_back(hp);
		c->e_dp.push_back(dp);
		e_up.push_back(up);
		e_lp.push_back(lp);
		e_sp.push_back(sp);
		e_np.push_back(np);
		e_zp.push_back(zp);

		e_hq.push_back(hq);
		c->e_dq.push_back(dq);
		e_uq.push_back(uq);
		e_lq.push_back(lq);
		e_sq.push_back(sq);
		e_nq.push_back(nq);
		e_zq.push_back(zq);

		c->e_kld.push_back(kld);

	 }

	void LayerPvrnnBeta::e_backward(int _time){

		ArrayXf up_next = ArrayXf::Zero(z_num);
		ArrayXf uq_next = ArrayXf::Zero(z_num);

		if (_time < e_window_size){
			up_next = *(up_bw_next_i++);
			uq_next = *(uq_bw_next_i++);
		}

		ArrayXf up = *(up_bw_i++);
		ArrayXf sp = *(sp_bw_i++);
		ArrayXf sq = *(sq_bw_i++);
		ArrayXf uq = *(uq_bw_i++);
		ArrayXf nq = *(nq_bw_i++);
		ArrayXf dq = *(dq_bw_i++);

		ArrayXf sp_pow_2 = sp.pow(2.0)  +  NON_ZERO;
		ArrayXf sq_pow_2 = sq.pow(2.0);
		ArrayXf uq_pow_2 = uq.pow(2.0);


		VectorXf g_d = eps*c->g_h_next.transpose()*Wdh;

		if (bottom){
			g_d +=  c->g_dqloss;
		}

		if (!top){
			g_d += ((VectorXf)(eps_top*c->g_hq_top.transpose()*Wdh_top_transpose)).transpose();
		}

		RowVectorXf g_uptanh_transpose = g_up_next_transpose.array()*(1.0 - up_next.pow(2.0).transpose());
		RowVectorXf g_uqtanh_transpose = g_uq_next_transpose.array()*(1.0 - uq_next.pow(2.0).transpose());

		g_d += g_uptanh_transpose*Wdup;
		g_d += g_uqtanh_transpose*Wduq;
		g_d += g_lp_next_transpose*Wdlp;
		g_d += g_lq_next_transpose*Wdlq;

		VectorXf g_h = g_d.array()*(1.0 - dq.pow(2.0)) + (one_sub_eps * c->g_h_next.array());
		ArrayXf g_z = eps*g_h.transpose()*Wzh;

		RowVectorXf g_up = (w_div_z_sum*((up - uq)/sp_pow_2));
		RowVectorXf g_lp = w_div_z_sum*(1.0 - (((ArrayXf)(uq-up)).pow(2.0) + sq_pow_2)/sp_pow_2);
		RowVectorXf g_uq = (g_z + w_div_z_sum*((uq - up)/sp_pow_2));
		RowVectorXf g_lq = g_z*sq*nq + w_div_z_sum*(-1.0 + (sq_pow_2/sp_pow_2));

		VectorXf g_uqtanh = g_uq.array().transpose()*(1.0 - uq_pow_2);

		*(g_au_bw_i++) = g_uqtanh;
		*(g_al_bw_i++) = g_lq;

		c->g_h_next = g_h;
		g_up_next_transpose = g_up;
		g_uq_next_transpose = g_uq;
		g_lp_next_transpose = g_lp;
		g_lq_next_transpose = g_lq;

	}

	void LayerPvrnnBeta::e_optAdam(int _epoch, float _alpha, float _beta1, float _beta2){


		vectorXf1DContainer::iterator 	 au_i = e_au.begin();
		vectorXf1DContainer::iterator  g_au_i = g_au.begin();
		vectorXf1DContainer::iterator  m_au_i = m_au.begin();
		vectorXf1DContainer::iterator  v_au_i = v_au.begin();

		vectorXf1DContainer::iterator    al_i = e_al.begin();
		vectorXf1DContainer::iterator  g_al_i = g_al.begin();
		vectorXf1DContainer::iterator  m_al_i = m_al.begin();
		vectorXf1DContainer::iterator  v_al_i = v_al.begin();

		for (int t = 0; t < e_window_size ; t++, au_i++, g_au_i++, m_au_i++, v_au_i++, al_i++, g_al_i++, m_al_i++, v_al_i++){

			ut->adam<VectorXf>(au_i.base(),   g_au_i.base(),   m_au_i.base(),   v_au_i.base(),   _epoch, _alpha, _beta1, _beta2 );
			ut->adam<VectorXf>(al_i.base(),   g_al_i.base(),   m_al_i.base(),   v_al_i.base(),   _epoch, _alpha, _beta1, _beta2 );

			ut->zero<VectorXf>(g_au_i.base());
			ut->zero<VectorXf>(g_al_i.base());
		}

		if (e_store_inference == true){
			arrayXf1DContainer::iterator hp_i = e_hp.begin();
			arrayXf1DContainer::iterator dp_i = c->e_dp.begin();
			arrayXf1DContainer::iterator up_i = e_up.begin();
			arrayXf1DContainer::iterator lp_i = e_lp.begin();
			arrayXf1DContainer::iterator sp_i = e_sp.begin();
			arrayXf1DContainer::iterator np_i = e_np.begin();
			arrayXf1DContainer::iterator zp_i = e_zp.begin();

			arrayXf1DContainer::iterator hq_i = e_hq.begin();
			arrayXf1DContainer::iterator dq_i = c->e_dq.begin();
			arrayXf1DContainer::iterator uq_i = e_uq.begin();
			arrayXf1DContainer::iterator lq_i = e_lq.begin();
			arrayXf1DContainer::iterator sq_i = e_sq.begin();
			arrayXf1DContainer::iterator nq_i = e_nq.begin();
			arrayXf1DContainer::iterator zq_i = e_zq.begin();
			float1DContainer::iterator kld_i = c->e_kld.begin();

			for (int i = 0; i < (int)c->e_kld.size() ; i++){
				e_hp_store.push_back(*(hp_i++));
				e_dp_store.push_back(*(dp_i++));
				e_up_store.push_back(*(up_i++));
				e_lp_store.push_back(*(lp_i++));
				e_sp_store.push_back(*(sp_i++));
				e_np_store.push_back(*(np_i++));
				e_zp_store.push_back(*(zp_i++));
				e_hq_store.push_back(*(hq_i++));
				e_dq_store.push_back(*(dq_i++));
				e_uq_store.push_back(*(uq_i++));
				e_lq_store.push_back(*(lq_i++));
				e_sq_store.push_back(*(sq_i++));
				e_nq_store.push_back(*(nq_i++));
				e_zq_store.push_back(*(zq_i++));
				e_kld_store.push_back(*(kld_i++));
			}

			au_i = e_au.begin();
			al_i = e_al.begin();

			for (int i = 0; i < (int)e_au.size() ; i++){
				e_au_store.push_back(*(au_i++));
				e_al_store.push_back(*(al_i++));
			}
		}

	}

	void LayerPvrnnBeta::e_copyParam(){

		vectorXf1DContainer::iterator  au_i 	 = e_au.begin();
		vectorXf1DContainer::iterator  au_copy_i = e_au_copy.begin();
		vectorXf1DContainer::iterator  al_i 	 = e_al.begin();
		vectorXf1DContainer::iterator  al_copy_i = e_al_copy.begin();

		for (int t = 0; t < e_window_size ; t++, au_i++, au_copy_i++, al_i++, al_copy_i++){
			*au_copy_i = *au_i;
			*al_copy_i = *al_i;
		}
		c->dp_gen = c->e_dq.back();
		hp_gen = e_hq.back();

		e_hq_opt = e_hq[1];
		e_dq_opt = c->e_dq[1];
	}

	void LayerPvrnnBeta::e_overwriteParam(){

		e_dq_tzero = e_dq_opt;
		e_hq_tzero = e_hq_opt;

		vectorXf1DContainer::iterator  au_i 	 = e_au.begin();
		vectorXf1DContainer::iterator  au_copy_i = e_au_copy.begin();
		vectorXf1DContainer::iterator  al_i 	 = e_al.begin();
		vectorXf1DContainer::iterator  al_copy_i = e_al_copy.begin();

		au_copy_i++;
		al_copy_i++;

		for (int t = 0; t < e_window_size-1 ; t++, au_i++, au_copy_i++, al_i++, al_copy_i++){
			*au_i = *au_copy_i;
			*al_i = *al_copy_i;
		}

		ut->zero<VectorXf>(au_i.base());
		ut->zero<VectorXf>(al_i.base());
	}

	float* LayerPvrnnBeta::e_getState(float* _f){

		if (e_hp.size() > 0){
			_f = ut->copyEigenData<ArrayXf>(&e_hp.back(),  _f);
			_f = ut->copyEigenData<ArrayXf>(&c->e_dp.back(),  _f);
			_f = ut->copyEigenData<ArrayXf>(&e_up.back(),  _f);
			_f = ut->copyEigenData<ArrayXf>(&e_lp.back(),  _f);
			_f = ut->copyEigenData<ArrayXf>(&e_sp.back(),  _f);
			_f = ut->copyEigenData<ArrayXf>(&e_np.back(),  _f);
			_f = ut->copyEigenData<ArrayXf>(&e_zp.back(),  _f);

			_f = ut->copyEigenData<ArrayXf>(&e_hq.back(),  _f);
			_f = ut->copyEigenData<ArrayXf>(&c->e_dq.back(),  _f);
			_f = ut->copyEigenData<ArrayXf>(&e_uq.back(),  _f);
			_f = ut->copyEigenData<ArrayXf>(&e_lq.back(),  _f);
			_f = ut->copyEigenData<ArrayXf>(&e_sq.back(),  _f);
			_f = ut->copyEigenData<ArrayXf>(&e_nq.back(),  _f);
			_f = ut->copyEigenData<ArrayXf>(&e_zq.back(),  _f);
		}else{

			_f = ut->copyEigenData<ArrayXf>(&hp_gen,  _f);
			_f = ut->copyEigenData<ArrayXf>(&c->dp_gen,  _f);
			_f = ut->copyEigenData<ArrayXf>(&up_gen,  _f);
			_f = ut->copyEigenData<ArrayXf>(&lp_gen,  _f);
			_f = ut->copyEigenData<ArrayXf>(&sp_gen,  _f);
			_f = ut->copyEigenData<ArrayXf>(&np_gen,  _f);
			_f = ut->copyEigenData<ArrayXf>(&zp_gen,  _f);

			ArrayXf d_0 = ArrayXf::Zero(d_num);
			ArrayXf z_0 = ArrayXf::Zero(z_num);
			_f = ut->copyEigenData<ArrayXf>(&d_0,  _f);
			_f = ut->copyEigenData<ArrayXf>(&d_0,  _f);
			_f = ut->copyEigenData<ArrayXf>(&z_0,  _f);
			_f = ut->copyEigenData<ArrayXf>(&z_0,  _f);
			_f = ut->copyEigenData<ArrayXf>(&z_0,  _f);
			_f = ut->copyEigenData<ArrayXf>(&z_0,  _f);
			_f = ut->copyEigenData<ArrayXf>(&z_0,  _f);
		}
		return _f;
	}



	void LayerPvrnnBeta::e_save(string _path){

		std::string delimiter = ut->getDelimiter();

		bool success = true;

		try{
			if (e_store_gen == true){

				vector<string> gen_names;
				gen_names.push_back("e_hp_gen");
				gen_names.push_back("e_dp_gen");
				gen_names.push_back("e_up_gen");
				gen_names.push_back("e_lp_gen");
				gen_names.push_back("e_sp_gen");
				gen_names.push_back("e_np_gen");
				gen_names.push_back("e_zp_gen");

				vector<ofstream> gen_File;
				for (unsigned int i = 0 ; i < gen_names.size() ; i++){
					stringstream strm; strm << _path << "/e_L" << id << "_" << gen_names[i]  << ".d";
					gen_File.push_back(ofstream(strm.str(),std::ofstream::out));
				}

				for (unsigned int i = 0 ; i < gen_names.size() ; i++)
					success = success ? gen_File[i].is_open() : false;

				if (success){

					vector<arrayXf1DContainer::iterator> gen_it;
					gen_it.push_back(hp_gen_store.begin());
					gen_it.push_back(dp_gen_store.begin());
					gen_it.push_back(up_gen_store.begin());
					gen_it.push_back(lp_gen_store.begin());
					gen_it.push_back(sp_gen_store.begin());
					gen_it.push_back(np_gen_store.begin());
					gen_it.push_back(zp_gen_store.begin());

					for (unsigned int k = 0 ; k < gen_it.size(); k++){
						for (unsigned int i = 0; i < hp_gen_store.size(); i++){
							//oist::Utils::writeArrToFile(&gen_File[k], (gen_it[k]++).base(), delimiter);
							ut->saveEigen<ArrayXf>(&gen_File[k], (gen_it[k]++).base(), delimiter);
						}
					}
				}
			}
			if (success == true && e_store_inference == true){

				vector<string> infer_names;
				infer_names.push_back("e_hp");
				infer_names.push_back("e_dp");
				infer_names.push_back("e_up");
				infer_names.push_back("e_lp");
				infer_names.push_back("e_sp");
				infer_names.push_back("e_np");
				infer_names.push_back("e_zp");
				infer_names.push_back("e_hq");
				infer_names.push_back("e_dq");
				infer_names.push_back("e_uq");
				infer_names.push_back("e_lq");
				infer_names.push_back("e_sq");
				infer_names.push_back("e_nq");
				infer_names.push_back("e_zq");

				vector<ofstream> er_File;
				for (unsigned int i = 0 ; i < infer_names.size() ; i++){
					stringstream strm; strm << _path << "/e_L" << id << "_" << infer_names[i]  << ".d";
					er_File.push_back(ofstream(strm.str(),std::ofstream::out));
				}

				for (unsigned int i = 0 ; i < infer_names.size() ; i++)
					success = success ? er_File[i].is_open(): false;

				vector<arrayXf1DContainer::iterator> er_it;
				if (success){
					er_it.push_back(e_hp_store.begin());
					er_it.push_back(e_dp_store.begin());
					er_it.push_back(e_up_store.begin());
					er_it.push_back(e_lp_store.begin());
					er_it.push_back(e_sp_store.begin());
					er_it.push_back(e_np_store.begin());
					er_it.push_back(e_zp_store.begin());
					er_it.push_back(e_hq_store.begin());
					er_it.push_back(e_dq_store.begin());
					er_it.push_back(e_uq_store.begin());
					er_it.push_back(e_lq_store.begin());
					er_it.push_back(e_sq_store.begin());
					er_it.push_back(e_nq_store.begin());
					er_it.push_back(e_zq_store.begin());

					for (unsigned int k = 0 ; k < er_it.size(); k++){
						for (unsigned int i = 0; i < hp_gen_store.size(); i++){
							ut->saveEigen<ArrayXf>(&er_File[k], (er_it[k]++).base(), delimiter);
						}
						er_File[k].close();
					}
					stringstream strm1; strm1 << _path << "/e_L" << id << "_" << "kld"  << ".d";
					ofstream klDiv_er_File(strm1.str(),std::ofstream::out);

					success =  success ? klDiv_er_File.is_open() : false;

					if (success){
						float1DContainer::iterator klDiv_er_it = e_kld_store.begin();
						for (unsigned int i = 0; i < e_kld_store.size(); i++){
							ut->saveScalar<float>(&klDiv_er_File, *(klDiv_er_it++));
							klDiv_er_File << delimiter;
						}
						klDiv_er_File.close();
					}

					stringstream strm2; strm2 << _path << "/e_L" << id << "_" << "au"  << ".d";
					ofstream AMu_er_File(strm2.str(),std::ofstream::out);

					success =  success ? AMu_er_File.is_open() : false;

					if (success){
						vectorXf1DContainer::iterator AMu_er_it = e_au_store.begin();
						for (unsigned int i = 0; i < e_au_store.size(); i++){
							ut->saveEigen<VectorXf>(&AMu_er_File, (AMu_er_it++).base(), delimiter);
						}
						AMu_er_File.close();
					}

					stringstream strm3; strm3 << _path << "/expMode_L" << id << "_" << "al"  << ".d";
					ofstream ALs_er_File(strm3.str(),std::ofstream::out);

					success =  success ? ALs_er_File.is_open() : false;

					if (success){
						vectorXf1DContainer::iterator ALs_er_it = e_al_store.begin();
						for (unsigned int i = 0; i < e_al_store.size(); i++){
							ut->saveEigen<VectorXf>(&ALs_er_File, (ALs_er_it++).base(), delimiter);
						}
						ALs_er_File.close();
					}
				}
			}
		}catch(oist::Exception& _e){
			stringstream stream;
			stream << "Unsuccessful saving of experimental data" << endl;
			throw oist::Exception(stream.str());
		}
		if (success ==  false){
			throw oist::Exception("Error while saving the experiment data");
		}
	 }

	// ------------------------- Alanysis mode methods -------------------------

	void LayerPvrnnBeta::a_init(float* _init_state){

		free_memory();

		float* init_state = _init_state;

		c->dp_gen = ArrayXf::Zero(d_num);
		hp_gen = ArrayXf::Zero(d_num);

		auto hp_gen_t_p = hp_gen.data();
		for (int d = 0 ; d < d_num ; d++, init_state++, hp_gen_t_p++){
			*hp_gen_t_p = *init_state;
		}

		auto dp_gen_t_p = c->dp_gen.data();
		for (int d = 0 ; d < d_num ; d++, init_state++, dp_gen_t_p++){
			*dp_gen_t_p = *init_state;
		}

	}

	void LayerPvrnnBeta::a_predict(){

		 VectorXf hp = hp_gen;
		 VectorXf dp = c->dp_gen;

		 ArrayXf mp_ = Wdup*dp + Bup;
		 ut->tanH<ArrayXf>(&mp_);
		 ArrayXf lsp_ = Wdlp*dp + Blp;
		 ArrayXf sp_ = lsp_.exp();
		 ArrayXf np_ = ArrayXf::Zero(z_num);
		 ut->randN<ArrayXf>(&np_);
		 VectorXf Zp_ = mp_ + sp_*np_;

		 hp = one_sub_eps*hp + eps*(Wdh*dp + Wzh*Zp_ + Bh);

		 if (!top)
			 hp += eps*(Wdh_top*c->dp_top);

		 dp = hp;
		 ut->tanH<VectorXf>(&dp);

		 hp_gen_store.push_back(hp);
		 dp_gen_store.push_back(dp);
		 up_gen_store.push_back(mp_);
		 lp_gen_store.push_back(lsp_);
		 sp_gen_store.push_back(sp_);
		 np_gen_store.push_back(np_);
		 zp_gen_store.push_back(Zp_);

		 hp_gen = hp;
		 c->dp_gen = dp;
	}

	void LayerPvrnnBeta::a_save(string _path){

		std::string delimiter = ut->getDelimiter();

		vector<string> gen_names;
		gen_names.push_back("a_hp_gen");
		gen_names.push_back("a_dp_gen");
		gen_names.push_back("a_up_gen");
		gen_names.push_back("a_lp_gen");
		gen_names.push_back("a_sp_gen");
		gen_names.push_back("a_np_gen");
		gen_names.push_back("a_zp_gen");


		vector<ofstream> gen_File;
		for (unsigned int i = 0 ; i < gen_names.size() ; i++){
			stringstream strm; strm << _path << "/a_L" << id << "_" << gen_names[i] << ".d";
			gen_File.push_back(ofstream(strm.str(),std::ofstream::out));
		}

		bool success = true;
		for (unsigned int i = 0 ; i < gen_names.size() ; i++)
			success = success ? gen_File[i].is_open() : false;

		if (success){

			vector<arrayXf1DContainer::iterator> gen_it;
			gen_it.push_back(hp_gen_store.begin());
			gen_it.push_back(dp_gen_store.begin());
			gen_it.push_back(up_gen_store.begin());
			gen_it.push_back(lp_gen_store.begin());
			gen_it.push_back(sp_gen_store.begin());
			gen_it.push_back(np_gen_store.begin());
			gen_it.push_back(zp_gen_store.begin());

			try{

				for (unsigned int k = 0 ; k < gen_it.size(); k++){
					for (unsigned int i = 0; i < hp_gen_store.size(); i++){
						ut->saveEigen<ArrayXf>(&gen_File[k], (gen_it[k]++).base(), delimiter);
					}
				}

				// closing the files
				for (unsigned int k = 0 ; k < gen_it.size(); k++){
					gen_File[k].close();
				}
			}catch(oist::Exception& _e){
				stringstream stream;
				stream << "Unsuccessful saving of analysis mode data" << endl;
				throw oist::Exception(stream.str());
			}
		}else{
			throw oist::Exception("Error while saving analysis mode data");
		}
	 }


	LayerPvrnnBeta::~LayerPvrnnBeta() {
		delete c;
		cout << "Layer #" << id << " deallocated" << endl;
	}

}/* namespace oist */

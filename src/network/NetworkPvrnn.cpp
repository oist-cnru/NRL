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

#include "NetworkPvrnn.h"
#include "../utils/Utils.h"

namespace oist {


	NetworkPvrnn::NetworkPvrnn(map<string,float1DContainer>& _float1DMap, Dataset* _dataset){

		dataset = _dataset;

		if(_float1DMap.find("d") == _float1DMap.end()) throw Exception("'d' property not found");
		float1DContainer fDNum = _float1DMap["d"];
		for (unsigned int i = 0; i < fDNum.size(); i++){
			int v = int(fDNum[i]);
			if (v <= 0)
				throw Exception("'d' property should include positive integer(s) greater than zero");
			d_num.push_back(v);
		}

		if(_float1DMap.find("z") == _float1DMap.end()) throw Exception("'z' property not found");
		float1DContainer fZNum = _float1DMap["z"];
		for (unsigned int i = 0; i < fZNum.size(); i++){
			int v = int(fZNum[i]);
			if (v <= 0)
				throw Exception("'z' property should include positive integer(s) greater than zero");
			z_num.push_back(v);
		}

		if(_float1DMap.find("t") == _float1DMap.end()) throw Exception("'t' property not found");
		float1DContainer fTau = _float1DMap["t"];
		for (unsigned int i = 0; i < fTau.size(); i++){
			int v = int(fTau[i]);
			if (v <= 0)
				throw Exception("'t' property should include positive integer(s) greater than zero");
			tau.push_back(v);
		}

		if(_float1DMap.find("w") == _float1DMap.end()) throw Exception("'w' property not found");
		w = _float1DMap["w"];
		for (unsigned int i = 0; i < w.size(); i++){
			if (w[i] < 0)
				throw Exception("'w' property should include positive real number(s)");
		}

		layer_num = d_num.size();
		ut = Utils::getInstance();

		int checksum = d_num.size() + z_num.size() + tau.size() + w.size();

		if (checksum != layer_num*4)
			throw  Exception("The network parameters are not correctly defined");

		dataset->getNunitsPerDim(o_num);
		prim_num = _dataset->getNPrim();
		prim_len = dataset->getPrimLength();

		int z_sum = 0;
		for (unsigned int i = 0 ; i < z_num.size(); i++)
			z_sum += z_num[i];

		o_dim = o_num.size();
		rec_coef = 1.0/((float)(o_dim*1.0));
		reg_coef = 1.0/((float)(z_sum*1.0));

		cout << "Number of internal layers: " << layer_num << endl;
		state_dim  = 0;
		for (int l = 0; l < layer_num; l++){

			int d_num_bottom = (l==0)? 0 : d_num[l-1];
			int d_num_top = (l==layer_num-1)? 0 : d_num[l+1];

			ILayer* layer = new LayerPvrnn(l, d_num[l], d_num_bottom, d_num_top, z_num[l], z_sum, tau[l], (l > 0 ? tau[l-1] : 0.0), (l < layer_num ? tau[l+1] : 0.0), prim_num, prim_len, w[l]);
			layers.push_back(layer);
			state_dim += layer->getStateDim();
	
		}

		// connections to output layers
		l0_d_num = d_num[0];
		l0 = layers[0];
		l0_context = static_cast<ContextPvrnn*>(l0->getContext());

		for (int o = 0; o < o_dim ; o++){

			int num = o_num[o];
			MatrixXf Wdx_ = ut->kaiming_uniform_initialization(num, l0_d_num, Utils::nonlinearity::Linear);
			MatrixXf WdxT_ = Wdx_.transpose();
			Wdo.push_back(Wdx_);
			Wdo_transpose.push_back(WdxT_);
			g_Wdo.push_back(MatrixXf::Zero(num, l0_d_num));
			m_Wdo.push_back(MatrixXf::Zero(num, l0_d_num));
			v_Wdo.push_back(MatrixXf::Zero(num, l0_d_num));
			Bo.push_back(ut->kaiming_uniform_initialization(num));
			g_Bo.push_back(VectorXf::Zero(num));
			m_Bo.push_back(VectorXf::Zero(num));
			v_Bo.push_back(VectorXf::Zero(num));

		}

		e_prim_id = 0;
		e_cur_time = 0;
		e_num_times = 0;
		e_window_size = 0;
		e_store_gen = false;
		e_store_inference  = false;;
	}

	int NetworkPvrnn::getNLayers(){

		return layer_num;
	}

	int NetworkPvrnn::getStateDim(){

		return state_dim;

	}

	void NetworkPvrnn::t_generate(int _n, int _prim_id, vectorXf2DContainer& _X){

		 for (int l = 0 ; l < layer_num; l++){
			 layers[l]->initContext(_prim_id);
		 }

		 for (int t = 0; t < _n; t++){
			 arrayXf1DContainer dp_prev;
			 for (int l = 0; l < layer_num; l++){
				 dp_prev.push_back(static_cast<ContextPvrnn*>(layers[l]->getContext())->t_dp[_prim_id].back());
			 }

			 for (int l = 0; l < layer_num; l++){
				 ILayer* ll = layers[l];
				 ContextPvrnn* lc = static_cast<ContextPvrnn*>(ll->getContext());

				 if (l > 0 ){
					 lc ->dp_bottom_prev = dp_prev[l-1];
				 }
				 if (l < layer_num-1){
					 lc ->dp_top_prev = dp_prev[l+1];
				 }

				 ll->t_generate(t, _prim_id);
			}
			VectorXf dp0 = l0_context->t_dp[_prim_id].back();

			vectorXf1DContainer Xt;
			for (int o = 0; o < o_dim; o++){

				 VectorXf Xto = Wdo[o]*dp0 + Bo[o];
				 ut->softmax<VectorXf>(&Xto);
				 Xt.push_back(Xto);
			}

			_X.push_back(Xt);
		 }
	 }


	 void NetworkPvrnn::t_forward(int _n, int _prim_id, vectorXf2DContainer& _X){

		 for (int l = 0 ; l < layer_num; l++){
			 layers[l]->initContext(_prim_id);
		 }

		 for (int t = 0; t < _n; t++){
			 arrayXf1DContainer dp_prev;
			 arrayXf1DContainer dq_prev;

			 for (int l = 0; l < layer_num; l++){

				 ILayer* ll = layers[l];
				 ContextPvrnn* lc = static_cast<ContextPvrnn*>(ll->getContext());
				 dp_prev.push_back(lc->t_dp[_prim_id].back());
				 dq_prev.push_back(lc->t_dq[_prim_id].back());

			 }

			 for (int l = 0; l < layer_num; l++){

				 ILayer* ll = layers[l];
				 ContextPvrnn* lc = static_cast<ContextPvrnn*>(ll->getContext());

				 if (l > 0 ){
					 lc->dp_bottom_prev = dp_prev[l-1];
					 lc->dq_bottom_prev = dq_prev[l-1];
				 }
				 if (l < layer_num-1){
					 lc->dp_top_prev = dp_prev[l+1];
					 lc->dq_top_prev = dq_prev[l+1];
				 }

				 ll->t_forward(t, _prim_id);
			 }
			 VectorXf dq0 = l0_context->t_dq[_prim_id].back();
			 vectorXf1DContainer Xt;
			 for (int o = 0; o < o_dim; o++){

				 VectorXf Xto = Wdo[o]*dq0 + Bo[o];
				 ut->softmax<VectorXf>(&Xto);
				 Xt.push_back(Xto);
			 }
			 _X.push_back(Xt);
		 }

	}

	 void NetworkPvrnn::t_backward(int _prim_id, vectorXf2DContainer& _X, vectorXf3DContainer& _Y, float& _rec, float& _reg, float& _loss){

		 float1DContainer klDiv_l;
		 for (int l = 0; l < layer_num; l++){
			 klDiv_l.push_back(0.0);
		 }

		 for (unsigned int s = 0; s < _Y.size(); s++){ // for all the batch samples


			 vectorXf2DContainer Ys = _Y[s];
			 vector<VectorXf> gH;
			 vector<VectorXf> gH_next;
			 for (int l = 0; l < layer_num; l++){
				 gH.push_back(VectorXf::Zero(d_num[l]));
				 gH_next.push_back(VectorXf::Zero(d_num[l]));
				 layers[l]->t_initBackward();
			 }

			 int t_prev = prim_len-1;

			 for (int t = prim_len; t > 0; t--, t_prev--){

				 vectorXf1DContainer X_t = _X[t_prev];
				 vectorXf1DContainer Y_st = Ys[t_prev];
				 VectorXf g_dqloss =  VectorXf::Zero(l0_d_num);
				 RowVectorXf L0_dq = l0_context->t_dq[_prim_id][t];

				 for (int o = 0; o < o_dim; o++){

					 ArrayXf Xpto = X_t[o];
					 ArrayXf Ypsto = Y_st[o];
					 ArrayXf yx = (Ypsto/Xpto) + NON_ZERO;
					 VectorXf recErr_t = Ypsto*(yx.log());
					 VectorXf gxloss_to = rec_coef*(Xpto-Ypsto);

					 g_dqloss += Wdo_transpose[o]*gxloss_to ;

					 g_Wdo[o] += gxloss_to*L0_dq;
					 g_Bo[o] += gxloss_to;

					 _rec += recErr_t.sum();

				}

				for (int l = 0; l < layer_num; l++){
					ILayer* ll = layers[l];
					ContextPvrnn* lc = static_cast<ContextPvrnn*>(ll->getContext());


					if (l > 0 ){
						lc->g_hq_bottom_next = gH_next[l-1];
						lc->dq_bottom_prev = static_cast<ContextPvrnn*>(layers[l-1]->getContext())->t_dq[_prim_id][t_prev];
					}
					else{
						lc->g_dqloss = g_dqloss;
					}
					if (l < layer_num-1){
						lc->g_hq_top_next = gH_next[l+1];
						lc->dq_top_prev = static_cast<ContextPvrnn*>(layers[l+1]->getContext())->t_dq[_prim_id][t_prev];
					}

					ll->t_backward(t, _prim_id);

					gH[l] = lc->g_h_next;

					klDiv_l[l] += lc->t_kld[_prim_id][t];
				}

				for (int l = 0; l < layer_num; l++){
					gH_next[l] = gH[l];
				}
			 }
		 }

		 for (int l = 0; l < layer_num; l++){
			 _reg += w[l]*klDiv_l[l];
		 }
		 _loss = rec_coef*_rec + reg_coef*_reg;
	 }


	 void NetworkPvrnn::t_optAdam(int _e, float _a, float _b1, float _b2){

		 for (int o = 0; o < o_dim; o++){

			 // updating parameters
			 ut->adam<MatrixXf>(&Wdo[o], &g_Wdo[o], &m_Wdo[o], &v_Wdo[o], _e, _a, _b1, _b2);
			 ut->adam<VectorXf>(&Bo[o], &g_Bo[o], &m_Bo[o], &v_Bo[o], _e, _a, _b1, _b2);

			 Wdo_transpose[o] = Wdo[o].transpose();

			 // clearing parameter gradients
			 ut->zero<MatrixXf>(&g_Wdo[o]);
			 ut->zero<VectorXf>(&g_Bo[o]);
		 }

		 // updating the layer parameters
		 for (int l = 0; l < layer_num; l++){
		 	layers[l]->t_optAdam(_e, _a, _b1, _b2);
		 }

	 }


	 float NetworkPvrnn::getRecError(vectorXf2DContainer& _X, vectorXf3DContainer&  _Y){

		 float rec = 0.0;

		 for (unsigned int s = 0; s < _Y.size(); s++){

			 vectorXf2DContainer Ys = _Y[s];

			 for (unsigned int t = 0; t < Ys.size(); t++){

				 vectorXf1DContainer Yst = Ys[t];
				 vectorXf1DContainer Xt = _X[t];

				 for (int o = 0; o < o_dim; o++){

					 ArrayXf Xpto = Xt[o];
					 ArrayXf Ypsto = Yst[o];
					 ArrayXf yx = (Ypsto/Xpto) + NON_ZERO;
					 VectorXf recErr_t = Ypsto*(yx.log());
					 rec += recErr_t.sum();
				 }
			 }
		 }
		 return rec;
	 }

	void NetworkPvrnn::print(){

		cout << "Output layer:" << endl;

		for (int o = 0; o < o_dim; o++){

			auto d = Wdo[o].data();
			cout << "Wdo"<< o << endl;
			cout << "data: ";

			for (int i = 0; i < Wdo[o].size(); i++, d++){
				cout << *d << " ";
			}

			cout << endl;
			cout << "size: [" << Wdo[o].rows() << "," << Wdo[o].cols() << "]"<< endl;

		}
		for (int o = 0; o < o_dim; o++){

			cout << "Bo"<< o << endl;
			cout << "data: ";
			auto d = Bo[o].data();

			for (int i = 0; i < Bo[o].size(); i++, d++){
				cout << *d << " ";
			}

			cout << endl;
			cout << "size: [" << Bo[o].rows() << "," << Bo[o].cols() << "]"<< endl;

		}

		cout << "Internal layers:" << endl;
		for (int l = 0; l < layer_num; l++){
			layers[l]->print();
		}
	}

	void NetworkPvrnn::load(string _path){

		std::string delimiter = ut->getDelimiter();

		for (int o = 0; o < o_dim; o++){

			// loading the output layer parameters

			MatrixXf* w_p = &Wdo[o];
			MatrixXf* w_m = &m_Wdo[o];
			MatrixXf* w_v = &v_Wdo[o];

			stringstream strmWp;	strmWp << _path << "/o" << o << "_w_p.d";
			stringstream strmWm; 	strmWm << _path << "/o" << o << "_w_m.d";
			stringstream strmWv;	strmWv << _path << "/o" << o << "_w_v.d";

			ifstream wFile(strmWp.str()); ifstream m_wFile(strmWm.str());	ifstream v_wFile(strmWv.str());

			VectorXf* b_p = &Bo[o];
			VectorXf* b_m = &m_Bo[o];
			VectorXf* b_v = &v_Bo[o];

			stringstream strmBp;	strmBp << _path << "/o" << o << "_b_p.d";
			stringstream strmBm; 	strmBm << _path << "/o" << o << "_b_m.d";
			stringstream strmBv;	strmBv << _path << "/o" << o << "_b_v.d";

			ifstream bFile(strmBp.str()); ifstream m_bFile(strmBm.str());	ifstream v_bFile(strmBv.str());

			ifstream* buff_wFile[] = 	{&wFile,   &m_wFile,   &v_wFile};
			ifstream* buff_bFile[] = 	{&bFile,   &m_bFile,   &v_bFile};

			if (wFile.is_open()   && m_wFile.is_open()   && v_wFile.is_open()   &&
				bFile.is_open()   && m_bFile.is_open()   && v_bFile.is_open()){

				try{
					MatrixXf* buff_w [] = {w_p, w_m, w_v};
					for (int k = 0 ; k < 3 ; k++){
						ut->loadEigen<MatrixXf>(buff_wFile[k], buff_w[k], delimiter);
					}

					VectorXf* buff_b [] = {b_p, b_m, b_v};
					for (int k = 0 ; k < 3 ; k++){
						ut->loadEigen<VectorXf>(buff_bFile[k], buff_b[k], delimiter);
					}

					//closing files
					wFile.close();		 m_wFile.close();		 v_wFile.close();
					bFile.close();		 m_bFile.close();		 v_bFile.close();
				}catch(...){
					throw oist::Exception("Unknown IO Error while loading layer parameters");
				}
			}
			else{
				throw oist::Exception("Fail to open the parameters files");
			}
		}
		try {
			// load layers data
			for (int l = 0; l < layer_num; l++){
				layers[l]->load(_path);
			}
			cout << "Model loaded!" << endl;
		}
		catch(Exception& _e){
			throw _e;
		}
	}


	void NetworkPvrnn::save(string _path){

		std::string delimiter = ut->getDelimiter();

		for (int o = 0; o < o_dim; o++){

			// saving the output layer parameters
			MatrixXf* w_p = &Wdo[o];
			MatrixXf* w_m = &m_Wdo[o];
			MatrixXf* w_v = &v_Wdo[o];

			stringstream strmWp, strmWm, strmWv;

			strmWp << _path << "/o" << o << "_w_p.d";
			strmWm << _path << "/o" << o << "_w_m.d";
			strmWv << _path << "/o" << o << "_w_v.d";

			ofstream wFile(strmWp.str(),std::ofstream::out); ofstream m_wFile(strmWm.str(),std::ofstream::out);	ofstream v_wFile(strmWv.str(),std::ofstream::out);

			VectorXf* b_p = &Bo[o];
			VectorXf* b_m = &m_Bo[o];
			VectorXf* b_v = &v_Bo[o];

			stringstream strmBp, strmBm, strmBv;

			strmBp << _path << "/o" << o << "_b_p.d";
			strmBm << _path << "/o" << o << "_b_m.d";
			strmBv << _path << "/o" << o << "_b_v.d";

			ofstream bFile(strmBp.str(),std::ofstream::out); ofstream m_bFile(strmBm.str(),std::ofstream::out);	ofstream v_bFile(strmBv.str(),std::ofstream::out);

			ofstream* buff_wFile[] = 	{&wFile,   &m_wFile,   &v_wFile};
			ofstream* buff_bFile[] = 	{&bFile,   &m_bFile,   &v_bFile};

			if (wFile.is_open()   && m_wFile.is_open()   && v_wFile.is_open()   &&
				bFile.is_open()   && m_bFile.is_open()   && v_bFile.is_open()){

				try{
					MatrixXf* buff_w [] = {w_p, w_m, w_v};

					for (int k = 0 ; k < 3 ; k++){

						ut->saveEigen<MatrixXf>(buff_wFile[k], buff_w[k], delimiter);
					}

					VectorXf* buff_b [] = {b_p, b_m, b_v};

					for (int k = 0 ; k < 3 ; k++){

						ut->saveEigen<VectorXf>(buff_bFile[k], buff_b[k], delimiter);
					}

					//closing files
					wFile.close();		 m_wFile.close();		 v_wFile.close();
					bFile.close();		 m_bFile.close();		 v_bFile.close();
				}
				catch(...){
					throw oist::Exception("Unknown IO Error while saving output layer parameters");
				}
			}
			else{
				stringstream strm; 
				strm << "Fail to open/create the model parameters files. Check the path [" << _path << "]";
				throw oist::Exception(strm.str());
			}
		}
		try{
			// saving layer's data
			for (int l = 0; l < layer_num; l++){

				layers[l]->save(_path);
			}
			cout << "Model saved!" << endl;
		}
		catch(Exception& _e){
			throw _e;
		}
	}

	 	 // ------------------------- Experiment model methods -------------------------

	void NetworkPvrnn::e_enable(int _seqId, int _winSize, float* _params, int _exp_num_times, bool _storeStates, bool _storeER){

		e_prim_id = _seqId;
		e_window_size = _winSize;
		e_store_gen = _storeStates;
		e_store_inference = e_store_gen && _storeER;
		e_cur_time = 0;
		e_num_times = _exp_num_times;

		for (int l = 0 ; l < layer_num; l++){
			w[l] = _params[l];
			layers[l]->e_enable(e_prim_id, e_window_size, w[l], e_num_times, e_store_gen, e_store_inference);
		}
	}


	bool NetworkPvrnn::e_initForward(){

		int erTime = e_cur_time - e_window_size;

		if (erTime < 0){
			cout << "Current time "<< e_cur_time << " is less than window_size: " << e_window_size<< ", Error regression unavailable" << endl;
			return false;
		}

		return true;
	}


	void NetworkPvrnn::e_generate(float* _tgt_pos){

		arrayXf1DContainer dp_prev;

		for (int l = 0; l < layer_num; l++)
		{
			dp_prev.push_back(static_cast<ContextPvrnn*>(layers[l]->getContext())->dp_gen);
		}

		for (int l = 0; l < layer_num; l++){
			ILayer* ll = layers[l];
			ContextPvrnn* lc = static_cast<ContextPvrnn*>(ll->getContext());

			 if (l > 0 ){
				 lc->dp_bottom_prev = dp_prev[l-1];
			 }
			 if (l < layer_num-1){
				 lc->dp_top_prev = dp_prev[l+1];
			 }

			 ll->e_generate();
		}
		VectorXf dp0 = l0_context->dp_gen;
		for (int o = 0; o < o_dim; o++, _tgt_pos++){

			 ArrayXf Xto = Wdo[o]*dp0 + Bo[o];
			 ut->softmax<ArrayXf>(&Xto);

			 *_tgt_pos = dataset->decodeSoftmax(Xto, o);
		 }
		e_cur_time++;

	}

	void NetworkPvrnn::e_forward(vectorXf2DContainer& _X){

		 for (int l = 0 ; l < layer_num; l++){
			 layers[l]->e_initForward();
		 }

		 for (int t = 0; t < e_window_size; t++){
			 vectorXf1DContainer Xt;
			 arrayXf1DContainer dp_prev;
			 arrayXf1DContainer dq_prev;

			 for (int l = 0; l < layer_num; l++){
				 ILayer* ll = layers[l];
				 ContextPvrnn* lc = static_cast<ContextPvrnn*>(ll->getContext());
				 dp_prev.push_back(lc->e_dp.back());
				 dq_prev.push_back(lc->e_dq.back());
			 }

			 for (int l = 0; l < layer_num; l++){
				 ILayer* ll = layers[l];
				 ContextPvrnn* lc = static_cast<ContextPvrnn*>(ll->getContext());

				 if (l > 0 ){
					 lc->dp_bottom_prev = dp_prev[l-1];
					 lc->dq_bottom_prev = dq_prev[l-1];;
				 }
				 if (l < layer_num-1){
					 lc->dp_top_prev = dp_prev[l+1];
					 lc->dq_top_prev = dq_prev[l+1];
				 }

				 ll->e_forward();
			 }

			 VectorXf dq0 = l0_context->e_dq.back();

			 for (int o = 0; o < o_dim; o++){

				 VectorXf Xto = Wdo[o]*dq0 + Bo[o];
				 ut->softmax<VectorXf>(&Xto);
				 Xt.push_back(Xto);
			 }
			 _X.push_back(Xt);
		 }
	}

	 void NetworkPvrnn::e_backward(vectorXf2DContainer& _X, vectorXf2DContainer& _Y, float& _rec, float& _reg, float& _loss){

		 float1DContainer kld_l;
		 vector<float1DContainer::reverse_iterator> kld_bw_i;
		 vector<VectorXf> gH;
		 vector<VectorXf> gH_next;

		 for (int l = 0; l < layer_num; l++){
			 gH.push_back(VectorXf::Zero(d_num[l]));
			 gH_next.push_back(VectorXf::Zero(d_num[l]));
			 ILayer* ll = layers[l];
			 ContextPvrnn* lc = static_cast<ContextPvrnn*>(ll->getContext());
			 ll->e_initBackward();
			 kld_bw_i.push_back(lc->e_kld.rbegin());
			 kld_l.push_back(0.0);
		 }

		 int t_prev = e_window_size-1;

		 for (int t = e_window_size; t > 0; t--, t_prev--){

			 vectorXf1DContainer X_t = _X[t_prev];
			 vectorXf1DContainer Y_t = _Y[t_prev];
			 VectorXf g_dqloss =  VectorXf::Zero(l0_d_num);

			 for (int o = 0; o < o_dim; o++){

				 ArrayXf Xpto = X_t[o];
				 ArrayXf Ypsto = Y_t[o];
				 ArrayXf yx = (Ypsto/Xpto) + NON_ZERO;
				 VectorXf rec_t = Ypsto*(yx.log());
				 VectorXf gxloss_to = rec_coef*(Xpto-Ypsto);
				 g_dqloss += Wdo_transpose[o]*gxloss_to ;
				 _rec += rec_t.sum();
			}

			for (int l = 0; l < layer_num; l++){
				ILayer* ll = layers[l];
				ContextPvrnn* lc = static_cast<ContextPvrnn*>(ll->getContext());

				if (l > 0 ){
					lc->g_hq_bottom_next = gH_next[l-1];
				}else{
					lc->g_dqloss = g_dqloss;
				}
				if (l < layer_num-1){
					lc->g_hq_top_next = gH_next[l+1];
				}

				ll->e_backward(t);

				gH[l] = lc->g_h_next;
				kld_l[l] += *(kld_bw_i[l]++);
			}

			for (int l = 0; l < layer_num; l++){
				gH_next[l] = gH[l];
			}
		 }

		 for (int l = 0; l < layer_num; l++){
			 _reg += w[l]*kld_l[l];
		 }
		 _loss = rec_coef*_rec + reg_coef*_reg;
	 }


	 void NetworkPvrnn::e_copyParam(){

		 for (int l = 0 ; l < layer_num; l++)
			 layers[l]->e_copyParam();
	 }

	void NetworkPvrnn::e_overwriteParam(){

		 for (int l = 0 ; l < layer_num; l++){
			 layers[l]->e_overwriteParam();
		 }
	 }

	void NetworkPvrnn::e_optAdam(int _epoch, float _alpha, float _beta1, float _beta2){

		 for (int l = 0 ; l < layer_num; l++){
			layers[l]->e_optAdam(_epoch, _alpha, _beta1, _beta2);
		 }

	}

	void NetworkPvrnn::e_getState(float* _f){

		for (int l = 0; l < layer_num ; l++){
			_f = layers[l]->e_getState(_f);
		}
	}

	// ------------------------- Analysis mode methods -------------------------

	void NetworkPvrnn::a_feedForwardOutputFromContext(float* _d0, float* _X){

			VectorXf d0 = ArrayXf::Zero(l0_d_num);
			auto d0_p = d0.data();
			_d0 += l0_d_num;
			for (int i = 0 ; i < l0_d_num; i++, d0_p++, _d0++)
				*d0_p = *_d0;
			for (int o = 0; o < o_dim; o++, _X++){

				 ArrayXf Xto = Wdo[o]*d0 + Bo[o];
				 ut->softmax<ArrayXf>(&Xto);

				 *_X = dataset->decodeSoftmax(Xto, o);
			 }
		}

	void NetworkPvrnn::a_predict(int _n, float* _initial_state, string _path){

		stringstream strm; strm << _path << "/off_X.d";
		ofstream offL_X_File(strm.str(),std::ofstream::out);

		if (!offL_X_File.is_open()){
			throw oist::Exception("The output file could not be opened for the Network off-line generation");
		}

		float* p_is = _initial_state;

		for (int l = 0 ; l < layer_num; l++){
			layers[l]->a_init(p_is);
			p_is+= layers[l]->getStateDim();
		}

		float2DContainer X;

		for (int t = 0; t < _n; t++){
			arrayXf1DContainer dp_prev;
			for (int l = 0; l < layer_num; l++){
				ContextPvrnn* lc = static_cast<ContextPvrnn*>(layers[l]->getContext());
				dp_prev.push_back(lc->dp_gen);
			}
			for (int l = 0; l < layer_num; l++){
				ILayer* ll = layers[l];
				ContextPvrnn* lc = static_cast<ContextPvrnn*>(ll->getContext());

				 if (l > 0 ){
					 lc->dp_bottom_prev = dp_prev[l-1];
				 }
				 if (l < layer_num-1){
					 lc->dp_top_prev = dp_prev[l+1];
				 }
				 layers[l]->a_predict();
			}

			VectorXf dp0 = l0_context->dp_gen;

			float1DContainer X_t;
			for (int o = 0; o < o_dim; o++){

				 ArrayXf Xto = Wdo[o]*dp0 + Bo[o];
				 ut->softmax<ArrayXf>(&Xto);

				 X_t.push_back(dataset->decodeSoftmax(Xto, o));
			 }
			X.push_back(X_t);
		}


		for (unsigned int i = 0 ; i < X.size(); i++){
			ut->saveContainer<float1DContainer>(&offL_X_File, &X[i], string(" "));
		}

		try{
			// saving layer's data
			for (int l = 0; l < layer_num; l++){
				layers[l]->a_save(_path);
			}
			cout << "Off-line generation saved!" << endl;
		}
		catch(Exception& _e){
			throw _e;
		}

	}
	void NetworkPvrnn::e_save(string _path){

		if (e_store_gen == true){

			try{
				// saving layer's data
				for (int l = 0; l < layer_num; l++){
					layers[l]->e_save(_path);
				}
				cout << "Experiment saved!" << endl;
			}
			catch(Exception& _e){
				throw _e;
			}
		}

	}



	NetworkPvrnn::~NetworkPvrnn() {
		 for (int l = 0 ; l < layers.size(); l++){
			 delete layers[l]; 
		 }
		 cout << "Network deallocated" << endl;
	}

} /* namespace oist */

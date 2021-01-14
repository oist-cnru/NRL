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



#include "LibNRL.h"

namespace oist {

LibNRL* LibNRL::myInstance = new LibNRL();

LibNRL* LibNRL::getInstance(){

	return myInstance;
}

LibNRL::LibNRL() : dataPrefix("primitive") {

		modelNUllMsg = string("Error: the model was not created yet!");
		propPath = string("");

		dataset = nullptr;
		model = nullptr;
		logFile = nullptr;
		robot = nullptr;
		ut = Utils::getInstance();

		modelPath = string("");
		dataPath = string("");
		robotName = string("");
		networkName = string("");
		strEpoch = string("");
		propPath = string("");
		stdoutLog = false;

		nDof= 0;
		nSeq = 0;
		seqLen = 0;

		loaded = false;;
		dsoft = 10;
		sigma = 0.2;
		maxLoss = std::numeric_limits<float>::max();

		// variables for training mode
		t_nEpoch = 0;
		t_step = 1;
		t_alpha = 0.001;
		t_beta1 = 0.9;
		t_beta2 = 0.999;
		t_shuffle = false;
		t_retrain = false;
		t_greedy= false;

		// variables for experiment mode
		e_winSize = 0;
		e_nEpoch = 0;
		e_step = 1;
		e_alpha = 0.1;
		e_beta1 = 0.9;
		e_beta2 = 0.999;

		}


	void LibNRL::newModel(string path){

		deallocate();

		propPath = path;

		map<string,float1DContainer> float1DMap;
		map<string,string> stringMap;
		map<string,bool> boolMap;

		loaded = false;
		dsoft = 10;
		sigma = 0.2;

		t_retrain = false;
		t_nEpoch = 0;
		t_greedy = false;
		t_beta1 = 0.9;
		t_beta2 = 0.999;
		t_alpha = 0.001;
		t_step=1;

		e_nEpoch = 0;
		e_step = 1;
		e_beta1 = 0.9;
		e_beta2 = 0.999;
		e_alpha = 0.1;
		e_winSize = 0;

		maxLoss = std::numeric_limits<float>::max();
		logFile = 0;
		stdoutLog = false;

		try{

			ut->getProperties(float1DMap, stringMap, boolMap, propPath);

			if(stringMap.find("modelpath") == stringMap.end()) throw Exception("'modelpath' property not found");
			modelPath = stringMap["modelpath"];

			if(stringMap.find("datapath") == stringMap.end()) throw Exception("'datapath' property not found");
			dataPath = stringMap["datapath"];

			if(stringMap.find("network") == stringMap.end()) throw Exception("'network' property not found");
			networkName = stringMap["network"];

			if(stringMap.find("robot") == stringMap.end()) throw Exception("'robot' property not found");
			robotName = stringMap["robot"];

			if(boolMap.find("shuffle") == boolMap.end()) throw Exception("'shuffle' property not found");
			t_shuffle = boolMap["shuffle"];

			if(boolMap.find("retrain") == boolMap.end()) throw Exception("'retrain' property not found");
			t_retrain = boolMap["retrain"];

			if(boolMap.find("greedy") == boolMap.end()) throw Exception("'greedy' property not found");
			t_greedy = boolMap["greedy"];

			if(float1DMap.find("epochs") == float1DMap.end()) throw Exception("'epochs' property not found");
			t_nEpoch = int(float1DMap["epochs"][0]);

			if(float1DMap.find("dsoft") == float1DMap.end()) throw Exception("'dsoft' property not found");
			dsoft = float1DMap["dsoft"][0];

			if(float1DMap.find("sigma") == float1DMap.end()) throw Exception("'sigma' property not found");
			sigma = float1DMap["sigma"][0];

			if(float1DMap.find("beta1") == float1DMap.end()) throw Exception("'beta1' property not found");
			t_beta1 = float1DMap["beta1"][0];

			if(float1DMap.find("beta2") == float1DMap.end()) throw Exception("'beta2' property not found");
			t_beta2 = float1DMap["beta2"][0];

			if(float1DMap.find("alpha") == float1DMap.end()) throw Exception("'alpha' property not found");
			t_alpha = float1DMap["alpha"][0];

			if(float1DMap.find("activejoints") == float1DMap.end()) throw Exception("'activejoints' property not found");
			float1DContainer fActiveJoints = float1DMap["activejoints"];
			for (unsigned int i = 0; i < fActiveJoints.size(); i++)
				activeJoints.push_back(int(fActiveJoints[i])==1);

			if(float1DMap.find("nsamples") == float1DMap.end()) throw Exception("'nsamples' property not found");
			float1DContainer fSamples = float1DMap["nsamples"];
			for (unsigned int i = 0; i < fSamples.size(); i++)
				nSamples.push_back(int(fSamples[i]));

			stringstream stream;
			stream << modelPath << "/epoch.d";
			strEpoch = stream.str();

			if (t_retrain){
				ifstream eFile(strEpoch);
				if (!eFile.is_open()){
					cout << "Warning: the file [" << strEpoch << "] is unavailable, retrain set false";
					t_retrain = false;
				}
				else{
					float1DContainer vec;
					ut->loadContainer<float1DContainer>(&eFile, &vec, ut->getDelimiter());
					t_step = int(vec[0]+1);
					maxLoss = vec[1];
				}
			}

			if (robotName == "torobo")
				robot = Torobo::getInstance(activeJoints);
			else if (robotName == "cartesian")
				robot = Cartesian::getInstance(activeJoints);
			else if (robotName == "generic")
				robot = Generic::getInstance(activeJoints);
			else{
				stringstream stream;
				stream << "unknown 'robot' property [" << robotName << "]";
				throw Exception(stream.str());
			}

			dataset = new oist::Dataset(dataPath, dataPrefix, nSamples, dsoft, sigma, ut, robot);
			dataset->encodeSoftmax(YSoftmax);

			nSeq = nSamples.size();

			t_prim_Ids.clear();
			for (int i = 0; i < nSeq; i++)
				t_prim_Ids.push_back(i);

			nDof = ((float)robot->getDOF())*1.0;
			seqLen = dataset->getPrimLength();

			if (networkName == "pvrnn")
				model = new NetworkPvrnn(float1DMap, dataset);
			else if (networkName == "pvrnnbeta")
				model = new NetworkPvrnnBeta(float1DMap, dataset);
			else{
				stringstream stream;
				stream << "unknown 'network' property [" << networkName << "]";
				throw Exception(stream.str());
			}


		}catch(Exception& _e){
			cout << "Error: " << _e.what() << endl;
			deallocate();
		}catch(...){
			cout << "Error: unknown exception when extracting data from properties "<< endl;
			deallocate();
		}



	}
	int LibNRL::getNDof(){
		if (model == nullptr){
			cout << modelNUllMsg << endl;
			return -1;
		}
		if (loaded == false){
			cout << "Warning: The model should be loaded before calling 'getNDof()'!" << endl;
			return -1;
		}
		return nDof;
	}

	int LibNRL::getNLayers(){
		if (model == nullptr){
			cout << modelNUllMsg << endl;
			return -1;
		}
		if (loaded == false){
			cout << "Warning: The model should be loaded before calling 'getNLayers()'!" << endl;
			return -1;
		}
		return model->getNLayers();
	}

	int LibNRL::getStateDim(){
		if (model == nullptr){
			cout << modelNUllMsg << endl;
			return -1;
		}
		if (loaded == false){
			cout << "Warning: The model should be loaded before calling 'getStateDim()'!" << endl;
				return -1;
			}
			return model->getStateDim();
	}

	void LibNRL::load(){
		if (model == nullptr){
			cout << modelNUllMsg << endl;
			return;
		}
		try{
			model->load(string(modelPath));
			loaded = true;
		}catch(oist::Exception& e){
			cout << "Error: "<<  e.what() << endl;
		}
	}

	// ------------- training mode ---------------------

	void LibNRL::t_init(bool show){
		if (model == nullptr){
			cout << modelNUllMsg << endl;
			return;
		}
		stdoutLog = show;

		t_step = 1;
		stringstream strm; strm << modelPath << "/training.txt";

		if(t_retrain)
			logFile = new ofstream(strm.str(),std::ofstream::app);
		else
			logFile = new ofstream(strm.str(),std::ofstream::out);

		if (!logFile->is_open()){
			cout << "Warning: the log file ["<< strm.str() << "] could not be opened." << endl;
		}

		try{
			if (t_retrain){
				ifstream eFile(strEpoch);
				if (!eFile.is_open()){
					cout << "Warning: the file [" << strEpoch << "] is unavailable, retrain set false";
					t_retrain = false;
				}
				else{
					float1DContainer vec;
					ut->loadContainer<float1DContainer>(&eFile, &vec, ut->getDelimiter());
					t_step = int(vec[0]+1);
					maxLoss = vec[1];
					model->load(modelPath);
					//net.print();
					cout << endl << "Retraining the model ..." << endl;
					*logFile << "Retraining the model ..." << endl;
				}
			}else{
				cout << endl << "Training from scratch ..." << endl;
				*logFile << "Training from scratch ..." << endl;
			}
		}catch(Exception& _e){
			cout << "Error: " << _e.what() << endl;
		}catch(...){
			cout << "Error: unknown exception when extracting data from properties "<< endl;
		}
	}

	void LibNRL::t_loop(float* output, int n){

		if (model == nullptr){
			cout << modelNUllMsg << endl;
			return;
		}
		bool keepRunning = true;

		t_nEpoch = t_step + n - 1;
		if (keepRunning){
			  float loss = 0.0;
			  float reconstruction = 0.0;
			  float regulation = 0.0;

			  chrono::high_resolution_clock::time_point mst1 = chrono::high_resolution_clock::now();

			  for (; t_step <= t_nEpoch; t_step++){
				  vectorXf3DContainer All_X;

				  if (t_shuffle)
					  ut->shuffle<int1DContainer>(&t_prim_Ids);

				  int1DContainer::iterator t_prim_Ids_i = t_prim_Ids.begin();
				  for (int pId = 0; pId < nSeq; pId++, t_prim_Ids_i++){
					  vectorXf2DContainer X;
					  model->t_forward(seqLen, *t_prim_Ids_i, X);
					  All_X.push_back(X);
				  }

				  t_prim_Ids_i = t_prim_Ids.begin();
				  loss = 0.0;
				  reconstruction = 0.0;
				  regulation = 0.0;
				  for (int pId = 0; pId < nSeq; pId++, t_prim_Ids_i++){
					  vectorXf3DContainer Y_p = YSoftmax[*t_prim_Ids_i];
					  vectorXf2DContainer X = All_X[pId];
					  model->t_backward(*t_prim_Ids_i, X, Y_p, reconstruction, regulation, loss);
				  }
				  model->t_optAdam(t_step, t_alpha, t_beta1, t_beta2);

				  if (t_step % n == 0){
					  float mseGen = 0.0;
					  for (int pId = 0; pId < nSeq; pId++){
						  vectorXf2DContainer X;
						  vectorXf3DContainer Y_p = YSoftmax[pId];
						  model->t_generate(seqLen, pId, X);
						  mseGen +=  model->getRecError(X, Y_p);
					  }

					  chrono::high_resolution_clock::time_point mst2 = chrono::high_resolution_clock::now();
					  chrono::duration<double, std::milli> msdiff = mst2 - mst1;

					  float oTime = (float)msdiff.count();

					  if (stdoutLog == true){
						  cout << "Epoch ["<< t_step <<
							  "] - Time ["<< oTime << 
							  "ms] - RE_Q ["<< reconstruction << 
							  "] - RE_P ["<< mseGen << 
							  "] - Regulation ["<< regulation << 
							  "] - loss [" << loss << "]" << endl;
					  }

					  *logFile<< "Epoch ["<< t_step <<
						     "] - Time ["<< oTime << 
						     "ms] - RE_Q ["<< reconstruction << 
						     "] - RE_P ["<< mseGen << 
						     "] - Regulation ["<< regulation << 
						     "] - loss [" << loss << "]" << endl;

					  output[0] = t_step;
					  output[1] = oTime;
					  output[2] = reconstruction;
					  output[3] = mseGen;
					  output[4] = regulation;
					  output[5] = loss;
					  output[6] = 0.0;

					  if (maxLoss > loss || !t_greedy){
						  maxLoss = loss;
						  model->save(modelPath);
						  output[6] = 1.0;
						  ofstream eFile(strEpoch,std::ofstream::out);
						  if (eFile.is_open()){
							  eFile << t_step << ut->getDelimiter() << loss;
							  eFile.close();
						  }
						  *logFile<< "The model has been saved" << endl;
					  }
					  mst1 = chrono::high_resolution_clock::now();
				  }

			  }
		 }
	}

	void LibNRL::t_end(){

		if (model == nullptr){
			cout << modelNUllMsg << endl;
			return;
		}
		cout << endl << "Training end" << endl;
		*logFile << endl << "Training  end" << endl;
		logFile->close();

	}

	void LibNRL::t_background(){

		if (model == nullptr){
			cout << modelNUllMsg << endl;
			return;
		}
		bool keepRunning = true;
		t_step = 1;

		stringstream strm; strm << modelPath << "/training.txt";
		if(t_retrain)
			logFile = new ofstream(strm.str(),std::ofstream::app);
		else
			logFile = new ofstream(strm.str(),std::ofstream::out);

		if (!logFile->is_open()){
			cout << "Warning: the log file ["<< strm.str() << "] could not be opened." << endl;
		}

		try{
			if (t_retrain){
				ifstream eFile(strEpoch);
				if (!eFile.is_open()){
					cout << "Warning: the file [" << strEpoch << "] is unavailable, retrain set false";
					t_retrain = false;
				}
				else{
					float1DContainer vec;
					ut->loadContainer<float1DContainer>(&eFile, &vec, ut->getDelimiter());
					t_step = int(vec[0]+1);
					maxLoss = vec[1];
					model->load(modelPath);
					//net.print();
					cout << endl << "Retraining the model ..." << endl;
					*logFile << "Retraining the model ..." << endl;
				}
			}else{
				cout << endl << "Training from scratch ..." << endl;
				*logFile << "Training from scratch ..." << endl;
			}
		}catch(Exception& _e){
			cout << "Error: " << _e.what() << endl;
			keepRunning = false;
		}catch(...){
			cout << "Error: unknown exception when extracting data from properties "<< endl;
			keepRunning = false;
		}


		if (keepRunning){

			try{
				
			  float loss = 0.0;
			  float reconstruction = 0.0;
			  float regulation = 0.0;

			  chrono::high_resolution_clock::time_point mst1 = chrono::high_resolution_clock::now();

			  for (; t_step <= t_nEpoch; t_step++){
				  vectorXf3DContainer All_X;

				  if (t_shuffle)
					  ut->shuffle<int1DContainer>(&t_prim_Ids);

				  int1DContainer::iterator t_prim_Ids_i = t_prim_Ids.begin();
				  for (int pId = 0; pId < nSeq; pId++, t_prim_Ids_i++){
					  vectorXf2DContainer X;
					  model->t_forward(seqLen, *t_prim_Ids_i, X);
					  All_X.push_back(X);
				  }

				  t_prim_Ids_i = t_prim_Ids.begin();
				  loss = 0.0;
				  reconstruction = 0.0;
				  regulation = 0.0;
				  for (int pId = 0; pId < nSeq; pId++, t_prim_Ids_i++){
					  vectorXf3DContainer Y_p = YSoftmax[*t_prim_Ids_i];
					  vectorXf2DContainer X = All_X[pId];
					  model->t_backward(*t_prim_Ids_i, X, Y_p, reconstruction, regulation, loss);
				  }
				  model->t_optAdam(t_step, t_alpha, t_beta1, t_beta2);

				  if (t_step % 100 == 0){
					  float mseGen = 0.0;
					  for (int pId = 0; pId < nSeq; pId++){
						  vectorXf2DContainer X;
						  vectorXf3DContainer Y_p = YSoftmax[pId];
						  model->t_generate(seqLen, pId, X);
						  mseGen +=  model->getRecError(X, Y_p);
					  }

					  chrono::high_resolution_clock::time_point mst2 = chrono::high_resolution_clock::now();
					  chrono::duration<double, std::milli> msdiff = mst2 - mst1;
					  float oTime = (float)msdiff.count();

					  cout << "Epoch ["<< t_step <<
						  "] - Time ["<< oTime << 
						  "ms] - RE_Q ["<< reconstruction << 
						  "] - RE_P ["<< mseGen << 
						  "] - Regulation ["<< regulation << 
						  "] - loss [" << loss << "]" << endl;

					  *logFile<< "Epoch ["<< t_step <<
						     "] - Time ["<< oTime << 
						     "ms] - RE_Q ["<< reconstruction << 
						     "] - RE_P ["<< mseGen << 
						     "] - Regulation ["<< regulation << 
						     "] - loss [" << loss << "]" << endl;
					
					  if (maxLoss > loss || !t_greedy){
						  maxLoss = loss;
						  model->save(modelPath);
						  ofstream eFile(strEpoch,std::ofstream::out);
						  if (eFile.is_open()){
							  eFile << t_step << ut->getDelimiter() << loss;
							  eFile.close();
						  }
						  *logFile<< "The model has been saved" << endl;
					  }
					  mst1 = chrono::high_resolution_clock::now();
				  }

			  }			  
			  if (maxLoss > loss || !t_greedy){
				  maxLoss = loss;
				  model->save(modelPath);
				  ofstream eFile(strEpoch,std::ofstream::out);
				  if (eFile.is_open()){
					  eFile << t_step << ut->getDelimiter() << loss;
					  eFile.close();
				  }
				  *logFile<< "The model has been saved" << endl;
			  }
		  }catch(oist::Exception& e){
			  cout << "Error: "<<  e.what() << endl;
			  *logFile << "Error: "<<  e.what() << endl;
		  }
		}
		cout << endl << "Training end" << endl;
		*logFile << endl << "Training  end" << endl;
		logFile->close();
	}

	// ------------- experiment mode ---------------------



	void LibNRL::e_enable(int pID, int ws, float* param, int ne, int epoch, float alpha, float beta1, float beta2, bool store_s, bool store_p){

		if (model == nullptr){
			cout << modelNUllMsg << endl;
			return;
		}
		if (loaded == false){
			cout << "Warning: The model should be loaded before calling e_enable!" << endl;
			return;
		}
		e_winSize = ws;

		if (pID > nSeq - 1){
			cout << "Warning: the primitive ID "<< pID << " is greater than " << nSeq-1 << " available IDs. Primitive 0 is selected by default" << endl;
			pID = 0;
		}
		e_nEpoch = epoch;

		e_alpha = alpha;
		e_beta1 = beta1;
		e_beta2 = beta2;
		e_step = 1;
		model->e_enable(pID, e_winSize, param, ne, store_s, store_p);

	}

	void LibNRL::e_postdict(float* input, float* output, bool show){
		if (model == nullptr){
			cout << modelNUllMsg << endl;
			return;
		}
		if (loaded == false){
			cout << "Warning: The model should be loaded before calling e_postdict!" << endl;
			return;
		}

		vectorXf2DContainer Y;
		int size []= {e_winSize, (int)nDof};

		dataset->encodeSoftmax(input, size, Y);

		float maxLoss = std::numeric_limits<float>::max();

		float loss = 0.0;
		float rec = 0.0;
		float reg = 0.0;

		if(model->e_initForward()){

			for (int e1 = 1 ; e1 <= e_nEpoch; e1++, e_step++){
				loss = 0.0;
				rec = 0.0;
				reg = 0.0;

				vectorXf2DContainer X;
				model->e_forward(X);
				model->e_backward(X, Y, rec, reg, loss);
				if (show)
					cout << "E[" << e_step << "]" << " REC[" << rec << "] " << " REG[" << reg << "] loss[" << loss << "]" << endl;
				model->e_optAdam(e_step, e_alpha, e_beta1, e_beta2);

				if (maxLoss > loss){
					  output[0] = loss;
					  output[1] = rec;
					  output[2] = reg;
					  maxLoss = loss;
					  model->e_copyParam();
				}
			}
			model->e_overwriteParam();
		}
	}

	void LibNRL::e_generate(float* output){
		if (model == nullptr){
			cout << modelNUllMsg << endl;
			return;
		}
		if (loaded == false){
			cout << "Warning: The model should be loaded before calling e_generate!" << endl;
			return;
		}
		model->e_generate(output);
	}

	void LibNRL::e_save(string path){
		if (model == nullptr){
			cout << modelNUllMsg << endl;
			return;
		}
		if(loaded == false){
			cout << "Warning: The model should be loaded before calling e_save!" << endl;
			return;
		}
		model->e_save(path);
	}


	void LibNRL::e_getState(float* output){
		if (model == nullptr){
			cout << modelNUllMsg << endl;
			return;
		}
		if (loaded == false){
			cout << "Warning: The model should be loaded before calling e_getState!" << endl;
			return;
		}
		model->e_getState(output);

	}

	// ------------- Analysis mode ---------------------

	void LibNRL::a_predict(const char* path, int n, float* input){
		if (model == nullptr){
			cout << modelNUllMsg << endl;
			return;
		}
		if (loaded == false){
			cout << "Warning: The model should be loaded before calling a_predict!" << endl;
			return;
		}
		model->a_predict(n, input, string(path));
	}

	void LibNRL::a_feedForwardOutputFromContext(float* input, float* output){
		if (model == nullptr){
			cout << modelNUllMsg << endl;
			return;
		}
		if (loaded == false){
			cout << "Warning: The model should be loaded before calling a_getOutputFromState!" << endl;
			return;
		}
		model->a_feedForwardOutputFromContext(input, output);
	}

	void LibNRL::deallocate(){

		loaded = false;

		if (model == nullptr){
			return;
		}
		cout << endl << "Model deallocation ..." << endl;

		try{
			// clearing data
			YSoftmax.clear();
			activeJoints.clear();
			nSamples.clear();

			if (dataset != nullptr)
				delete dataset;
			if (model != nullptr)
				delete model;
			if (logFile != nullptr)
				delete logFile;

			model = nullptr;
			dataset = nullptr;		
			logFile = nullptr;	
			cout << endl;

		}catch(...){
			cout << "Error: unknown exception occurred when deallocating the model "<< endl;
		}
	}

	LibNRL::~LibNRL(){
		cout << "LibNRL destroyed " << endl;
	}

} /* namespace oist */

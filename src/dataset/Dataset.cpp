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

#include "Dataset.h"

namespace oist {

Dataset::Dataset(string _path, string _dataPrefix, int1DContainer& _samples, float _dsoft, float _sigma, Utils* _ut, IRobot* _robot){

	// setting path and number os samples
	ut = _ut;
	path = _path;
	dataPrefix = _dataPrefix;
	dsoft = _dsoft;
	sigma = _sigma;
	nPrims = _samples.size();
	copy(_samples.begin(), _samples.end(), back_inserter(nSamples));

	nDof = _robot->getDOF();
	_robot->getJointHighLimit(jmax);
	_robot->getJointLowLimit(jmin);
	_robot->getJointRange(jrange);

	// setting the encoding resolution
	encDim = 0;
	seqLen = 0;

	for (int j = 0; j< nDof; j ++){
		int r = (int)ceil(jrange[j]/dsoft);
		encDim += r;
		nUnits.push_back(r);
		float ri = 0.0;
		float linStep = 1.0/((float)((r-1)*1.0));
		float1DContainer ref_i;
		for (int i = 0; i < r ; i++ ){
			ref_i.push_back(ri);
			ri += linStep;
		}
		ref.push_back(ref_i);
	}

}

void Dataset::loadData(string _path, string _dataPrefix, int _nPrims, int1DContainer _nSamples, float4DContainer& _data){

	for (int p = 0; p < _nPrims; p++){
		float3DContainer d_p;
		for (int s = 0; s < nSamples[p]; s++){
			stringstream stream;
			stream << _path << "/" << dataPrefix << "_" << p << "_" << s << ".csv";
			std::ifstream file(stream.str());
			if (!file.good()){
				stringstream stream2;
				stream2 << "The file [" << stream.str() << "] could not be opened. Hint: check if the parameter 'nsamples' in the model property file corresponds to the dataset samples'" << endl;
				throw  Exception(stream2.str());				
			}
			std::string line = "";
			float2DContainer d_ps;

			// Iterate through each line and split the content using delimiter
			int nLine = 0;
			while (getline(file, line)){
				nLine ++;
				float1DContainer d_pst;

				std::vector<string> vec;
				ut->split(vec, line);
			
				if (nDof > (int)vec.size()){
					stringstream stream;
					stream << "Error: Please check the data row delimiter '" << ut->getDelimiter() << "'. Its was obtained less data that available Degrees of Freedom at line #" << nLine << endl;
					throw  Exception(stream.str());
				}
				for (std::vector<string>::iterator j = vec.begin() ; j != vec.end(); j++)
					d_pst.push_back(stof(*j));
				d_ps.push_back(d_pst);
			}
			if ((int)d_ps.size() > seqLen)
				seqLen = d_ps.size();

			d_p.push_back(d_ps);
			file.close();
		}
		_data.push_back(d_p);
	}
	if (seqLen == 0){
		stringstream stream;
		stream << "A sequence with length zero was found. Please check the dataset directory path. Alternatively, make sure all the sequences were recorded!" << endl;
		throw  Exception(stream.str());
	}

	cout << "Dataset loaded. Primitive number: " << _data.size() << ", length: " << seqLen << " steps" << endl;

}

int Dataset::getNPrim(){
	return nPrims;
}
int Dataset::getPrimLength(){
	return seqLen;
}

void Dataset::getNunitsPerDim(int1DContainer& _vec){
	for (int j = 0; j < nDof ; j++)
		_vec.push_back(nUnits[j]);
}

void Dataset::encodeSoftmax(float* input, int* size, vectorXf2DContainer& output){

	float sigma2 = sigma*sigma;
	int nT = size[0];
	int nDof = size[1];

	for (int t = 0; t < nT; t++){

		vectorXf1DContainer encData_t;
		float* d_t = input+(t*nDof);

		float2DContainer::iterator ref_ = ref.begin();
		float1DContainer::iterator jmin_ = jmin.begin();
		float1DContainer::iterator jrange_ = jrange.begin();
		int1DContainer::iterator encU_ = nUnits.begin();

		for (int j = 0; j < nDof; j++, ref_++, jmin_++, jrange_++, encU_++){

			float d_tj = *(d_t+j);
			float v = ((d_tj - (*jmin_))/(*jrange_)) + NON_ZERO;
			int encUnits = *encU_;

			float normalization = 0.0;
			float1DContainer::iterator ref_j = ref_->begin();
			VectorXf encPSJ = VectorXf::Zero(encUnits);
			auto dv_ = encPSJ.data();
			for (int r = 0; r < encUnits; r++, ref_j++, dv_++){
				float softmax = exp(-(pow((*ref_j)- v, 2.0))/sigma2);
				*dv_ = softmax;
				normalization += softmax;
			}

			encPSJ/= normalization;
			encData_t.push_back(encPSJ);

		}
		output.push_back(encData_t);
	}
}

void Dataset::softmax(float2DContainer& input, vectorXf2DContainer& output){

	float sigma2 = sigma*sigma;
	int nT = input.size();

	for (int t = 0; t < nT; t++){

		float1DContainer d_t = input[t];
		vectorXf1DContainer encUnits_j;

		for (int j = 0; j < nDof; j++){

			int encUnits = nUnits[j];

			VectorXf encPSJ = VectorXf::Zero(encUnits);

			float d_tj = d_t[j];

			vector<float> ref_j = ref[j];

			float v = (d_tj - jmin[j])/jrange[j];

			float normalization = 0.0;

			for (int r = 0; r < encUnits; r++){

				float softmax = exp(-(pow(ref_j[r]- v, 2.0))/sigma2);
				encPSJ(r) = softmax;
				normalization += softmax;
			}

			// normalizing the distributions
			for (int r = 0; r < encUnits; r++){
				encPSJ(r) /= normalization;
			}
			encUnits_j.push_back(encPSJ);

		}
		output.push_back(encUnits_j);
	}
}

void Dataset::encodeSoftmax(vectorXf4DContainer& output){

	float4DContainer dec_data;

	loadData(path, dataPrefix, nPrims, nSamples, dec_data);

	for (int p = 0; p < nPrims; p++){

		vectorXf3DContainer enc_p;
		float3DContainer dec_p = dec_data[p];

		for (unsigned int s = 0; s <dec_p.size(); s++){
			float2DContainer dec_ps = dec_p[s];
			vectorXf2DContainer enc_ps;
			softmax(dec_ps, enc_ps);
			enc_p.push_back(enc_ps);
		}
		output.push_back(enc_p);
	}

}


Dataset::~Dataset() {
	cout << "Dataset deallocated" << endl;
}

} /* namespace oist */

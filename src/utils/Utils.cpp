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

#include "Utils.h"

namespace oist {

std::default_random_engine Utils::generator = std::default_random_engine();
std::normal_distribution<float> Utils::distribution = std::normal_distribution<float>(0.0,1.0);

Utils* Utils::myInstance = nullptr;


Utils::Utils() : delimiter(","){	
	generator.seed(0);

}

Utils* Utils::getInstance(){

	if (myInstance == nullptr){
		myInstance = new Utils();
	}
	return myInstance;
}

void Utils::split(vector<string>& vec, string str){
	vec.clear();
	istringstream f(str.c_str());
	string s;
	while (getline(f, s, delimiter[0])) {
		vec.push_back(s);
	}
}

void Utils::ltrim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
    str.erase(0, str.find_first_not_of(chars));
}

void Utils::rtrim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
    str.erase(str.find_last_not_of(chars) + 1);
}

void Utils::trim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
	ltrim(str, chars);
	rtrim(str, chars);
}

void Utils::tolower(std::string& str){
	std::transform(str.begin(), str.end(), str.begin(),
	    [](unsigned char c){ return std::tolower(c); });
}


void Utils::getProperties(map<string,float1DContainer>& _mapFloat1D, map<string,string>& _mapString, map<string,bool>& _mapBool, string _path){

	string path = _path;
	if (path.empty()){
		path = string("config/properties.d");
	}

	ifstream propFile(path);
	if (!propFile.is_open())
		throw Exception("The properties file could not be opened");

	string line;
	std::string::size_type sz;
	std::string token;
	size_t pos = 0;
	string delim1 = "=";

	//cout << "Configurations: "<< endl;
	//cout << "------------------------" << endl;

	while (getline(propFile, line)){

		if (line.size() == 0)
			continue;
		pos = line.find(delim1);
		string key = line.substr(0, pos);
		trim(key);
		tolower(key);

		if (key.at(0) == '#')
			continue;

		//cout << line << endl;

		line.erase(0, pos + delim1.length());

		if (key == "modelpath"){
			trim(line);
			_mapString["modelpath"] = line;
			continue;
		}
		else if (key == "datapath"){
			trim(line);
			_mapString["datapath"] = line;
			continue;
		}
		else if (key == "network"){
			trim(line);
			_mapString["network"] = line;
			continue;
		}
		else if (key == "robot"){
			trim(line);
			_mapString["robot"] = line;
			continue;
		}
		else if (key == "shuffle"){
			trim(line);
			_mapBool["shuffle"] = (line == "true");
			continue;
		}
		else if (key == "retrain"){
			trim(line);
			_mapBool["retrain"] = (line == "true");
			continue;
		}
		else if (key == "greedy"){
			trim(line);
			_mapBool["greedy"] = (line == "true");
			continue;
		}

		float1DContainer value;

		while ((pos = line.find(delimiter)) != std::string::npos) {
			 token = line.substr(0, pos);
			 try{
				 value.push_back(std::stof(token,&sz));
			 }
			 catch (const std::invalid_argument& ia) {
				  std::cerr << "Invalid argument: " << ia.what() << '\n';
			 }catch (const std::out_of_range& oor) {
				  std::cerr << "Out of Range error: " << oor.what() << " token ["<< token <<  "]" << endl;
			 }
			 line.erase(0, pos + delimiter.length());
		 }

		try{
			value.push_back(std::stof(line,&sz));
		}catch (const std::invalid_argument& ia) {
			  std::cerr << "Invalid argument: " << ia.what() << '\n';
		}catch (const std::out_of_range& oor) {
		  std::cerr << "Out of Range error: " << oor.what() << " token ["<< token <<  "]" << endl;
		}
		_mapFloat1D[key] = value;
	}
	//cout << "------------------------" << endl;

	propFile.close();

}

string Utils::getDelimiter(){

	return delimiter;
}

MatrixXf Utils::kaiming_uniform_initialization(int _d0, int _d1, nonlinearity _nl){

	float gain = 0.0;
	float negative_slope = sqrt(5.0);

	switch (_nl){
		case nonlinearity::Linear:
		case nonlinearity::Conv:
		case nonlinearity::Sigmoid: 	gain = 1.0; break;
		case nonlinearity::Tanh: 	gain = 5.0/3.0; break;
		case nonlinearity::ReLU: 	gain = sqrt(2.0); break;
		case nonlinearity::ReLU_Leaky: 	gain = sqrt(2.0 / (1 + (negative_slope*negative_slope))); break;
	}

	float stdev = gain / sqrt(((float)_d1));

	// Calculating uniform bounds from standard deviation
	float bound = sqrt(3.0) * stdev ;
	MatrixXf W = MatrixXf::Random(_d0,_d1)*bound;
	return W;

}

VectorXf Utils::kaiming_uniform_initialization(int _d){

	 float bound = 1.0 / sqrt(((float)_d));
	 VectorXf bias = VectorXf::Random(_d)*bound;
	 return bias;

}


Utils::~Utils() {
	if (myInstance != nullptr)
		delete myInstance;
	cout << "Utils deallocated" << endl;
}

}/* namespace oist */


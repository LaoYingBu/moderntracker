#ifndef mt_h
#define mt_h

#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <cstring>
#include <cfloat>
using namespace std;

#include "Eigen/Dense"
using namespace Eigen;

const float PI_2 = acos(-1.0f) * 0.5f;

/*
//a faster configuration
const int cell_min = 1;
const int fast_n = 16;
const int fast_step = 4;
const int fine_n = 81;
const int cell_n = 81;
const int fine_steps[] = { 60, 20, 5, 1 };
const int detect_interval = 20;
const int max_iteration = 4;
const float padding = 1.6f;
const float sigmoid_factor = 7.141f;
const float sigmoid_bias = 0.482f;
const float translate_eps = 1.0f;
const float error_eps = 0.01f;
const float fine_threshold = 0.4f;
const float fast_threshold = 0.6f;
const float detect_threshold = 0.6f;
*/

//each SURF feature is 4 cells, each cell covers max(area_of_face / cell_n, 4 * cell_min^2) pixels
const int cell_min = 1;
const int cell_n = 150;

//fast template contatins about fast_n features	
const int fast_n = 25;
//the fast template moves at fast_step pixels
const int fast_step = 2;
//the search area of fast template is (1 + padding) * area_of_face
const float padding = 1.6f;
//when error is lower than threshold, carry out fast search
const float fast_threshold = 0.5f;

//fine template contatins about fine_n features
const int fine_n = 361;
//steps of multi-scale Lucas-Kanade
const int fine_steps[] = { 27, 9, 3, 1 };
//if the error is lower than threshold, update the fine model
const float fine_threshold = 0.4f;

//The tracker try to call the detector if the average error over the last interval frames is lower than the threshold	
const int detect_interval = 10;
const float detect_threshold = 0.4f;

//max number of iteration of each scale
const int max_iteration = 9;
//termination condition of iteration
const float translate_eps = 0.005f;
const float error_eps = 0.001f;

//sigmoid function
const float sigmoid_factor = 7.141f;
const float sigmoid_bias = 0.482f;

//types for Surf
typedef Matrix<float, 8, 1> Vector8f;
typedef Matrix<float, 32, 1> Vector32f;

//a simple rectangle class
typedef struct rect_t {
	float x;
	float y;
	float width;
	float height;
} rect_t;
//related tools
float rectArea(rect_t rect);
float rectOverlap(rect_t a, rect_t b);
ostream& operator<<(ostream& cout, const rect_t &rect);

//A simple multi-channel matrix
template<typename T, int channels>
class Data
{
public:
	Data(int rows, int cols)
	{
		this->rows = rows;
		this->cols = cols;
		step0 = channels;
		step1 = cols * step0;
		step2 = rows * step1;
		_data = new T[rows * cols * channels];
	}

	~Data()
	{
		delete _data;
	}

	void set(T val)
	{
		for (int i = 0; i < step2; ++i)
			_data[i] = val;
	}

	T* data()
	{
		return _data;
	}

	T* ptr(int row)
	{
		return _data + row * step1;
	}

	T* ptr(int row, int col)
	{
		return _data + row * step1 + col * step0;
	}

	T& operator()(int index)
	{
		return _data[index];
	}

	T& operator()(int row, int col)
	{
		return _data[row * step1 + col];
	}

	T& operator()(int row, int col, int channel)
	{
		return _data[row * step1 + col * step0 + channel];
	}

public:
	int rows, cols;
	int step0, step1, step2;

private:
	T* _data;
};

//Reconfigurable U-Surf
class Surf
{
public:
	//allocate storage for width * height gray image
	Surf(int width, int height);

	//rotate the image (virtually)
	void rotate(float angle, float kernel[]);
	//build the integral image
	void process(const unsigned char *gray, float angle);
	//change Surf settings
	void set_cell(float cell);
	void set_step(int step);

	//detailed computation
	inline float* cell_hist(int x, int y);
	inline void descriptor(float x, float y, float *f);
	inline void gradient(float x, float y, float *f, float *dx, float *dy);
	void descriptor4(float x, float y, float *f);
	void gradient4(float x, float y, float *f, float *dx, float *dy);

public:
	float A, X[4], Y[4];
	int W, H, C, step;

private:
	Data<float, 1> img;
	Data<int, 1> flag;
	Data<float, 8> sum, hist, zero;
};

//3D motion
class Warp
{
public:
	//set the focal length and image center
	Warp(int width, int height);

	//set rotation
	void setr(Vector3f rotate);
	//set translation
	void sett(Vector3f translate);

	//project 3D point to 2D coordinate
	Vector2f project(Vector3f p);
	//3D rigid transform
	Vector3f transform(Vector3f p);
	//3D rigid transform + project
	Vector2f transform2(Vector3f p);

	//(dx, dy) / (dR, dT)
	inline Vector2f gradient(Vector3f p, Matrix<float, 2, 6> &dW);
	//update
	void steepest(Matrix<float, 6, 1> parameters);
	//get eular angles
	void euler(float &roll, float &yaw, float &pitch);

public:
	Vector2f c;
	float f;
	Vector3f r;
	Vector3f t;

private:
	Matrix3f R, Dx, Dy, Dz;
};

//The tracker
class MT
{
public:
	//init a tracker, omit os to turn off the log
	MT(const unsigned char *gray, int width, int height, rect_t rect, ostream *os = NULL);
	//track a new frame
	rect_t track(const unsigned char *gray);
	//track a new frame with the help of some detection result
	rect_t retrack(const unsigned char *gray, const vector<rect_t> &detections);
	//whether the face is succesfully tracked now
	bool check();

private:
	Vector3f locate(rect_t rect);
	rect_t window(Vector3f translate);

	void update(Warp w, float e);
	void fast_train(Warp w);
	void fine_train(Warp w);
	Vector3f fast_test(Warp w);
	Warp fine_test(Warp w);

	inline float sigmoid(float x);
	inline void hessian(Matrix<float, 6, 6> &H, float w, const Matrix<float, 2, 6> &dW, const Matrix<float, 32, 2> &dF);

	Warp Lucas_Kanade(Warp w);
	float evaluate(Warp w);

public:
	int image_width, image_height;
	float window_width, window_height;
	Warp warp;
	float error, roll, yaw, pitch;
	int number_coarse, number_MLK, number_iteration;

private:
	Surf feature;
	vector<Vector2i> fast_samples;
	vector<Vector3f> fine_samples;
	MatrixXf fast_model, fine_model;
	deque<float> fine_errors;
	ostream *log;
	int N;
};

#endif /* mt_h */

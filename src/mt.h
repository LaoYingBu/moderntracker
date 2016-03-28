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
//a fast but not that accurate configuration
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

//the fast template contains about fast_n points
const int fast_n = 25;
//the fast template moves at fast_step pixels
const int fast_step = 2;
//the search area of fast template is (1 + padding) * area_of_face
const float padding = 1.6f;
//when error is lower than threshold, we carry out fast search
const float fast_threshold = 0.5f;

//the fine template contains about points
const int fine_n = 361;
//steps of multi-scale Lucas-Kanade
const int fine_steps[] = { 27, 9, 3, 1 };
//if the error is lower than threshold, we update the fine model
const float fine_threshold = 0.4f;

//The tracker try to call the detector if the average error over the last interval frames is lower than the threshold	
const int detect_interval = 10;
const float detect_threshold = 0.4f;

//the maximum number of iterations in each scale
const int max_iteration = 9;
//the termination conditions
const float translate_eps = 0.005f;
const float error_eps = 0.001f;

//the constant in the sigmoid function
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

	//generate the rotation kernel
	void rotate(float angle, float kernel[4]);	
	//build the integral image
	void process(const unsigned char *gray, float angle);
	//change cell and step
	void set_cell(float cell);
	void set_step(int step);

	//the histogram of gradient in a cell
	inline float* cell_hist(int x, int y);
	//the histogram in subpixel location using bilinear interpolation
	inline void descriptor(float x, float y, float *f);
	//the corresponding gradient, i.e. df / (dx, dy)
	inline void gradient(float x, float y, float *f, float *dx, float *dy);
	//the descriptor and gradient of 2x2 cells
	void descriptor4(float x, float y, float *f);
	void gradient4(float x, float y, float *f, float *dx, float *dy);

public:
	//the reconfigured locations of the 2x2 cells
	float A, X[4], Y[4];
	//the image width, image height, cell size and step size
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
	//get euler angles
	void euler(float &roll, float &yaw, float &pitch);

public:
	//the image center, focal length, rotaion vector and translation vector
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
	//initialize a tracker (omit os to turn off the log)
	MT(const unsigned char *gray, int width, int height, rect_t rect, ostream *os = NULL);
	//track a new frame
	rect_t track(const unsigned char *gray);
	//track a new frame with the help of some re-detection
	rect_t retrack(const unsigned char *gray, const vector<rect_t> &detections);
	//check whether the face is succesfully tracked now
	bool check();

private:
	//the conversion between the rectangle and the translation
	Vector3f locate(rect_t rect);
	rect_t window(Vector3f translate);

	//update the motion and the template depending on the error
	void update(Warp w, float e);
	//train and test the fast and fine templates
	void fast_train(Warp w);
	void fine_train(Warp w);
	Vector3f fast_test(Warp w);
	Warp fine_test(Warp w);

	//sigmoid function
	inline float sigmoid(float x);
	//handcraft hessian matrix computation
	inline void hessian(Matrix<float, 6, 6> &H, float w, const Matrix<float, 2, 6> &dW, const Matrix<float, 32, 2> &dF);

	//Lucas-Kanade algorithm
	Warp Lucas_Kanade(Warp w);
	//evaluate the motion
	float evaluate(Warp w);

public:
	//the image size, window size, current motion, current error, current euler angles
	int image_width, image_height;
	float window_width, window_height;
	Warp warp;
	float error, roll, yaw, pitch;
	//the number of coarse searching, number of multi-scale LK and number of iterations
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

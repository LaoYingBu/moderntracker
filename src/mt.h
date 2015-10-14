#pragma once

#include <opencv2\world.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\calib3d.hpp>
using namespace cv;

#include <iostream>
#include <cmath>
#include <cstring>
using namespace std;

const int L = 8, L4 = L * 4;
const float PI_2 = acos(-1.0f) * 0.5f;

const int cell_min = 2;
const int fast_n = 25;
const int fast_step = 2;
const int fine_n = 600;
const int cell_n = 150;
const int fine_steps[] = { 27, 9, 3, 1 };
const int max_iteration = 9;
const float padding = 1.6f;
const float sigmoid_factor = 7.2f;
const float sigmoid_bias = 0.48f;
const float translate_eps = 0.01f;
const float threshold_error = 0.4f;

class Surf
{
public:
	Surf(Size size);
	
	Matx14f kernel(float angle);
	void process(Mat gray, float angle);
	void set_cell(float cell);
	void set_step(int step);	

	float* cell_hist(int x, int y);
	float cell_norm(int x, int y);
	void descriptor(float x, float y, float *f);
	void gradient(float x, float y, float *f, float *dx, float *dy);
	void descriptor4(float x, float y, float *f);
	void gradient4(float x, float y, float *f, float *dx, float *dy);
	
public:
	float angle, tx[4], ty[4];
	Matx14f kx, ky;
	int W, H, C, step;

private:
	Mat grad, sum, zero, flag, hist, norm;
};

class Warp
{
public:	
	Warp(Size size);

	void set(Matx13f rotate);
	void set(Point3f translate);
			
	Point2f project(Point3f p);
	Point3f transform(Point3f p);
	Point2f transform2(Point3f p);

	Matx<float, 2, 6> gradient(Point3f p);
	void steepest(Matx61f parameters);
	void euler(float &roll, float &yaw, float &pitch);
				
public:
	Point2f c;
	float f;
	Matx13f r;
	Point3f t;	

private:
	Matx33f R, Dx, Dy, Dz;	
};

class MT
{
public:
	MT(Mat gray, Rect2f rect, ostream *os = NULL);

	void restart(Rect2f rect);
	Rect2f track(Mat gray);
	bool miss();
	
private:
	Point3f locate(Rect2f rect);
	Rect2f window(Point3f translate);
	
	void fast_train(Warp warp);
	void fine_train(Warp warp);
	Point3f fast_test(Warp warp);
	Warp fine_test(Warp warp);
		
	float sigmoid(float x);
	Warp Lucas_Kanade(Warp warp);
	float evaluate(Warp warp);

private:
	ostream *log;	
	Size image_size;
	Size2f window_size;
	Surf feature;
	Warp warp;	
	vector<Point3f> candidates;
	vector<Point> fast_samples;				
	vector<Point3f> fine_samples;
	Mat fast_model, fine_model;
	int failed, trained;

public:
	float error, roll, yaw, pitch;	
};

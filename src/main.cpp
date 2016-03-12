#include "common.h"

extern void run_benchmark();

/*
#include "fartracker.h"
#include <opencv2\calib3d.hpp>
class Warp2
{
public:
	Warp2(Size size);

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

class Surf2
{
public:
	Surf2(Size size);

	Matx14f kernel(float angle);
	void process(Mat img, float angle);
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

public:
	Mat grad, sum, zero, flag, hist, norm;
};

Warp2::Warp2(Size size) :
c(size.width * 0.5f, size.height * 0.5f),
f(float(max(size.width, size.height)))
{
	set(Point3f(0.0f, 0.0f, 0.0f));
	set(Matx13f(0.0f, 0.0f, 0.0f));
}

void Warp2::set(Matx13f rotate)
{
	r = rotate;
	Matx<float, 3, 9> J;
	Rodrigues(r, R, J);
	Dx = Matx33f(
		J(0, 0), J(1, 0), J(2, 0),
		J(0, 3), J(1, 3), J(2, 3),
		J(0, 6), J(1, 6), J(2, 6)
		);
	Dy = Matx33f(
		J(0, 1), J(1, 1), J(2, 1),
		J(0, 4), J(1, 4), J(2, 4),
		J(0, 7), J(1, 7), J(2, 7)
		);
	Dz = Matx33f(
		J(0, 2), J(1, 2), J(2, 2),
		J(0, 5), J(1, 5), J(2, 5),
		J(0, 8), J(1, 8), J(2, 8)
		);
}

void Warp2::set(Point3f translate)
{
	t = translate;
}

Point2f Warp2::project(Point3f p)
{
	return Point2f(p.x / p.z * f, p.y / p.z * f) + c;
}

Point3f Warp2::transform(Point3f p)
{
	Matx31f P = R * Matx31f(p.x, p.y, p.z);
	return Point3f(P(0), P(1), P(2)) + t;
}

Point2f Warp2::transform2(Point3f p)
{
	return project(transform(p));
}

Matx<float, 2, 6> Warp2::gradient(Point3f p)
{
	Matx33f D1 = p.x * Dx + p.y * Dy + p.z * Dz;
	Point3f tp = transform(p);
	Matx23f D2(
		f / tp.z, 0.0f, -f * tp.x / (tp.z * tp.z),
		0.0f, f / tp.z, -f * tp.y / (tp.z * tp.z)
		);
	Matx23f D3 = D2 * D1;
	return Matx<float, 2, 6>(
		D3(0, 0), D3(0, 1), D3(0, 2), D2(0, 0), D2(0, 1), D2(0, 2),
		D3(1, 0), D3(1, 1), D3(1, 2), D2(1, 0), D2(1, 1), D2(1, 2)
		);
}

void Warp2::steepest(Matx61f parameters)
{
	float rx = r(0) + parameters(0);
	float ry = r(1) + parameters(1);
	float rz = r(2) + parameters(2);
	float tx = t.x + parameters(3);
	float ty = t.y + parameters(4);
	float tz = t.z + parameters(5);
	set(Matx13f(rx, ry, rz));
	set(Point3f(tx, ty, tz));
}

void Warp2::euler(float &roll, float &yaw, float &pitch)
{
	if (abs(1 - abs(R(2, 1))) > 1.0e-7f) {
		roll = atan2(-R(0, 1), R(1, 1));
		yaw = atan2(-R(2, 0), R(2, 2));
		pitch = asin(R(2, 1));
	}
	else {
		roll = atan2(R(1, 0), R(0, 0));
		yaw = 0.0f;
		pitch = R(2, 1) > 0 ? PI_2 : -PI_2;
	}
}

const int L = 8, L4 = L * 4;

Surf2::Surf2(Size size)
{
	W = size.width;
	H = size.height;
	C = 0;
	step = 0;
	grad = Mat(H, W, CV_32FC(L));
	zero = Mat(1, L, CV_32FC1);
	flag = Mat(H, W, CV_32SC1);
	hist = Mat(H, W, CV_32FC(L));
	norm = Mat(H, W, CV_32FC1);
}

Matx14f Surf2::kernel(float angle)
{
	float c = cos(angle), s = sin(angle);
	float wx = c * (1.0f - abs(s));
	float wy = s * (1.0f - abs(c));
	float wu = 0.0f, wv = 0.0f;
	if (c >= 0)
		(s >= 0 ? wu : wv) = c * s;
	else
		(s >= 0 ? wv : wu) = -c * s;
	return Matx14f(wx, wy, wu, wv);
}

void Surf2::process(Mat img, float angle)
{
	this->angle = angle;
	Matx14f kx = kernel(angle), ky = kernel(angle + PI_2);
	grad = 0.0f;
	for (int y = 0; y < H; ++y) {
		int y0 = y > 0 ? y - 1 : y;
		int y1 = y < H - 1 ? y + 1 : y;
		uchar *ptr_y0 = img.ptr<uchar>(y0);
		uchar *ptr_y = img.ptr<uchar>(y);
		uchar *ptr_y1 = img.ptr<uchar>(y1);
		for (int x = 0; x < W; ++x) {
			int x0 = x > 0 ? x - 1 : x;
			int x1 = x < W - 1 ? x + 1 : x;
			float gx = float(ptr_y[x1]) - float(ptr_y[x0]);
			float gy = float(ptr_y1[x]) - float(ptr_y0[x]);
			float gu = float(ptr_y1[x1]) - float(ptr_y0[x0]);
			float gv = float(ptr_y1[x0]) - float(ptr_y0[x1]);
			Matx14f g(gx, gy, gu, gv);
			float dx = kx.dot(g), dy = ky.dot(g);
			float *f = grad.ptr<float>(y, x);
			switch ((dx > 0.0f ? 1 : 0) + (dy > 0.0f ? 2 : 0)) {
			case 0:
				f[0] = dx;
				f[2] = -dx;
				f[4] = dy;
				f[6] = -dy;
				break;
			case 1:
				f[0] = dx;
				f[2] = dx;
				f[5] = dy;
				f[7] = -dy;
				break;
			case 2:
				f[1] = dx;
				f[3] = -dx;
				f[4] = dy;
				f[6] = dy;
				break;
			case 3:
				f[1] = dx;
				f[3] = dx;
				f[5] = dy;
				f[7] = dy;
				break;
			}
		}
	}
	integral(grad, sum, CV_32F);
	C = 0;
	step = 0;
	flag = 0;
	zero = 0.0f;
}

void Surf2::set_cell(float cell)
{
	cell = cell * 0.5f;
	tx[0] = -cell; ty[0] = -cell;
	tx[1] = cell; ty[1] = -cell;
	tx[2] = cell; ty[2] = cell;
	tx[3] = -cell; ty[3] = cell;
	for (int i = 0; i < 4; ++i) {
		float x = cos(angle) * tx[i] - sin(angle) * ty[i];
		float y = sin(angle) * tx[i] + cos(angle) * ty[i];
		tx[i] = x;
		ty[i] = y;
	}
	C = max(int(floor(cell)), cell_min);
}

void Surf2::set_step(int step)
{
	this->step = step;
}

float* Surf2::cell_hist(int x, int y)
{
	if (x < 0 || x >= W || y < 0 || y >= H)
		return zero.ptr<float>();
	if (flag.at<int>(y, x) != C) {
		int x0 = max(x - C, 0);
		int x1 = min(x + C + 1, W);
		int y0 = max(y - C, 0);
		int y1 = min(y + C + 1, H);
		float *s00 = sum.ptr<float>(y0, x0);
		float *s01 = sum.ptr<float>(y0, x1);
		float *s10 = sum.ptr<float>(y1, x0);
		float *s11 = sum.ptr<float>(y1, x1);
		float *f = hist.ptr<float>(y, x);
		float S = 0.0f;
		for (int i = 0; i < L; ++i) {
			f[i] = s11[i] + s00[i] - s01[i] - s10[i];
			S += f[i] * f[i];
		}
		norm.at<float>(y, x) = S;
		flag.at<int>(y, x) = C;
	}
	return hist.ptr<float>(y, x);
}

float Surf2::cell_norm(int x, int y)
{
	if (x < 0 || x >= W || y < 0 || y >= H)
		return 0.0f;
	else
		return norm.at<float>(y, x);
}

void Surf2::descriptor(float x, float y, float *f)
{
	x = x / step;
	y = y / step;
	int ixp = (int)floor(x);
	int iyp = (int)floor(y);
	float wx1 = x - ixp, wx0 = 1.0f - wx1;
	float wy1 = y - iyp, wy0 = 1.0f - wy1;
	float w00 = wx0 * wy0;
	float w01 = wx0 * wy1;
	float w10 = wx1 * wy0;
	float w11 = wx1 * wy1;
	float *f00 = cell_hist(ixp * step, iyp * step);
	float *f01 = cell_hist(ixp * step, (iyp + 1) * step);
	float *f10 = cell_hist((ixp + 1) * step, iyp * step);
	float *f11 = cell_hist((ixp + 1) * step, (iyp + 1) * step);
	for (int i = 0; i < L; ++i)
		f[i] = f00[i] * w00 + f01[i] * w01 + f10[i] * w10 + f11[i] * w11;
}

void Surf2::gradient(float x, float y, float *f, float *dx, float *dy)
{
	x = x / step;
	y = y / step;
	int ixp = (int)floor(x);
	int iyp = (int)floor(y);
	float wx1 = x - ixp, wx0 = 1.0f - wx1;
	float wy1 = y - iyp, wy0 = 1.0f - wy1;
	float w00 = wx0 * wy0;
	float w01 = wx0 * wy1;
	float w10 = wx1 * wy0;
	float w11 = wx1 * wy1;
	float *f00 = cell_hist(ixp * step, iyp * step);
	float *f01 = cell_hist(ixp * step, (iyp + 1) * step);
	float *f10 = cell_hist((ixp + 1) * step, iyp * step);
	float *f11 = cell_hist((ixp + 1) * step, (iyp + 1) * step);
	wx0 /= step;
	wy0 /= step;
	wx1 /= step;
	wy1 /= step;
	for (int i = 0; i < L; ++i) {
		f[i] = f00[i] * w00 + f01[i] * w01 + f10[i] * w10 + f11[i] * w11;
		dx[i] = (f10[i] - f00[i]) * wy0 + (f11[i] - f01[i]) * wy1;
		dy[i] = (f01[i] - f00[i]) * wx0 + (f11[i] - f10[i]) * wx1;
	}
}

void Surf2::descriptor4(float x, float y, float *f)
{
	for (int i = 0; i < 4; ++i)
		descriptor(x + tx[i], y + ty[i], f + i * L);
	float S = 0.0f;
	for (int i = 0; i < L4; ++i)
		S += f[i] * f[i];
	float iS = S < 1.0f ? 0.0f : 1.0f / sqrt(S);
	for (int i = 0; i < L4; ++i)
		f[i] *= iS;
}

void Surf2::gradient4(float x, float y, float *f, float *dx, float *dy)
{
	for (int i = 0; i < 4; ++i)
		gradient(x + tx[i], y + ty[i], f + i * L, dx + i * L, dy + i * L);
	float S = 0.0f, Sx = 0.0f, Sy = 0.0f;
	for (int i = 0; i < L4; ++i) {
		S += f[i] * f[i];
		Sx += f[i] * dx[i];
		Sy += f[i] * dy[i];
	}
	float iS = S < 1.0f ? 0.0f : 1.0f / sqrt(S);
	float iSx = Sx * iS * iS * iS;
	float iSy = Sy * iS * iS * iS;
	for (int i = 0; i < L4; ++i) {
		dx[i] = dx[i] * iS - f[i] * iSx;
		dy[i] = dy[i] * iS - f[i] * iSy;
		f[i] *= iS;
	}
}

void test_wrap()
{
	int width = 640, height = 360;
	float roll, yaw, pitch;
	
	Warp w(width, height);
	w.euler(roll, yaw, pitch);
	cout << roll << " " << yaw << " " << pitch << endl;
	cout << w.gradient(Vector3f(0.0f, 0.0f, 1.0f)) << endl;
	cout << w.gradient(Vector3f(-1.0f, -1.0f, 10000.0f)) << endl;
	cout << w.gradient(Vector3f(1.0f, 2.0f, -3.0f)) << endl;	
	w.sett(Vector3f(-30.0f, 20.0f, 10.0f));
	w.setr(Vector3f(0.1f, -0.2f, 0.3f));	
	w.euler(roll, yaw, pitch);
	cout << roll << " " << yaw << " " << pitch << endl;
	cout << w.gradient(Vector3f(0.0f, 0.0f, 0.0f)) << endl;
	cout << w.gradient(Vector3f(-1.0f, -1.0f, 10000.0f)) << endl;
	cout << w.gradient(Vector3f(1.0f, 2.0f, -3.0f)) << endl;

	Warp2 w2(Size(width, height));
	w2.euler(roll, yaw, pitch);
	cout << roll << " " << yaw << " " << pitch << endl;
	cout << w2.gradient(Point3f(0.0f, 0.0f, 1.0f)) << endl;
	cout << w2.gradient(Point3f(-1.0f, -1.0f, 10000.0f)) << endl;
	cout << w2.gradient(Point3f(1.0f, 2.0f, -3.0f)) << endl;
	w2.set(Point3f(-30.0f, 20.0f, 10.0f));
	w2.set(Matx13f(0.1f, -0.2f, 0.3f));
	w2.euler(roll, yaw, pitch);
	cout << roll << " " << yaw << " " << pitch << endl;
	cout << w2.gradient(Point3f(0.0f, 0.0f, 0.0f)) << endl;
	cout << w2.gradient(Point3f(-1.0f, -1.0f, 10000.0f)) << endl;
	cout << w2.gradient(Point3f(1.0f, 2.0f, -3.0f)) << endl;

	system("pause");
}

void test_surf()
{
	Mat gray = imread(dir_common + "0001.jpg", 0);
	int width = gray.cols, height = gray.rows;	
	float px = width * 0.5f, py = height * 0.5f;

	Surf feature(width, height);	
	feature.process(gray.data, 0.0f);
	cout << feature.sum.col(height / 2).segment<8>(width / 2 * 8).transpose() << endl;
	cout << feature.sum.col(height).segment<8>(width * 8).transpose() << endl;
	Vector8f h, f, dx, dy;
	feature.set_cell(20);
	feature.set_step(3);	
	h = feature.cell_hist(px, py);
	feature.gradient(px, py, f, dx, dy);
	cout << h.transpose() << endl << f.transpose() << endl << dx.transpose() << endl << dy.transpose() << endl;

	Surf2 feature2(Size(width, height));
	feature2.process(gray, 0.0f);
	for (int i = 0; i < 8; ++i)
		cout << feature2.sum.at<float>(height / 2, width / 2 * 8 + i) << " ";
	cout << endl;
	for (int i = 0; i < 8; ++i)
		cout << feature2.sum.at<float>(height, width * 8 + i) << " ";
	cout << endl;
	Matx<float, 1, 8> h2, f2, dx2, dy2;	
	feature2.set_cell(20);
	feature2.set_step(3);
	memcpy(h2.val, feature2.cell_hist(px, py), 8 * sizeof(float));
	feature2.gradient(width * 0.5f, height * 0.5f, f2.val, dx2.val, dy2.val);
	cout << h2 << endl << f2 << endl << dx2 << endl << dy2 << endl;

	system("pause");
}

#include "Eigen/Dense"
using namespace Eigen;

inline void Eigen_hessian(Matrix<float, 6, 6> &H, float w, const Matrix<float, 2, 6> &dW, const Matrix<float, 32, 2> &dF)
{
	H += w * (dW.transpose() * (dF.transpose() * dF) * dW);
}

inline void Hand_hessian(Matrix<float, 6, 6> &H, float w, const Matrix<float, 2, 6> &dW, const Matrix<float, 32, 2> &dF)
{
	float F00 = w * dF.col(0).squaredNorm();
	float F11 = w * dF.col(1).squaredNorm();
	float F01 = w * dF.col(0).dot(dF.col(1));
	float x0 = dW(0, 0) * F00 + dW(1, 0) * F01;
	float y0 = dW(0, 0) * F01 + dW(1, 0) * F11;
	H(0, 0) += x0 * dW(0, 0) + y0 * dW(1, 0);
	H(0, 1) += x0 * dW(0, 1) + y0 * dW(1, 1);
	H(0, 2) += x0 * dW(0, 2) + y0 * dW(1, 2);
	H(0, 3) += x0 * dW(0, 3) + y0 * dW(1, 3);
	H(0, 4) += x0 * dW(0, 4) + y0 * dW(1, 4);
	H(0, 5) += x0 * dW(0, 5) + y0 * dW(1, 5);
	float x1 = dW(0, 1) * F00 + dW(1, 1) * F01;
	float y1 = dW(0, 1) * F01 + dW(1, 1) * F11;
	H(1, 1) += x1 * dW(0, 1) + y1 * dW(1, 1);
	H(1, 2) += x1 * dW(0, 2) + y1 * dW(1, 2);
	H(1, 3) += x1 * dW(0, 3) + y1 * dW(1, 3);
	H(1, 4) += x1 * dW(0, 4) + y1 * dW(1, 4);
	H(1, 5) += x1 * dW(0, 5) + y1 * dW(1, 5);
	float x2 = dW(0, 2) * F00 + dW(1, 2) * F01;
	float y2 = dW(0, 2) * F01 + dW(1, 2) * F11;
	H(2, 2) += x2 * dW(0, 2) + y2 * dW(1, 2);
	H(2, 3) += x2 * dW(0, 3) + y2 * dW(1, 3);
	H(2, 4) += x2 * dW(0, 4) + y2 * dW(1, 4);
	H(2, 5) += x2 * dW(0, 5) + y2 * dW(1, 5);
	float x3 = dW(0, 3) * F00 + dW(1, 3) * F01;
	float y3 = dW(0, 3) * F01 + dW(1, 3) * F11;
	H(3, 3) += x3 * dW(0, 3) + y3 * dW(1, 3);
	H(3, 4) += x3 * dW(0, 4) + y3 * dW(1, 4);
	H(3, 5) += x3 * dW(0, 5) + y3 * dW(1, 5);
	float x4 = dW(0, 4) * F00 + dW(1, 4) * F01;
	float y4 = dW(0, 4) * F01 + dW(1, 4) * F11;
	H(4, 4) += x4 * dW(0, 4) + y4 * dW(1, 4);
	H(4, 5) += x4 * dW(0, 5) + y4 * dW(1, 5);
	float x5 = dW(0, 5) * F00 + dW(1, 5) * F01;
	float y5 = dW(0, 5) * F01 + dW(1, 5) * F11;
	H(5, 5) += x5 * dW(0, 5) + y5 * dW(1, 5);
}

void test_hessian()
{	
	Matrix<float, 6, 6> H = Matrix<float, 6, 6>::Constant(0.0f);
	Matrix<float, 2, 6> dW;		
	Matrix<float, 32, 2> dF;
	const int N = 10000000;

	clock_t start_clock = clock();
	for (int i = 0; i < N; ++i) {
		float w = float(rand()) / float(RAND_MAX);
		dW.setRandom();
		dF.setRandom();		
		//Eigen_hessian(H, w, dW, dF);
		Hand_hessian(H, w, dW, dF);
	}
	clock_t end_clock = clock();
	cerr << H << endl;
	cerr << double(end_clock - start_clock) / N << endl;
	//system("pause");
}

void generate_code()
{
	freopen("common/hand_hessian.txt", "w", stdout);
	printf("float F00 = w * dF.col(0).squaredNorm();\n");
	printf("float F11 = w * dF.col(1).squaredNorm();\n");
	printf("float F01 = w * dF.col(0).dot(dF.col(1));\n");
	for (int i = 0; i < 6; ++i) {
		printf("float x%d = dW(0, %d) * F00 + dW(1, %d) * F01;\n", i, i, i);
		printf("float y%d = dW(0, %d) * F01 + dW(1, %d) * F11;\n", i, i, i);
		for (int j = i; j < 6; ++j) 
			printf("H(%d, %d) += x%d * dW(0, %d) + y%d * dW(1, %d);\n", i, j, i, j, i, j);		
	}
}
*/

int main(int argc, char **argv)
{
	expr = new Expr();
	if (argc > 1) {
		expr->load(argv[1]);
		cout << "Base configuration : " << argv[1] << endl;
	}
	for (int i = 2; i + 1 < argc; i += 2) {
		expr->edit(argv[i], argv[i + 1]);
		cout << "Edit param " << argv[i] << " = " << argv[i + 1] << endl;
	}
	if (strcmp(argv[argc - 1], "-preload") == 0) {
		Sequence *seq = NULL;
		while ((seq = Sequence::getSeq()) != NULL) {
			seq->loadImage();
			Sequence::setSeq(seq);			
		}
		Sequence::root.clear();
	}
	run_benchmark();
	delete expr;

	return 0;
}

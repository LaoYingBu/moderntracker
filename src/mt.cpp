#include "mt.h"

Surf::Surf(Size size)
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

Matx14f Surf::kernel(float angle)
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

void Surf::process(Mat img, float angle)
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

void Surf::set_cell(float cell)
{
	/*
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
	*/
	C = max(int(floor(cell)), cell_min);
}

void Surf::set_step(int step)
{
	this->step = step;
}

float* Surf::cell_hist(int x, int y)
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

float Surf::cell_norm(int x, int y)
{
	if (x < 0 || x >= W || y < 0 || y >= H)
		return 0.0f;
	else
		return norm.at<float>(y, x);
}

void Surf::descriptor(float x, float y, float *f)
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

void Surf::gradient(float x, float y, float *f, float *dx, float *dy)
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

void Surf::descriptor4(float x, float y, float *f)
{	
	descriptor(x, y, f);
	float S = 0.0f;
	for (int i = 0; i < L; ++i)
		S += f[i] * f[i];
	float iS = S < 1.0f ? 0.0f : 1.0f / sqrt(S);
	for (int i = 0; i < L; ++i)
		f[i] *= iS;
}

void Surf::gradient4(float x, float y, float *f, float *dx, float *dy)
{	
	gradient(x, y, f, dx, dy);
	float S = 0.0f, Sx = 0.0f, Sy = 0.0f;
	for (int i = 0; i < L; ++i) {
		S += f[i] * f[i];
		Sx += f[i] * dx[i];
		Sy += f[i] * dy[i];
	}
	float iS = S < 1.0f ? 0.0f : 1.0f / sqrt(S);
	float iSx = Sx * iS * iS * iS;
	float iSy = Sy * iS * iS * iS;
	for (int i = 0; i < L; ++i) {
		dx[i] = dx[i] * iS - f[i] * iSx;
		dy[i] = dy[i] * iS - f[i] * iSy;
		f[i] *= iS;
	}
}

Warp::Warp(Size size) :
c(size.width * 0.5f, size.height * 0.5f),
f(float(max(size.width, size.height)))
{
	set(Point3f(0.0f, 0.0f, 0.0f));
	set(Matx13f(0.0f, 0.0f, 0.0f));
}

void Warp::set(Matx13f rotate)
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

void Warp::set(Point3f translate)
{
	t = translate;
}

Point2f Warp::project(Point3f p)
{
	return Point2f(p.x / p.z * f, p.y / p.z * f) + c;
}

Point3f Warp::transform(Point3f p)
{
	Matx31f P = R * Matx31f(p.x, p.y, p.z);
	return Point3f(P(0), P(1), P(2)) + t;
}

Point2f Warp::transform2(Point3f p)
{
	return project(transform(p));
}

Matx<float, 2, 6> Warp::gradient(Point3f p)
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

void Warp::steepest(Matx61f parameters)
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

void Warp::euler(float &roll, float &yaw, float &pitch)
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

MT::MT(Mat img, Rect2f rect, ostream *os) :
log(os),
image_size(img.size()),
window_size(rect.size()),
feature(image_size),
warp(image_size)
{
	warp.set(locate(rect));
	float fine_stride = sqrt(window_size.area() / fine_n);
	int W = int(floor(window_size.width / (2.0f * fine_stride)));
	int H = int(floor(window_size.height / (2.0f * fine_stride)));
	for (int y = 0; y <= 2 * H; ++y)
	for (int x = 0; x <= 2 * W; ++x)
		fine_samples.push_back(Point3f((x - W) * fine_stride, (y - H) * fine_stride, 0.0f));

	feature.process(img, 0.0f);
	fine_train(warp);
	fast_train(warp);
	error = 0.0f;
	roll = yaw = pitch = 0.0f;
	count = 0;
}

bool MT::miss()
{
	return count >= 10;
}

void MT::restart(Rect2f rect)
{
	count = 0;
	candidates.push_back(locate(rect));
}

Rect2f MT::track(Mat img)
{
	if (log != NULL) {
		(*log) << "roll = " << roll * 90.0f / PI_2 << endl;
		(*log) << "yaw = " << yaw * 90.0f / PI_2 << endl;
		(*log) << "pitch = " << pitch * 90.0f / PI_2 << endl;
	}
	feature.process(img, roll);

	candidates.push_back(warp.t);
	candidates.push_back(fast_test(warp));

	Warp best_warp(image_size);
	float best_error = 1.0f;
	for (auto& t : candidates) {
		if (log != NULL)
			(*log) << "candidate " << t << " " << window(t) << endl;
		Warp w(image_size);
		w.set(warp.r);
		w.set(t);
		w = fine_test(w);
		float e = evaluate(w);
		if (log != NULL) {
			(*log) << "final translation = " << w.t << " " << window(w.t) << endl;
			(*log) << "final rotation = " << w.r << endl;
			(*log) << "final error = " << e << endl;
		}
		if (e < best_error) {
			best_warp = w;
			best_error = e;
		}
	}
	candidates.clear();

	error = best_error;
	if (error < threshold_error) {
		count = 0;
		warp = best_warp;
		warp.euler(roll, yaw, pitch);
		fine_train(warp);
	}
	else {
		++count;
		warp.t = best_warp.t * (warp.t.z / best_warp.t.z);
	}
	fast_train(warp);

	return window(best_warp.t);
}


Point3f MT::locate(Rect2f rect)
{
	float scale = sqrt(window_size.area() / rect.area());
	float x = rect.x + rect.width * 0.5f - warp.c.x;
	float y = rect.y + rect.height * 0.5f - warp.c.y;
	return Point3f(x, y, warp.f) * scale;
}

Rect2f MT::window(Point3f translate)
{
	Point2f center = warp.project(translate);
	float scale = warp.f / translate.z;
	float w = window_size.width * scale;
	float h = window_size.height * scale;
	float x = center.x - w * 0.5f;
	float y = center.y - h * 0.5f;
	return Rect2f(x, y, w, h);
}

void MT::fast_train(Warp warp)
{
	Rect2f rect = window(warp.t);
	float fast_stride = sqrt(rect.area() / fast_n);
	feature.set_cell(fast_stride);
	int W = int(floor(rect.width * 0.5f / fast_stride));
	int H = int(floor(rect.height * 0.5f / fast_stride));
	int ox = int(round(rect.width * 0.5f));
	int oy = int(round(rect.height * 0.5f));
	int stride = int(round(fast_stride));
	fast_samples.clear();
	for (int y = 0; y <= 2 * H; ++y)
	for (int x = 0; x <= 2 * W; ++x)
		fast_samples.push_back(Point(ox + (x - W) * stride, oy + (y - H) * stride));

	fast_model.create(fast_samples.size(), L, CV_32FC1);
	int x = int(round(rect.x));
	int y = int(round(rect.y));
	for (int i = 0; i < fast_samples.size(); ++i) {
		int tx = x + fast_samples[i].x;
		int ty = y + fast_samples[i].y;
		float *dst = fast_model.ptr<float>(i);
		float *src = feature.cell_hist(tx, ty);
		memcpy(dst, src, L * sizeof(float));
	}
}

void MT::fine_train(Warp warp)
{
	Rect2f rect = window(warp.t);
	float fine_cell = sqrt(rect.area() / cell_n);
	feature.set_cell(fine_cell);
	feature.set_step(1);

	Mat model(fine_samples.size(), L, CV_32FC1);
	for (int i = 0; i < fine_samples.size(); ++i) {
		Point2f p = warp.transform2(fine_samples[i]);
		feature.descriptor4(p.x, p.y, model.ptr<float>(i));
	}
	if (fine_model.empty())
		fine_model = model;
	else
		fine_model = (1.0f - interp_factor) * fine_model + interp_factor * model;
}

Point3f MT::fast_test(Warp warp)
{
	Rect2f rect = window(warp.t);
	float fast_stride = sqrt(rect.area() / fast_n);
	feature.set_cell(fast_stride);
	Rect2f region = window(warp.t / (1.0f + padding));
	float minminx = -rect.width * 0.5f;
	float minminy = -rect.height * 0.5f;
	float maxmaxx = image_size.width + rect.width * 0.5f;
	float maxmaxy = image_size.height + rect.height * 0.5f;
	int minx = int(round(max(region.x, minminx)));
	int miny = int(round(max(region.y, minminy)));
	int maxx = int(round(min(region.x + region.width, maxmaxx) - rect.width));
	int maxy = int(round(min(region.y + region.height, maxmaxy) - rect.height));

	float best_score = 0.0f;
	Point3f best_translate = warp.t;
	for (int y = miny; y <= maxy; y += fast_step)
	for (int x = minx; x <= maxx; x += fast_step) {
		float S = 0.0f, score = 0.0f;
		for (int i = 0; i < fast_samples.size(); ++i) {
			int tx = x + fast_samples[i].x;
			int ty = y + fast_samples[i].y;
			float *f = fast_model.ptr<float>(i);
			float *g = feature.cell_hist(tx, ty);
			S += feature.cell_norm(tx, ty);
			for (int j = 0; j < L; ++j)
				score += f[j] * g[j];
		}
		score *= S < 1.0f ? 0.0f : 1.0f / sqrt(S);
		if (score > best_score) {
			best_translate = locate(Rect2f(float(x), float(y), rect.width, rect.height));
			best_score = score;
		}
	}
	return best_translate;
}

Warp MT::fine_test(Warp warp)
{
	Rect2f rect = window(warp.t);
	float fine_cell = sqrt(rect.area() / cell_n);
	feature.set_cell(fine_cell);
	for (auto fine_step : fine_steps) {
		if (fine_step > 2.0f * fine_cell)
			continue;
		feature.set_step(fine_step);
		if (log != NULL)
			(*log) << "\tcell = " << fine_cell << " step = " << fine_step << endl;
		warp = Lucas_Kanade(warp);
	}
	return warp;
}

float MT::sigmoid(float x)
{
	return 1.0f / (1.0f + exp(-sigmoid_factor * (x - sigmoid_bias)));
}

Warp MT::Lucas_Kanade(Warp warp)
{
	for (int iter = 0; iter < max_iteration; ++iter) {
		Matx61f G;
		Matx<float, 6, 6> H;
		G = 0.0f;
		H = 0.0f;
		float E = 0.0f;
		for (int i = 0; i < fine_samples.size(); ++i) {
			Matx<float, L, 1> T(fine_model.ptr<float>(i)), F;
			Matx<float, 2, L> dF;
			Matx<float, 2, 6> dW = warp.gradient(fine_samples[i]);
			Point2f p = warp.transform2(fine_samples[i]);
			feature.gradient4(p.x, p.y, F.val, dF.val, dF.val + L);
			T -= F;
			float e = sigmoid(T.dot(T));
			E += e;
			float w = sigmoid_factor * e * (1.0f - e);
			G += w * (dW.t() * (dF * T));
			H += w * (dW.t() * (dF * dF.t()) * dW);
		}


		E = E / fine_samples.size();
		if (log != NULL)
			(*log) << "\terror in iteration " << iter << " = " << E << endl;
		Matx61f D;
		solve(H, G, D, DECOMP_SVD);
		warp.steepest(D);
		if (iter > 1 && D(3) * D(3) + D(4) * D(4) + D(5) * D(5) < translate_eps)
			break;
	}
	return warp;
}

float MT::evaluate(Warp warp)
{
	Rect2f rect = window(warp.t);
	float fine_cell = sqrt(rect.area() / cell_n);
	feature.set_cell(fine_cell);
	feature.set_step(1);

	float E = 0.0f;
	for (int i = 0; i < fine_samples.size(); ++i) {
		Matx<float, L, 1> T(fine_model.ptr<float>(i)), I;
		Point2f p = warp.transform2(fine_samples[i]);
		feature.descriptor4(p.x, p.y, I.val);
		T -= I;
		E = E + sigmoid(T.dot(T));
	}
	return E / fine_samples.size();
}

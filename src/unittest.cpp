#include "common.h"
#include "mt.h"

namespace unittest
{
	const string dir_result = "./unittest/";
	const string path_img = dir_result + "lena.jpg";
	const Rect gt(264, 372, 269, 269);	

	void draw(Mat &img, Rect2f rect, float angle, Scalar color)
	{
		Point2f rcenter(rect.x + rect.width * 0.5f, rect.y + rect.height * 0.5f);
		RotatedRect rrect(rcenter, rect.size(), angle);
		Point2f p[4];
		rrect.points(p);
		for (int i = 0; i < 4; ++i)
			line(img, p[i], p[(i + 1) % 4], color, 5);
	}

	void main()
	{
		ofstream log(dir_result + "log.txt");

		Mat img = imread(path_img, 1), blurred, warped, gray;
		float dx = 0.0f, dy = 0.0f, scale = 1.0f, angle = 0.0f;						
		
		MT *mt = NULL;
		Statistics st(true);
		for (int i = 1; i <= 2500; ++i) {
			log << endl << "Frame " << i << " " << endl;

			int KL = 3 + 2 * theRNG().uniform(0, 13);
			switch (theRNG().uniform(0, 3)) {
			case 0:
				blur(img, blurred, Size(KL, KL));
				break; 
			case 1:
				GaussianBlur(img, blurred, Size(KL, KL), 0, 0);
				break;
			case 2:
				medianBlur(img, blurred, KL);
				break;
			case 3:
				bilateralFilter(img, blurred, KL, KL * 2, KL / 2);
				break;
			}
			Point2f center(gt.x + 0.5f * gt.width, gt.y + 0.5f * gt.height);
			Mat M = getRotationMatrix2D(center, -angle, scale);
			M.at<double>(0, 2) += dx;
			M.at<double>(1, 2) += dy;
			warpAffine(blurred, warped, M, img.size(), INTER_LINEAR | WARP_FILL_OUTLIERS);
			cvtColor(warped, gray, CV_BGR2GRAY);
						
			float width = gt.width * scale;
			float height = gt.height * scale;
			Rect2f rect(center.x + dx - width * 0.5f, center.y + dy - height * 0.5f, width, height), ret;

			if (mt == NULL)
				mt = new MT(gray, ret = rect, &log);
			else
				ret = mt->track(gray);
			float ret_angle = mt->roll * 90.0f / PI_2;
			st.frame(rect, ret, NULL);
			log << "result : " << ret << "(" << ret_angle << ")" << endl;
			log << "groundtruth : " << rect << "(" << angle << ")" << endl;
			log << "error : " << mt->error << endl;
			float score = overlap(ret, rect);
			log << "score : " << score << endl;
			if (score < 0.5f)
				log << "Wrong!!!" << endl;

			draw(warped, rect, angle, Scalar(255, 255, 255));
			draw(warped, ret, ret_angle, Scalar(255, 0, 0));
			stringstream ss;			
			ss << dir_result << i << ".jpg";
			imwrite(ss.str(), warped);
										
			float timing = 1.0f + float(i) / 500;
			dx = dx + theRNG().gaussian(scale * timing * 20);
			dy = dy + theRNG().gaussian(scale * timing * 20);
			while (center.x + dx - gt.width * scale * 0.5f < 0)
				++dx;
			while (center.x + dx + gt.width * scale * 0.5f >= img.cols)
				--dx;
			while (center.y + dy - gt.height * scale * 0.5f < 0)
				++dy;
			while (center.y + dy + gt.height * scale * 0.5f >= img.rows)
				--dy;
			scale = scale + theRNG().gaussian(0.02 * timing);
			if (scale > 3)
				scale = 3;
			if (scale < 0.3)
				scale = 0.3;
			angle = angle + theRNG().gaussian(3);
		}
		delete mt;
	
		log << st << endl;
		log.close();
	}
}

void run_unittest()
{
	unittest::main();
}

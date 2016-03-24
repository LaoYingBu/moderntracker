#include "common.h"

namespace video {
	const string title = "Face Tracking -- Mo Tao";	
	const string path_log = dir_log + "video.txt";

	void main(VideoCapture &reader)
	{				
		if (!dir_log.empty())
			mkdir(dir_log);
		ofstream flog(path_log);
		namedWindow(title);
		moveWindow(title, 1, 1);
		
		int width = cvRound(reader.get(CAP_PROP_FRAME_WIDTH));
		int height = cvRound(reader.get(CAP_PROP_FRAME_HEIGHT));		
		float image_scale = sqrt(float(resolution_width * resolution_height) / float(width * height));
		if (image_scale < 1.0f)	{
			width = cvRound(width * image_scale);
			height = cvRound(height * image_scale);
		}
		flog << "Tracking in " << width << "x" << height << endl;
									
		int idx = 0, last_detect = 0, nDetection = 0;
		Mat raw, img, gray;
		CascadeClassifier detector(detector_model);
		vector<Rect2f> detections;
		MT *mt = NULL;
		Rect2f rect;		
		
		while (reader.read(raw)) {			
			++idx;
			flog << "Frame " << idx << endl;
			if (raw.cols != width || raw.rows != height)
				resize(raw, img, Size(width, height));
			else
				img = raw;
			cvtColor(img, gray, CV_BGR2GRAY);			

			if ((idx == 1 || idx - last_detect >= detector_frequence) && (mt == NULL || !mt->check())) {
				detect(detector, gray, detections);
				flog << "Detect " << detections.size() << " faces " << endl;
				for (auto d : detections)
					flog << "\t" << d << endl;
				last_detect = idx;
				++nDetection;				
			}
			else
				detections.clear();
			if (mt == NULL) {
				if (!detections.empty()) {
					rect = detections.front();
					for (auto d : detections)
					if (d.area() > rect.area())
						rect = d;
					mt = new MT(gray.data, gray.cols, gray.rows, cv2mt(rect), &flog);
					flog << "track start at " << rect << endl;
				}
			}
			else {
				if (!detections.empty()) {
					vector<rect_t> mt_detections;
					for (auto d : detections)
						mt_detections.push_back(cv2mt(d));
					rect = mt2cv(mt->retrack(gray.data, mt_detections));					
				}
				else
					rect = mt2cv(mt->track(gray.data));
				flog << "track at " << rect << endl;
			}
			
			stringstream ss;
			(ss << fixed).precision(2);			
			ss << "detection: " << nDetection;
			putText(img, ss.str(), Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5 , Scalar(0, 0, 255));
			if (mt != NULL) {
				ss.str("");
				ss << "error: " << mt->error * 90 / PI_2;
				putText(img, ss.str(), Point(0, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));				
				ss.str("");
				ss << "roll: " << mt->roll * 90 / PI_2;
				putText(img, ss.str(), Point(0, 45), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
				ss.str("");
				ss << "yaw: " << mt->yaw * 90 / PI_2;
				putText(img, ss.str(), Point(0, 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
				ss.str("");
				ss << "pitch: " << mt->pitch * 90 / PI_2;
				putText(img, ss.str(), Point(0, 75), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

				Point2f center(rect.x + rect.width * 0.5f, rect.y + rect.height * 0.5f);
				RotatedRect rrect(center, rect.size(), mt->roll * 90 / PI_2);
				Point2f p[4];
				rrect.points(p);				
				for (int i = 0; i < 4; ++i)
					line(img, p[i], p[(i + 1) % 4], Scalar(0, 0, 255), 3);
			}
			imshow(title, img);			
			waitKey(1);
		}
	}
}

void run_video(string path)
{
	if (path == "camera") {
		cout << "Run camera tracking" << endl;
		video::main(VideoCapture(0));
	}
	else {
		cout << "Run video tracking on " << path << endl;
		video::main(VideoCapture(path));
	}
}

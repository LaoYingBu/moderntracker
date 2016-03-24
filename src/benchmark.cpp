#include "common.h"

namespace benchmark
{		
	const string path_log = dir_log + "benchmark.txt";

	mutex M_global;
	Statistics global_short(false), global_long(false), global_whole(false);
	ofstream global_log;

	void invoker(int threadID)
	{		
		CascadeClassifier detector(detector_model);
		Sequence *seq = NULL;		
		int cnt = 0;
		while ((seq = Sequence::getSeq()) != NULL) {									
			stringstream ss;
			ss << "[" << seq->getType() << "]" << seq->getName();
			ss << "(" << seq->getStart() << "-" << seq->getEnd() << ")";
			ss << seq->getWidth() << "x" << seq->getHeight();
			string alias = ss.str();
			bool verbose = !dir_log.empty();

			ofstream log;
			if (verbose) {
				log.open(dir_log + alias + ".txt");
				log << alias << endl;
			}
			M_global.lock();
			cout << threadID << ": " << alias << endl;
			M_global.unlock();
			
			seq->loadImage();
			MT *mt = NULL;
			int last_detect = 0;						

			Statistics st(true);
			st.tic();
			for (int i = seq->getStart(); i <= seq->getEnd(); ++i) {
				if (verbose) {
					log << endl << "Frame " << i << " ";
					log << (seq->getClear(i) ? "Clear" : "Unclear") << endl;
				}
				Mat gray = seq->getImage(i);
				Rect2f gt = seq->getRect(i), result;
				vector<Rect2f> detections;

				if (mt == NULL) {
					result = gt;			
					mt = new MT(gray.data, gray.cols, gray.rows, cv2mt(result), verbose ? &log : NULL);
				}
				else {					
					if (!mt->check() && i - last_detect >= detector_frequence) {
						detect(detector, gray, detections);	
						last_detect = i;
					}										
					if (detections.empty()) 
						result = mt2cv(mt->track(gray.data));
					else {
						vector<rect_t> mt_detections;
						for (auto d : detections)
							mt_detections.push_back(cv2mt(d));
						result = mt2cv(mt->retrack(gray.data, mt_detections));
					}
				}				

				float error = mt->error;
				if (verbose) {
					log << "result : " << Rect(int(result.x), int(result.y), int(result.width), int(result.height)) << endl;
					if (seq->getClear(i))
						log << "groundtruth : " << Rect(int(gt.x), int(gt.y), int(gt.width), int(gt.height)) << endl;
					log << "error : " << error << endl;
					if (seq->getClear(i)) {
						float score = overlap(gt, result);
						log << "score : " << score << endl;
						if (score < 0.5f)
							log << "Wrong!!!" << endl;
					}
				}
				st.track(gt, result, error < fine_threshold, last_detect == i, mt->number_coarse, mt->number_MLK, mt->number_iteration);
				seq->setRect(i, result);
			}			
			st.toc();
			if (verbose) {
				log << endl << st << endl;
				log.close();
			}

			M_global.lock();
			global_log << alias << endl << st << endl;
			if (seq->getType() == "short") {
				global_short += st;
				global_log << "Short benchmark : " << endl << global_short << endl;
			}
			if (seq->getType() == "long") {
				global_long += st;
				global_log << "Long benchmark : " << endl << global_long << endl;
			}
			if (seq->getType() == "whole") {
				global_whole += st;
				global_log << "Whole benchmark : " << endl << global_whole << endl;
			}
			M_global.unlock();

			delete mt;
			Sequence::setSeq(seq);
		}
	}

	void main()
	{		
		if (!dir_log.empty())
			mkdir(dir_log);
		global_log.open(path_log);		

		setNumThreads(0);
		vector<thread> ths;
		for (int i = 0; i < 3; ++i)
			ths.push_back(thread(&invoker, i));
		for (auto& th : ths)
			th.join();
			
		if (!global_short.empty())
			global_log << "Short benchmark : " << endl << global_short << endl;
		if (!global_long.empty())
			global_log << "Long benchmark : " << endl << global_long << endl;
		if (!global_whole.empty())
			global_log << "Whole benchmark : " << endl << global_whole << endl;
		global_whole += global_short;
		global_whole += global_long;
		if (!global_whole.empty())
			global_log << "Overall : " << endl << global_whole << endl;
		global_log.close();
	}
}

void run_benchmark()
{	
	cout << "Run benchmark" << endl;
	benchmark::main();
}

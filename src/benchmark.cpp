#include "common.h"
#include "exprtracker.h"

namespace benchmark
{		
	mutex M_global;
	Statistics global_short(false), global_long(false), global_whole(false);
	ofstream global_log;

	far_rect_t cv2far(Rect2f rect)
	{
		far_rect_t _rect;
		_rect.x = rect.x;
		_rect.y = rect.y;
		_rect.width = rect.width;
		_rect.height = rect.height;
		return _rect;
	}

	Rect2f far2cv(far_rect_t rect)
	{
		Rect2f _rect;
		_rect.x = rect.x;
		_rect.y = rect.y;
		_rect.width = rect.width;
		_rect.height = rect.height;
		return _rect;		
	}

	void detect(CascadeClassifier &detector, Mat gray, vector<Rect2f> &rects)
	{		
		vector<Rect> faces;
		detector.detectMultiScale(gray, faces);
		rects.clear();
		for (auto& r : faces) {
			float w = 0.78f * r.width;
			float h = 0.78f * r.height;
			float x = r.x + r.width * 0.5f - w * 0.5f;
			float y = r.y + r.height * 0.55f - h * 0.5f;
			rects.push_back(Rect2f(x, y, w, h));
		}		
	}

	void invoker(int threadID)
	{		
		CascadeClassifier detector(expr->detector_model);
		Sequence *seq = NULL;		
		int cnt = 0;
		while ((seq = Sequence::getSeq()) != NULL) {									
			stringstream ss;
			ss << "[" << seq->getType() << "]" << seq->getName();
			ss << "(" << seq->getStart() << "-" << seq->getEnd() << ")";
			ss << seq->getWidth() << "x" << seq->getHeight();
			string alias = ss.str();
			bool verbose = !expr->dir_detail.empty();

			ofstream log;
			if (verbose) {
				log.open(expr->dir_detail + alias + ".txt");
				log << alias << endl;
			}
			M_global.lock();
			cout << threadID << ": " << alias << endl;
			M_global.unlock();
			
			seq->loadImage();
			ExprTracker *fart = NULL;
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

				if (fart == NULL) {
					result = gt;			
					fart = new ExprTracker(gray.data, gray.cols, gray.rows, cv2far(result), verbose ? &log : NULL);
				}
				else {					
					if (!fart->check() && i - last_detect >= expr->detector_frequence) {
						detect(detector, gray, detections);	
						last_detect = i;
					}										
					if (detections.empty()) 
						result = far2cv(fart->track(gray.data));
					else {
						vector<far_rect_t> far_detections;
						for (auto d : detections)
							far_detections.push_back(cv2far(d));
						result = far2cv(fart->retrack(gray.data, far_detections));
					}
				}				

				float error = fart->error;
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
				st.track(gt, result, error < expr->fine_threshold, fart->number_MLK, fart->number_iteration);
				if (fart->is_fine)
					st.fine_track(fart->final_choice, gt, far2cv(fart->fine_start));
				if (fart->is_fast)
					st.fast_track(fart->final_choice, gt, far2cv(fart->fast_start));
				if (fart->is_detect)
					st.detect_track(fart->final_choice, gt, far2cv(fart->detect_start));
				else
				if (i == last_detect)
					st.detect_track(fart->final_choice, gt, Rect2f(0.0f, 0.0f, 0.0f, 0.0f));
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

			delete fart;
			Sequence::setSeq(seq);
		}
	}

	void main()
	{		
		if (!expr->dir_detail.empty())
			mkdir(expr->dir_detail);
		global_log.open(expr->path_log);
		global_log << "Base configuration : " << expr->base_configuration << endl;
		global_log << expr->save() << endl << endl;

		setNumThreads(0);
		vector<thread> ths;
		for (int i = 0; i < expr->nThreads; ++i)
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
	cout << "Run benchmark with " << expr->nThreads << " threads" << endl;
	benchmark::main();
}

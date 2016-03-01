#include "common.h"
#include "fartracker.h"

namespace benchmark
{	
	const string dir_log = "./benchmark/";
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

	void invoker(int threadID)
	{		
		Sequence *seq = NULL;		
		int cnt = 0;
		while ((seq = Sequence::getSeq()) != NULL) {									
			stringstream ss;
			ss << "[" << seq->getType() << "]" << seq->getName();
			ss << "(" << seq->getStart() << "-" << seq->getEnd() << ")";
			ss << seq->getWidth() << "x" << seq->getHeight();
			string alias = ss.str();
			
			ofstream log(dir_log + alias + ".txt");
			log << alias << endl;
			M_global.lock();
			cout << threadID << ": " << alias << endl;
			M_global.unlock();
			
			seq->loadImage();
			FARTracker *fart = NULL;			
			int last_detect = -1;						

			Statistics st(true);
			for (int i = seq->getStart(); i <= seq->getEnd(); ++i) {
				log << endl << "Frame " << i << " ";
				log << (seq->getClear(i) ? "Clear" : "Unclear") << endl;
				Mat gray = seq->getImage(i);
				Rect2f gt = seq->getRect(i), result;
				vector<Rect2f> detections;

				if (fart == NULL) {
					result = gt;			
					fart = new FARTracker(gray.data, gray.cols, gray.rows, cv2far(result), &log);
				}
				else {					
					if (!fart->check() && i - last_detect >= 30) {
						detect(gray, detections);
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

				log << "result : " << Rect(result) << endl;
				if (seq->getClear(i)) 
					log << "groundtruth : " << Rect(gt) << endl;
				float error = fart->error;
				log << "error : " << error << endl;
				if (seq->getClear(i)) {
					float score = overlap(gt, result);
					log << "score : " << score << endl;
					if (score < 0.5f)
						log << "Wrong!!!" << endl;
				}									
				if (last_detect != i)
					st.track(gt, result, error < threshold_error);
				else
					st.retrack(gt, result, error < threshold_error, detections);
				seq->setRect(i, result);
			}
			log << endl << st << endl;
			log.close();

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
		mkdir(dir_log);
		global_log.open(dir_log + "log.txt");

		vector<thread> ths;
		for (int i = 0; i < nThreads; ++i)
			ths.push_back(thread(&invoker, i));
		for (auto& th : ths)
			th.join();
			
		if (!global_short.empty())
			global_log << "Short benchmark : " << endl << global_short << endl;
		if (!global_long.empty())
			global_log << "Long benchmark : " << endl << global_long << endl;
		if (!global_whole.empty())
			global_log << "Whole benchmark : " << endl << global_whole << endl;
		global_log.close();
	}
}

void run_benchmark()
{	
	cout << "Run benchmark with " << nThreads << " threads" << endl;
	benchmark::main();
}

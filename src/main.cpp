#include "common.h"

extern void run_benchmark();
extern void run_unittest();
extern void run_camera();
extern void run_video(string path, int direction);

int main(int argc, char **argv)
{			
	Mat gray = imread("D:/face tracking/benchmark/image/ab3_4/0244.jpg", 0);
	Detector detector;
	vector<Rect2f> rects;
	detector.detect(gray, rects);

	string help = "Possible commands:\n";
	help += ".\\mt benchmark";
	help += ".\\mt unittest";
	help += ".\\mt camera";
	help += ".\\mt video path direction";
	if (argc < 2) {
		cerr << help << endl;
		return 0;
	}

	if (strcmp(argv[1], "benchmark") == 0)
		run_benchmark();
	else
	if (strcmp(argv[1], "unittest") == 0)
		run_unittest();
	else
	if (strcmp(argv[1], "camera") == 0)
		run_camera();
	else
	if (strcmp(argv[1], "video") == 0 && argc == 4)
		run_video(string(argv[2]), atoi(argv[3]));
	else
		cerr << help << endl;

	return 0;
}

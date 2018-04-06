#include<iostream>
#include<opencv2\opencv.hpp>

/*
	Function: myCannyFilter
	Description:
		This function finds edges in an image using Canny algortihm
	Parameters:
		iImg: input image (single channel 32bit float image)
		oImg: output edge image (single channels 8bit image)
		thr1: first threshold for the hysteresis procedure
		thr2: second threshold for the hysteresis procedure
		process: if true, process images are outputed to result directory
*/
void myCannyFilter(cv::Mat& iImg, cv::Mat& oImg, float thr1, float thr2, bool process = false) {
	cv::Size imgSize = iImg.size();

	// gaussian filtered image
	cv::Mat gImg = cv::Mat(imgSize, CV_32FC1);
	// sobel filtered images
	cv::Mat sxImg = cv::Mat(imgSize, CV_32FC1), syImg = cv::Mat(imgSize, CV_32FC1);
	// gradient value image and gradient direction image
	cv::Mat gradvImg = cv::Mat(imgSize, CV_32FC1), graddImg = cv::Mat(imgSize, CV_32FC1);
	// non maximum suppression image
	cv::Mat nmsImg = cv::Mat(imgSize, CV_32FC1);
	
	// Gaussian Filter
	cv::GaussianBlur(iImg, gImg, cv::Size(3, 3), 0.8);

	// Sobel Filter
	cv::Sobel(gImg, sxImg, CV_32F, 1, 0, 3);
	cv::Sobel(gImg, syImg, CV_32F, 0, 1, 3);

	// Calculate value and direction of gradient
	for (int y = 0; y < imgSize.height; y++) {
		for (int x = 0; x < imgSize.width; x++) {
			float gx = sxImg.at<float>(y, x), gy = syImg.at<float>(y, x);
			gradvImg.at<float>(y, x) = std::sqrt(std::pow(gx, 2.0) + std::pow(gy, 2.0));
			graddImg.at<float>(y, x) = std::atan2(gy, gx);
		}
	}

	// Non Maximum Suppression
	for (int y = 1; y < imgSize.height - 1; y++) {
		for (int x = 1; x < imgSize.width - 1; x++) {
			float dir = graddImg.at<float>(y, x);
			float val = gradvImg.at<float>(y, x);

			if (dir < 0) dir += CV_PI;

			if (dir <= CV_PI / 8.0 || CV_PI * 7.0 / 8.0 <= dir) {
				if (val < gradvImg.at<float>(y, x + 1) || val < gradvImg.at<float>(y, x - 1)) {
					nmsImg.at<float>(y, x) = 0;
				}
				else {
					nmsImg.at<float>(y, x) = val;
				}
			}
			else if (CV_PI / 8.0 <= dir && dir <= CV_PI * 3.0 / 8.0) {
				if (val < gradvImg.at<float>(y + 1, x + 1) || val < gradvImg.at<float>(y - 1, x - 1)) {
					nmsImg.at<float>(y, x) = 0;
				}
				else {
					nmsImg.at<float>(y, x) = val;
				}
			}
			else if (CV_PI * 3.0 / 8.0 <= dir && dir <= CV_PI * 5.0 / 8.0) {
				if (val < gradvImg.at<float>(y + 1, x) || val < gradvImg.at<float>(y - 1, x)) {
					nmsImg.at<float>(y, x) = 0;
				}
				else {
					nmsImg.at<float>(y, x) = val;
				}
			}
			else if (CV_PI * 5.0 / 8.0 <= dir && dir <= CV_PI * 7.0 / 8.0) {
				if (val < gradvImg.at<float>(y - 1, x + 1) || val < gradvImg.at<float>(y + 1, x - 1)) {
					nmsImg.at<float>(y, x) = 0;
				}
				else {
					nmsImg.at<float>(y, x) = val;
				}
			}
		}
	}

	// Hysteresis Threshold
	for (int y = 1; y < imgSize.height - 1; y++) {
		for (int x = 1; x < imgSize.width - 1; x++) {
			float val = nmsImg.at<float>(y, x);
			if (val < thr2) {
				oImg.at<uchar>(y, x) = 0;
			}
			else if (val < thr1) {
				for (int h = -1; h <= 1; h++) {
					for (int w = -1; w <= 1; w++) {
						if (nmsImg.at<float>(y, x) > thr1) {
							oImg.at<uchar>(y, x) = 255;
							break;
						}
					}
				}
			}
			else {
				oImg.at<uchar>(y, x) = 255;
			}
		}
	}


	if (process) {
		gImg.convertTo(gImg, CV_8UC1, 255.0);
		sxImg.convertTo(sxImg, CV_8UC1, 255.0);
		syImg.convertTo(syImg, CV_8UC1, 255.0);
		gradvImg.convertTo(gradvImg, CV_8UC1, 255.0);
		nmsImg.convertTo(nmsImg, CV_8UC1, 255.0);

		cv::imwrite("result/Gaussian-Filtered-Image.jpg", gImg);
		cv::imwrite("result/Sobel-Filtered-Image(x-axis).jpg", sxImg);
		cv::imwrite("result/Sobel-Filtered-Image(y-axis).jpg", syImg);
		cv::imwrite("result/Gradient-Image.jpg", gradvImg);
		cv::imwrite("result/Non-Maximum-Suppression-Image.jpg", nmsImg);
	}
}

int main(int argc, char* argv[]) {
	cv::Mat colorImg = cv::imread("lenna.png");
	cv::Mat grayImg_8U, grayImg_32F;
	cv::Mat cannyImg = cv::Mat(colorImg.size(), CV_8UC1);

	cv::cvtColor(colorImg, grayImg_8U, CV_RGB2GRAY);
	grayImg_8U.convertTo(grayImg_32F, CV_32F, 1.0 / 255.0);

	myCannyFilter(grayImg_32F, cannyImg, 0.2, 0.3, true);

	cv::imshow("Canny Filter", cannyImg);
	cv::imwrite("result/Canny-Filtered-Image.jpg", cannyImg);
	cv::waitKey(0);

	return 0;
}
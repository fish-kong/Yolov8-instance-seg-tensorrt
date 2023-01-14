#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "NvInferPlugin.h"
#include "logging.h"
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <string>
using namespace nvinfer1;
using namespace cv;

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 640;
static const int INPUT_W = 640;
static const int _segWidth = 160;
static const int _segHeight = 160;
static const int _segChannels = 32;
static const int CLASSES = 80;
static const int Num_box = 8400;
static const int OUTPUT_SIZE = Num_box * (CLASSES+4 + _segChannels);//output0
static const int OUTPUT_SIZE1 = _segChannels * _segWidth * _segHeight ;//output1


static const float CONF_THRESHOLD = 0.1;
static const float NMS_THRESHOLD = 0.5;
static const float MASK_THRESHOLD = 0.5;
const char* INPUT_BLOB_NAME = "images";
const char* OUTPUT_BLOB_NAME = "output0";//detect
const char* OUTPUT_BLOB_NAME1 = "output1";//mask


struct OutputSeg {
	int id;             //������id
	float confidence;   //������Ŷ�
	cv::Rect box;       //���ο�
	cv::Mat boxMask;       //���ο���mask����ʡ�ڴ�ռ�ͼӿ��ٶ�
};

void DrawPred(Mat& img,std:: vector<OutputSeg> result) {
	//���������ɫ
	std::vector<Scalar> color;
	srand(time(0));
	for (int i = 0; i < CLASSES; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(Scalar(b, g, r));
	}
	Mat mask = img.clone();
	for (int i = 0; i < result.size(); i++) {
		int left, top;
		left = result[i].box.x;
		top = result[i].box.y;
		int color_num = i;
		rectangle(img, result[i].box, color[result[i].id], 2, 8);
		
		mask(result[i].box).setTo(color[result[i].id], result[i].boxMask);
		char label[100];
		sprintf(label, "%d:%.2f", result[i].id, result[i].confidence);

		//std::string label = std::to_string(result[i].id) + ":" + std::to_string(result[i].confidence);
		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height);
		putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
	}
	
	addWeighted(img, 0.5, mask, 0.8, 1, img); //��mask����ԭͼ����

	
}



static Logger gLogger;
void doInference(IExecutionContext& context, float* input, float* output, float* output1, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 3);
    void* buffers[3];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
	const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
	const int outputIndex1 = engine.getBindingIndex(OUTPUT_BLOB_NAME1);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));//
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * OUTPUT_SIZE1 * sizeof(float)));
	// cudaMalloc�����ڴ� cudaFree�ͷ��ڴ� cudaMemcpy�� cudaMemcpyAsync ���������豸֮�䴫������
	// cudaMemcpy cudaMemcpyAsync ��ʽ���������� ��ʽ�ط��������� 
    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(output1, buffers[outputIndex1], batchSize * OUTPUT_SIZE1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
	CHECK(cudaFree(buffers[outputIndex1]));
}



int main(int argc, char** argv)
{
	if (argc < 2) {
		argv[1] = "../models/yolov8n-seg.engine";
		argv[2] = "../images/bus.jpg";
	}
	// create a model using the API directly and serialize it to a stream
	char* trtModelStream{ nullptr }; //char* trtModelStream==nullptr;  ���ٿ�ָ��� Ҫ��new���ʹ�ã�����89�� trtModelStream = new char[size]
	size_t size{ 0 };//��int�̶��ĸ��ֽڲ�ͬ������ͬ,size_t��ȡֵrange��Ŀ��ƽ̨�������ܵ�����ߴ�,һЩƽ̨��size_t�ķ�ΧС��int��������Χ,�ֻ��ߴ���unsigned int. ʹ��Int���п����˷ѣ����п��ܷ�Χ������

	std::ifstream file(argv[1], std::ios::binary);
	if (file.good()) {
		std::cout << "load engine success" << std::endl;
		file.seekg(0, file.end);//ָ���ļ�������ַ
		size = file.tellg();//���ļ����ȸ��߸�size
		//std::cout << "\nfile:" << argv[1] << " size is";
		//std::cout << size << "";

		file.seekg(0, file.beg);//ָ���ļ��Ŀ�ʼ��ַ
		trtModelStream = new char[size];//����һ��char �������ļ��ĳ���
		assert(trtModelStream);//
		file.read(trtModelStream, size);//���ļ����ݴ���trtModelStream
		file.close();//�ر�
	}
	else {
		std::cout << "load engine failed" << std::endl;
		return 1;
	}

	
	Mat src = imread(argv[2], 1);
	if (src.empty()) { std::cout << "image load faild" << std::endl; return 1; }
	int img_width = src.cols;
	int img_height = src.rows;
	std::cout << "��ߣ�" << img_width << " " << img_height << std::endl;
	// Subtract mean from image
	static float data[3 * INPUT_H * INPUT_W];
	Mat pr_img0, pr_img;
	std::vector<int> padsize;
	pr_img = preprocess_img(src, INPUT_H, INPUT_W, padsize);       // Resize
	int newh = padsize[0], neww = padsize[1], padh = padsize[2], padw = padsize[3];
	float ratio_h = (float)src.rows / newh;
	float ratio_w = (float)src.cols / neww;
	int i = 0;// [1,3,INPUT_H,INPUT_W]
	//std::cout << "pr_img.step" << pr_img.step << std::endl;
	for (int row = 0; row < INPUT_H; ++row) {
		uchar* uc_pixel = pr_img.data + row * pr_img.step;//pr_img.step=widthx3 ����ÿһ����width��3ͨ����ֵ
		for (int col = 0; col < INPUT_W; ++col)
		{

			data[i] = (float)uc_pixel[2] / 255.0;
			data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
			data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.;
			uc_pixel += 3;
			++i;
		}
	}

	IRuntime* runtime = createInferRuntime(gLogger);
	assert(runtime != nullptr);
	bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
	ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
	assert(engine != nullptr);
	IExecutionContext* context = engine->createExecutionContext();
	assert(context != nullptr);
	delete[] trtModelStream;

	// Run inference
	static float prob[OUTPUT_SIZE];
	static float prob1[OUTPUT_SIZE1];

	//for (int i = 0; i < 10; i++) {//����10�ε������ٶ�
	//       auto start = std::chrono::system_clock::now();
	//       doInference(*context, data, prob, prob1, 1);
	//       auto end = std::chrono::system_clock::now();
	//       std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
	//   }
	auto start = std::chrono::system_clock::now();
	doInference(*context, data, prob, prob1, 1);
	auto end = std::chrono::system_clock::now();
	std::cout << "����ʱ�䣺" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

	std::vector<int> classIds;//���id����
	std::vector<float> confidences;//���ÿ��id��Ӧ���Ŷ�����
	std::vector<cv::Rect> boxes;//ÿ��id���ο�
	std::vector<cv::Mat> picked_proposals;  //��������mask

	

	// ����box
	int net_length = CLASSES + 4 + _segChannels;
	cv::Mat out1 = cv::Mat(net_length, Num_box, CV_32F, prob);

	start = std::chrono::system_clock::now();
	for (int i = 0; i < Num_box; i++) {
		//�����1*net_length*Num_box;����ÿ��box��������ÿ��Num_boxȡһ��ֵ����net_length��ֵ
		cv::Mat scores = out1(Rect(i, 4, 1, CLASSES)).clone();
		Point classIdPoint;
		double max_class_socre;
		minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
		max_class_socre = (float)max_class_socre;
		if (max_class_socre >= CONF_THRESHOLD) {
			cv::Mat temp_proto = out1(Rect(i, 4 + CLASSES, 1, _segChannels)).clone();
			picked_proposals.push_back(temp_proto.t());
			float x = (out1.at<float>(0, i) - padw) * ratio_w;  //cx
			float y = (out1.at<float>(1, i) - padh) * ratio_h;  //cy
			float w = out1.at<float>(2, i) * ratio_w;  //w
			float h = out1.at<float>(3, i) * ratio_h;  //h
			int left = MAX((x - 0.5 * w), 0);
			int top = MAX((y - 0.5 * h), 0);
			int width = (int)w;
			int height = (int)h;
			if (width <= 0 || height <= 0) { continue; }

			classIds.push_back(classIdPoint.y);
			confidences.push_back(max_class_socre);
			boxes.push_back(Rect(left, top, width, height));
		}

	}
	//ִ�з�����������������нϵ����Ŷȵ������ص���NMS��
	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, nms_result);
	std::vector<cv::Mat> temp_mask_proposals;
	std::vector<OutputSeg> output;
	for (int i = 0; i < nms_result.size(); ++i) {
		int idx = nms_result[i];
		OutputSeg result;
		result.id = classIds[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx];
		output.push_back(result);
		temp_mask_proposals.push_back(picked_proposals[idx]);
	}

	// ����mask
	Mat maskProposals;
	for (int i = 0; i < temp_mask_proposals.size(); ++i)
		maskProposals.push_back(temp_mask_proposals[i]);

	Mat protos = Mat(_segChannels, _segWidth * _segHeight, CV_32F, prob1);
	Mat matmulRes = (maskProposals * protos).t();//n*32 32*25600 A*B������ѧ�����о�����˵ķ�ʽʵ�ֵģ�Ҫ��A����������B������ʱ
	Mat masks = matmulRes.reshape(output.size(), { _segWidth,_segHeight });//n*160*160

	std::vector<Mat> maskChannels;
	cv::split(masks, maskChannels);
	Rect roi(int((float)padw / INPUT_W * _segWidth), int((float)padh / INPUT_H * _segHeight), int(_segWidth - padw / 2), int(_segHeight - padh / 2));
	for (int i = 0; i < output.size(); ++i) {
		Mat dest, mask;
		cv::exp(-maskChannels[i], dest);//sigmoid
		dest = 1.0 / (1.0 + dest);//160*160
		dest = dest(roi);
		resize(dest, mask, cv::Size(src.cols, src.rows), INTER_NEAREST);
		//crop----��ȡbox�е�mask��Ϊ��box��Ӧ��mask
		Rect temp_rect = output[i].box;
		mask = mask(temp_rect) > MASK_THRESHOLD;
		output[i].boxMask = mask;
	}
	end = std::chrono::system_clock::now();
	std::cout << "����ʱ�䣺" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

	DrawPred(src, output);
	cv::imshow("output.jpg", src);
	char c = cv::waitKey(0);
	
	// Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

	system("pause");
    return 0;
}

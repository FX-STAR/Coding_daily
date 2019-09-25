/*
 * @Description: In User Settings Edit
 * @Author: your name
 * @Date: 2019-09-25 12:47:18
 * @LastEditTime: 2019-09-25 15:40:17
 * @LastEditors: Please set LastEditors
 */
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;
using std::string;
#define CPU_ONLY


class Caffe_Basic {
public:
  Caffe_Basic(const string& model_file, const string& trained_file);
  void Forward(const cv::Mat& image, std::vector<float>& output);

private:
  void Feed(const cv::Mat& image);
  shared_ptr<Net<float>> net_;
  cv::Size input_geometry_;
};


Caffe_Basic::Caffe_Basic(const string& prototxt, const string& caffemodel) {
  #ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
  #else
    Caffe::SetDevice(0);
    Caffe::set_mode(Caffe::GPU);
  #endif
  /* Load the network. */
  net_.reset(new Net<float>(prototxt, TEST));
  net_->CopyTrainedLayersFrom(caffemodel);
  Blob<float>* input_layer = net_->input_blobs()[0];
  CHECK(input_layer->channels() == 3) << "Network Input only support channel as 3.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}


void Caffe_Basic::Feed(const cv::Mat& image){
  /* Wrap InputLayer */
  Blob<float>* input_layer = net_->input_blobs()[0];
  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  std::vector<cv::Mat> input_channels;
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels.push_back(channel);
    input_data += width * height;
  }

  /* Convert the input image to the input image format of the network. */
  cv::Mat resized;
  if (image.size() != input_geometry_)
    cv::resize(image, resized, input_geometry_);
  else
    resized = image;

  /* Normalization: [0,255] -> [-1, 1] */
  cv::Mat normed;
  resized.convertTo(normed, CV_32FC3, 1.0 / 128.0, -127.5 / 128.0);
  cv::cvtColor(normed, normed, cv::COLOR_BGR2RGB);
  cv::split(normed, input_channels);
  
}

void Caffe_Basic::Forward(const cv::Mat& image, std::vector<float>& output){
  Feed(image);
  net_->Forward();
  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  output = std::vector<float>(begin, end);
}

int main(){
  Caffe_Basic basic("../../caffe.prototxt", "../../caffe.caffemodel");
  cv::Mat image = cv::imread("../../res/test_race(1).jpg");
  std::vector<float> out;
  basic.Forward(image, out);
  for(int i=0;i<out.size();i++){
    std::cout<<out.at(i)<<std::endl;
  }
  return 0;
}

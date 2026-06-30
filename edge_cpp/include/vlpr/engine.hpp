#pragma once

#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "vlpr/types.hpp"

namespace vlpr {

class PlateDetector {
 public:
  PlateDetector(Ort::Env& environment, const RuntimeConfig& config);
  std::vector<Box> detect(const cv::Mat& frame);

 private:
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_;
  std::string output_name_;
  int input_size_{};
  float confidence_{};
  float nms_threshold_{};
};

class PlateRecognizer {
 public:
  PlateRecognizer(Ort::Env& environment, const RuntimeConfig& config);
  OcrResult recognize(const cv::Mat& plate_crop);

 private:
  struct LineResult {
    std::string text;
    float confidence{};
  };

  LineResult recognize_line(const cv::Mat& line);
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_;
  std::string output_name_;
  std::vector<std::string> charset_;
};

class Pipeline {
 public:
  explicit Pipeline(const RuntimeConfig& config);
  FrameResult process(const cv::Mat& frame, bool stream_mode);
  void reset();

 private:
  struct Track {
    Box box;
    OcrResult ocr;
    std::chrono::steady_clock::time_point last_seen;
    std::chrono::steady_clock::time_point last_ocr;
    bool has_ocr{false};
  };

  static cv::Mat crop_plate(const cv::Mat& frame, const Box& box);
  Ort::Env environment_;
  PlateDetector detector_;
  PlateRecognizer recognizer_;
  std::vector<Track> tracks_;
};

}  // namespace vlpr

#pragma once

#include <chrono>
#include <cstdint>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace vlpr {

struct Box {
  int x1{};
  int y1{};
  int x2{};
  int y2{};
  float score{};
};

struct OcrResult {
  std::string text;
  std::string raw_text;
  float confidence{};
  bool valid{};
};

struct Recognition {
  Box box;
  OcrResult ocr;
};

struct FrameResult {
  std::vector<Recognition> recognitions;
  double detector_ms{};
  double total_ms{};
};

struct RuntimeConfig {
  std::string source{"0"};
  std::string detector_model;
  std::string recognizer_model;
  std::string charset_file;
  std::string event_log;
  float detector_confidence{0.30F};
  float nms_threshold{0.45F};
  float min_ocr_confidence{0.65F};
  int threads{2};
  int frame_stride{1};
  int heartbeat_seconds{30};
  int dedupe_seconds{8};
  bool dry_run{false};
  bool self_test{false};
  bool benchmark{false};
  int benchmark_iterations{20};
};

}  // namespace vlpr

#include "vlpr/engine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <utility>

#include "vlpr/text.hpp"

namespace vlpr {
namespace {

std::unique_ptr<Ort::Session> create_session(Ort::Env& environment, const std::string& model_path,
                                             int threads) {
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  options.SetIntraOpNumThreads(std::max(1, threads));
  options.SetInterOpNumThreads(1);
#ifdef _WIN32
  const std::wstring wide_path = std::filesystem::path(model_path).wstring();
  return std::make_unique<Ort::Session>(environment, wide_path.c_str(), options);
#else
  return std::make_unique<Ort::Session>(environment, model_path.c_str(), options);
#endif
}

std::string get_input_name(Ort::Session& session) {
  Ort::AllocatorWithDefaultOptions allocator;
  auto name = session.GetInputNameAllocated(0, allocator);
  return std::string(name.get());
}

std::string get_output_name(Ort::Session& session) {
  Ort::AllocatorWithDefaultOptions allocator;
  auto name = session.GetOutputNameAllocated(0, allocator);
  return std::string(name.get());
}

float intersection_over_union(const Box& first, const Box& second) {
  const int left = std::max(first.x1, second.x1);
  const int top = std::max(first.y1, second.y1);
  const int right = std::min(first.x2, second.x2);
  const int bottom = std::min(first.y2, second.y2);
  const float intersection =
      static_cast<float>(std::max(0, right - left) * std::max(0, bottom - top));
  const float first_area =
      static_cast<float>(std::max(1, first.x2 - first.x1) * std::max(1, first.y2 - first.y1));
  const float second_area =
      static_cast<float>(std::max(1, second.x2 - second.x1) * std::max(1, second.y2 - second.y1));
  return intersection / std::max(1.0F, first_area + second_area - intersection);
}

cv::Mat enhance_plate(const cv::Mat& crop) {
  cv::Mat resized = crop;
  if (crop.cols < 320) {
    const double scale = 320.0 / static_cast<double>(std::max(1, crop.cols));
    cv::resize(crop, resized, cv::Size(320, std::max(32, cvRound(crop.rows * scale))), 0.0, 0.0,
               cv::INTER_CUBIC);
  }
  cv::Mat gray;
  cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
  auto clahe = cv::createCLAHE(2.5, cv::Size(4, 4));
  clahe->apply(gray, gray);
  cv::Mat blurred;
  cv::GaussianBlur(gray, blurred, cv::Size(), 2.0);
  cv::Mat sharpened;
  cv::addWeighted(gray, 1.5, blurred, -0.5, 0.0, sharpened);
  cv::Mat output;
  cv::cvtColor(sharpened, output, cv::COLOR_GRAY2BGR);
  const int border = std::max(4, output.cols / 40);
  cv::copyMakeBorder(output, output, border, border, border, border, cv::BORDER_CONSTANT,
                     cv::Scalar(255, 255, 255));
  return output;
}

int find_line_split(const cv::Mat& crop) {
  if (crop.rows < 4) {
    return std::max(1, crop.rows / 2);
  }
  cv::Mat gray;
  cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0.0);
  cv::Mat binary;
  cv::threshold(gray, binary, 0.0, 255.0, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

  const int lower = std::max(1, static_cast<int>(crop.rows * 0.38));
  const int upper = std::min(crop.rows - 1, static_cast<int>(crop.rows * 0.62));
  int best_row = crop.rows / 2;
  double best_ink = std::numeric_limits<double>::max();
  for (int row = lower; row <= upper; ++row) {
    const int start = std::max(0, row - 2);
    const int end = std::min(crop.rows, row + 3);
    const double ink = cv::sum(binary.rowRange(start, end))[0];
    if (ink < best_ink) {
      best_ink = ink;
      best_row = row;
    }
  }
  return std::clamp(best_row, 1, crop.rows - 1);
}

float candidate_quality(const OcrResult& result) {
  const float validity = result.valid ? 10.0F : 0.0F;
  const float length =
      1.0F - std::min(1.0F, std::abs(static_cast<float>(result.text.size()) - 8.0F) / 8.0F);
  return validity + result.confidence + 0.25F * length;
}

}  // namespace

PlateDetector::PlateDetector(Ort::Env& environment, const RuntimeConfig& config)
    : session_(create_session(environment, config.detector_model, config.threads)),
      input_name_(get_input_name(*session_)),
      output_name_(get_output_name(*session_)),
      confidence_(config.detector_confidence),
      nms_threshold_(config.nms_threshold) {
  const auto shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  if (shape.size() != 4 || shape[2] <= 0 || shape[3] <= 0 || shape[2] != shape[3]) {
    throw std::runtime_error("Detector must have a fixed square NCHW input");
  }
  input_size_ = static_cast<int>(shape[2]);
}

std::vector<Box> PlateDetector::detect(const cv::Mat& frame) {
  if (frame.empty() || frame.type() != CV_8UC3) {
    throw std::invalid_argument("Input frame must be a non-empty BGR uint8 image");
  }

  const float scale = std::min(static_cast<float>(input_size_) / static_cast<float>(frame.cols),
                               static_cast<float>(input_size_) / static_cast<float>(frame.rows));
  const int resized_width = std::max(1, cvRound(frame.cols * scale));
  const int resized_height = std::max(1, cvRound(frame.rows * scale));
  const int pad_x = (input_size_ - resized_width) / 2;
  const int pad_y = (input_size_ - resized_height) / 2;

  cv::Mat resized;
  cv::resize(frame, resized, cv::Size(resized_width, resized_height));
  cv::Mat canvas(input_size_, input_size_, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(canvas(cv::Rect(pad_x, pad_y, resized_width, resized_height)));
  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
  canvas.convertTo(canvas, CV_32FC3, 1.0 / 255.0);

  std::vector<cv::Mat> channels;
  cv::split(canvas, channels);
  std::vector<float> tensor_data(static_cast<std::size_t>(3 * input_size_ * input_size_));
  const std::size_t channel_size = static_cast<std::size_t>(input_size_ * input_size_);
  for (std::size_t channel = 0; channel < 3; ++channel) {
    std::memcpy(tensor_data.data() + channel * channel_size, channels[channel].ptr<float>(),
                channel_size * sizeof(float));
  }

  const std::array<int64_t, 4> input_shape{1, 3, input_size_, input_size_};
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<float>(memory, tensor_data.data(), tensor_data.size(),
                                                input_shape.data(), input_shape.size());
  const char* input_names[] = {input_name_.c_str()};
  const char* output_names[] = {output_name_.c_str()};
  auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names, 1);

  const auto output_shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
  if (output_shape.size() != 3 || output_shape[0] != 1) {
    throw std::runtime_error("Detector output is not a YOLOv8 [1,F,N] tensor");
  }
  const int64_t first = output_shape[1];
  const int64_t second = output_shape[2];
  const bool feature_first = first < second;
  const int64_t features = feature_first ? first : second;
  const int64_t anchors = feature_first ? second : first;
  constexpr int plate_class = 2;
  if (features <= 4 + plate_class) {
    throw std::runtime_error("Detector output has no plate class at index 2");
  }
  const float* output = outputs.front().GetTensorData<float>();
  const auto at = [&](int64_t anchor, int64_t feature) {
    return feature_first ? output[feature * anchors + anchor] : output[anchor * features + feature];
  };

  std::vector<Box> candidates;
  for (int64_t anchor = 0; anchor < anchors; ++anchor) {
    int best_class = 0;
    float best_score = at(anchor, 4);
    for (int64_t feature = 5; feature < features; ++feature) {
      if (at(anchor, feature) > best_score) {
        best_score = at(anchor, feature);
        best_class = static_cast<int>(feature - 4);
      }
    }
    if (best_class != plate_class || best_score < confidence_) {
      continue;
    }

    const float center_x = at(anchor, 0);
    const float center_y = at(anchor, 1);
    const float width = at(anchor, 2);
    const float height = at(anchor, 3);
    Box box;
    box.x1 =
        std::clamp(static_cast<int>((center_x - width / 2.0F - pad_x) / scale), 0, frame.cols - 1);
    box.y1 =
        std::clamp(static_cast<int>((center_y - height / 2.0F - pad_y) / scale), 0, frame.rows - 1);
    box.x2 =
        std::clamp(static_cast<int>((center_x + width / 2.0F - pad_x) / scale), 0, frame.cols - 1);
    box.y2 =
        std::clamp(static_cast<int>((center_y + height / 2.0F - pad_y) / scale), 0, frame.rows - 1);
    box.score = best_score;
    if (box.x2 > box.x1 && box.y2 > box.y1) {
      candidates.push_back(box);
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const Box& left, const Box& right) { return left.score > right.score; });
  std::vector<Box> selected;
  for (const Box& candidate : candidates) {
    const bool suppressed = std::any_of(selected.begin(), selected.end(), [&](const Box& box) {
      return intersection_over_union(candidate, box) > nms_threshold_;
    });
    if (!suppressed) {
      selected.push_back(candidate);
    }
  }
  return selected;
}

PlateRecognizer::PlateRecognizer(Ort::Env& environment, const RuntimeConfig& config)
    : session_(create_session(environment, config.recognizer_model, config.threads)),
      input_name_(get_input_name(*session_)),
      output_name_(get_output_name(*session_)) {
  std::ifstream input(config.charset_file, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Cannot open OCR charset: " + config.charset_file);
  }
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    charset_.push_back(line);
  }
  if (charset_.empty()) {
    throw std::runtime_error("OCR charset is empty");
  }
}

PlateRecognizer::LineResult PlateRecognizer::recognize_line(const cv::Mat& line) {
  if (line.empty()) {
    return {};
  }
  constexpr int target_height = 48;
  constexpr float default_ratio = 320.0F / 48.0F;
  const float ratio = static_cast<float>(line.cols) / static_cast<float>(line.rows);
  const int tensor_width =
      std::max(1, static_cast<int>(std::ceil(target_height * std::max(default_ratio, ratio))));
  const int resized_width =
      std::min(tensor_width, std::max(1, static_cast<int>(std::ceil(target_height * ratio))));

  cv::Mat resized;
  cv::resize(line, resized, cv::Size(resized_width, target_height));
  resized.convertTo(resized, CV_32FC3, 1.0 / 127.5, -1.0);
  std::vector<cv::Mat> channels;
  cv::split(resized, channels);

  const std::size_t plane = static_cast<std::size_t>(target_height * tensor_width);
  std::vector<float> tensor_data(3 * plane, 0.0F);
  for (std::size_t channel = 0; channel < 3; ++channel) {
    for (int row = 0; row < target_height; ++row) {
      std::memcpy(
          tensor_data.data() + channel * plane + static_cast<std::size_t>(row * tensor_width),
          channels[channel].ptr<float>(row),
          static_cast<std::size_t>(resized_width) * sizeof(float));
    }
  }

  const std::array<int64_t, 4> input_shape{1, 3, target_height, tensor_width};
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<float>(memory, tensor_data.data(), tensor_data.size(),
                                                input_shape.data(), input_shape.size());
  const char* input_names[] = {input_name_.c_str()};
  const char* output_names[] = {output_name_.c_str()};
  auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names, 1);
  const auto shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
  if (shape.size() != 3 || shape[0] != 1 || shape[2] != static_cast<int64_t>(charset_.size() + 2)) {
    throw std::runtime_error("OCR model and charset are incompatible");
  }

  const float* scores = outputs.front().GetTensorData<float>();
  const int64_t steps = shape[1];
  const int64_t classes = shape[2];
  int64_t previous = -1;
  float confidence_sum = 0.0F;
  int confidence_count = 0;
  std::string text;
  for (int64_t step = 0; step < steps; ++step) {
    const float* row = scores + step * classes;
    const auto best = std::max_element(row, row + classes);
    const int64_t index = std::distance(row, best);
    if (index != 0 && index != previous) {
      if (index == static_cast<int64_t>(charset_.size() + 1)) {
        text.push_back(' ');
      } else if (index > 0 && index <= static_cast<int64_t>(charset_.size())) {
        text += charset_[static_cast<std::size_t>(index - 1)];
      }
      confidence_sum += *best;
      ++confidence_count;
    }
    previous = index;
  }
  return {text, confidence_count == 0 ? 0.0F : confidence_sum / confidence_count};
}

OcrResult PlateRecognizer::recognize(const cv::Mat& plate_crop) {
  if (plate_crop.empty()) {
    return {};
  }

  const cv::Mat whole_image = enhance_plate(plate_crop);
  const LineResult whole = recognize_line(whole_image);
  OcrResult best{clean_plate_text(whole.text), whole.text, whole.confidence, false};
  best.valid = is_valid_vietnam_plate(best.text);

  const float aspect =
      static_cast<float>(plate_crop.cols) / static_cast<float>(std::max(1, plate_crop.rows));
  if (plate_crop.rows >= 8 && (aspect < 1.75F || !best.valid)) {
    const int split = find_line_split(plate_crop);
    const LineResult top = recognize_line(enhance_plate(plate_crop.rowRange(0, split)));
    const LineResult bottom =
        recognize_line(enhance_plate(plate_crop.rowRange(split, plate_crop.rows)));
    const std::string split_raw = top.text + bottom.text;
    const int character_count = static_cast<int>(top.text.size() + bottom.text.size());
    const float split_confidence =
        character_count == 0 ? 0.0F
                             : (top.confidence * static_cast<float>(top.text.size()) +
                                bottom.confidence * static_cast<float>(bottom.text.size())) /
                                   static_cast<float>(character_count);
    OcrResult split_result{clean_plate_text(split_raw), split_raw, split_confidence, false};
    split_result.valid = is_valid_vietnam_plate(split_result.text);
    if (candidate_quality(split_result) > candidate_quality(best)) {
      best = std::move(split_result);
    }
  }
  return best;
}

Pipeline::Pipeline(const RuntimeConfig& config)
    : environment_(ORT_LOGGING_LEVEL_WARNING, "vietnam-lpr-edge"),
      detector_(environment_, config),
      recognizer_(environment_, config) {}

cv::Mat Pipeline::crop_plate(const cv::Mat& frame, const Box& box) {
  const int width = box.x2 - box.x1;
  const int height = box.y2 - box.y1;
  const int pad_x = std::max(8, width / 10);
  const int pad_y = std::max(6, height / 5);
  const int left = std::max(0, box.x1 - pad_x);
  const int top = std::max(0, box.y1 - pad_y);
  const int right = std::min(frame.cols, box.x2 + pad_x);
  const int bottom = std::min(frame.rows, box.y2 + pad_y);
  return frame(cv::Rect(left, top, right - left, bottom - top));
}

FrameResult Pipeline::process(const cv::Mat& frame, bool stream_mode) {
  const auto started = std::chrono::steady_clock::now();
  const auto detections_started = started;
  const std::vector<Box> detections = detector_.detect(frame);
  const auto detections_finished = std::chrono::steady_clock::now();
  const auto now = detections_finished;

  FrameResult output;
  output.detector_ms =
      std::chrono::duration<double, std::milli>(detections_finished - detections_started).count();
  std::vector<Track> active;
  std::vector<bool> used(tracks_.size(), false);

  for (const Box& detection : detections) {
    int best_index = -1;
    float best_overlap = 0.0F;
    if (stream_mode) {
      for (std::size_t index = 0; index < tracks_.size(); ++index) {
        if (used[index]) {
          continue;
        }
        const float overlap = intersection_over_union(detection, tracks_[index].box);
        if (overlap > best_overlap) {
          best_overlap = overlap;
          best_index = static_cast<int>(index);
        }
      }
    }

    Track track;
    if (best_index >= 0 && best_overlap >= 0.25F) {
      used[static_cast<std::size_t>(best_index)] = true;
      track = tracks_[static_cast<std::size_t>(best_index)];
    }
    track.box = detection;
    track.last_seen = now;
    const bool retry_ocr =
        !track.has_ocr ||
        (!track.ocr.valid && now - track.last_ocr >= std::chrono::milliseconds(900));
    if ((!stream_mode || retry_ocr) && detection.x2 - detection.x1 >= 40) {
      OcrResult candidate = recognizer_.recognize(crop_plate(frame, detection));
      if (!track.has_ocr || candidate_quality(candidate) >= candidate_quality(track.ocr)) {
        track.ocr = std::move(candidate);
      }
      track.last_ocr = now;
      track.has_ocr = true;
    }
    output.recognitions.push_back({track.box, track.ocr});
    active.push_back(std::move(track));
  }

  if (stream_mode) {
    for (std::size_t index = 0; index < tracks_.size(); ++index) {
      if (!used[index] && now - tracks_[index].last_seen < std::chrono::milliseconds(1500)) {
        active.push_back(std::move(tracks_[index]));
      }
    }
    tracks_ = std::move(active);
  }
  output.total_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  return output;
}

void Pipeline::reset() { tracks_.clear(); }

}  // namespace vlpr

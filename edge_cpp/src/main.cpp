#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "vlpr/engine.hpp"
#include "vlpr/text.hpp"

#ifndef VLPR_VERSION
#define VLPR_VERSION "dev"
#endif

namespace {

std::atomic<bool> stop_requested{false};

void request_stop(int) { stop_requested.store(true); }

bool is_integer(const std::string& value) {
  return !value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char character) {
    return std::isdigit(character) != 0;
  });
}

bool starts_with(const std::string& value, const std::string& prefix) {
  return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

std::string safe_source_name(const std::string& source) {
  const std::size_t scheme = source.find("://");
  const std::size_t credentials_end =
      source.find('@', scheme == std::string::npos ? 0 : scheme + 3);
  if (scheme == std::string::npos || credentials_end == std::string::npos) {
    return source;
  }
  return source.substr(0, scheme + 3) + "***@" + source.substr(credentials_end + 1);
}

std::filesystem::path executable_directory(const char* argv0) {
  std::error_code error;
  const auto absolute = std::filesystem::absolute(argv0, error);
  return error ? std::filesystem::current_path() : absolute.parent_path();
}

std::string find_resource(const std::filesystem::path& executable_dir,
                          const std::vector<std::filesystem::path>& relative_candidates) {
  const std::vector<std::filesystem::path> roots{
      std::filesystem::current_path(),
      executable_dir,
      executable_dir / "..",
      executable_dir / "../share/vietnam-lpr-edge",
  };
  for (const auto& root : roots) {
    for (const auto& relative : relative_candidates) {
      const auto candidate = (root / relative).lexically_normal();
      std::error_code error;
      if (std::filesystem::is_regular_file(candidate, error)) {
        return std::filesystem::absolute(candidate).string();
      }
    }
  }
  return {};
}

void print_help() {
  std::cout
      << "Vietnam LPR Edge " << VLPR_VERSION << "\n\n"
      << "Usage: vlpr_edge [options]\n\n"
      << "  --source VALUE        Camera index, video, image, RTSP or HTTP stream (default: 0)\n"
      << "  --detector PATH       YOLOv8 ONNX detector\n"
      << "  --recognizer PATH     PP-OCRv6 ONNX recognizer\n"
      << "  --charset PATH        UTF-8 OCR charset generated from model metadata\n"
      << "  --event-log PATH      Append accepted events as JSON Lines\n"
      << "  --confidence FLOAT    Detector threshold, 0..1 (default: 0.30)\n"
      << "  --ocr-confidence F    Event OCR threshold, 0..1 (default: 0.65)\n"
      << "  --threads N           ONNX Runtime CPU threads (default: 2)\n"
      << "  --frame-stride N      Process every Nth captured frame (default: 1)\n"
      << "  --dedupe-seconds N    Suppress duplicate plate events (default: 8)\n"
      << "  --heartbeat N         Health event period in seconds (default: 30)\n"
      << "  --dry-run             Load and validate both models, then exit\n"
      << "  --self-test IMAGE     Run one image and return non-zero if no plate is accepted\n"
      << "  --benchmark IMAGE     Benchmark a single image\n"
      << "  --iterations N        Benchmark iterations (default: 20)\n"
      << "  --version             Print version\n"
      << "  --help                Print this help\n";
}

template <typename Number>
Number parse_number(const std::string& value, const std::string& option);

template <>
int parse_number<int>(const std::string& value, const std::string& option) {
  std::size_t consumed = 0;
  const int parsed = std::stoi(value, &consumed);
  if (consumed != value.size()) {
    throw std::invalid_argument("Invalid integer for " + option + ": " + value);
  }
  return parsed;
}

template <>
float parse_number<float>(const std::string& value, const std::string& option) {
  std::size_t consumed = 0;
  const float parsed = std::stof(value, &consumed);
  if (consumed != value.size()) {
    throw std::invalid_argument("Invalid number for " + option + ": " + value);
  }
  return parsed;
}

vlpr::RuntimeConfig parse_arguments(int argc, char** argv) {
  vlpr::RuntimeConfig config;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    const auto next = [&]() -> std::string {
      if (++index >= argc) {
        throw std::invalid_argument("Missing value after " + option);
      }
      return argv[index];
    };
    if (option == "--source")
      config.source = next();
    else if (option == "--detector")
      config.detector_model = next();
    else if (option == "--recognizer")
      config.recognizer_model = next();
    else if (option == "--charset")
      config.charset_file = next();
    else if (option == "--event-log")
      config.event_log = next();
    else if (option == "--confidence")
      config.detector_confidence = parse_number<float>(next(), option);
    else if (option == "--ocr-confidence")
      config.min_ocr_confidence = parse_number<float>(next(), option);
    else if (option == "--threads")
      config.threads = parse_number<int>(next(), option);
    else if (option == "--frame-stride")
      config.frame_stride = parse_number<int>(next(), option);
    else if (option == "--dedupe-seconds")
      config.dedupe_seconds = parse_number<int>(next(), option);
    else if (option == "--heartbeat")
      config.heartbeat_seconds = parse_number<int>(next(), option);
    else if (option == "--iterations")
      config.benchmark_iterations = parse_number<int>(next(), option);
    else if (option == "--dry-run")
      config.dry_run = true;
    else if (option == "--self-test") {
      config.self_test = true;
      config.source = next();
    } else if (option == "--benchmark") {
      config.benchmark = true;
      config.source = next();
    } else if (option == "--help") {
      print_help();
      std::exit(0);
    } else if (option == "--version") {
      std::cout << VLPR_VERSION << '\n';
      std::exit(0);
    } else
      throw std::invalid_argument("Unknown option: " + option);
  }
  if (config.detector_confidence < 0.0F || config.detector_confidence > 1.0F ||
      config.min_ocr_confidence < 0.0F || config.min_ocr_confidence > 1.0F) {
    throw std::invalid_argument("Confidence thresholds must be between 0 and 1");
  }
  if (config.threads < 1 || config.frame_stride < 1 || config.heartbeat_seconds < 1 ||
      config.dedupe_seconds < 0 || config.benchmark_iterations < 1) {
    throw std::invalid_argument("Thread/count/time options are outside their valid range");
  }
  return config;
}

std::string recognition_json(const vlpr::Recognition& recognition, const vlpr::FrameResult& frame,
                             const std::string& source, const std::string& event_type) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(4) << "{\"ts\":\"" << vlpr::utc_timestamp()
         << "\",\"type\":\"" << event_type << "\",\"source\":\""
         << vlpr::json_escape(safe_source_name(source)) << "\",\"plate\":\""
         << vlpr::json_escape(recognition.ocr.text) << "\",\"raw_text\":\""
         << vlpr::json_escape(recognition.ocr.raw_text)
         << "\",\"valid\":" << (recognition.ocr.valid ? "true" : "false")
         << ",\"detection_confidence\":" << recognition.box.score
         << ",\"ocr_confidence\":" << recognition.ocr.confidence << ",\"bbox\":["
         << recognition.box.x1 << ',' << recognition.box.y1 << ',' << recognition.box.x2 << ','
         << recognition.box.y2 << ']' << ",\"detector_ms\":" << frame.detector_ms
         << ",\"total_ms\":" << frame.total_ms << '}';
  return output.str();
}

class EventWriter {
 public:
  explicit EventWriter(const std::string& path) {
    if (!path.empty()) {
      const std::filesystem::path file_path(path);
      if (file_path.has_parent_path()) {
        std::filesystem::create_directories(file_path.parent_path());
      }
      file_.open(path, std::ios::app | std::ios::binary);
      if (!file_) {
        throw std::runtime_error("Cannot open event log: " + path);
      }
    }
  }

  void emit(const std::string& json) {
    std::cout << json << std::endl;
    if (file_) {
      file_ << json << '\n';
      file_.flush();
    }
  }

 private:
  std::ofstream file_;
};

bool accepted(const vlpr::Recognition& recognition, const vlpr::RuntimeConfig& config) {
  return recognition.ocr.valid || (recognition.ocr.text.size() >= 7 &&
                                   recognition.ocr.confidence >= config.min_ocr_confidence);
}

int run_image(vlpr::Pipeline& pipeline, const vlpr::RuntimeConfig& config, EventWriter& writer) {
  const cv::Mat image = cv::imread(config.source, cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("Cannot read image: " + config.source);
  }
  const auto result = pipeline.process(image, false);
  bool any_accepted = false;
  for (const auto& recognition : result.recognitions) {
    writer.emit(recognition_json(recognition, result, config.source, "recognition"));
    any_accepted = any_accepted || accepted(recognition, config);
  }
  if (result.recognitions.empty()) {
    writer.emit("{\"ts\":\"" + vlpr::utc_timestamp() +
                "\",\"type\":\"no_detection\",\"source\":\"" +
                vlpr::json_escape(safe_source_name(config.source)) + "\"}");
  }
  return any_accepted ? 0 : 3;
}

int run_benchmark(vlpr::Pipeline& pipeline, const vlpr::RuntimeConfig& config) {
  const cv::Mat image = cv::imread(config.source, cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("Cannot read benchmark image: " + config.source);
  }
  pipeline.process(image, false);
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(config.benchmark_iterations));
  for (int iteration = 0; iteration < config.benchmark_iterations; ++iteration) {
    pipeline.reset();
    samples.push_back(pipeline.process(image, false).total_ms);
  }
  std::sort(samples.begin(), samples.end());
  const double mean =
      std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
  const auto percentile = [&](double fraction) {
    return samples[std::min(
        samples.size() - 1,
        static_cast<std::size_t>(fraction * static_cast<double>(samples.size() - 1)))];
  };
  std::cout << std::fixed << std::setprecision(3)
            << "{\"type\":\"benchmark\",\"iterations\":" << samples.size()
            << ",\"mean_ms\":" << mean << ",\"p50_ms\":" << percentile(0.50)
            << ",\"p95_ms\":" << percentile(0.95) << "}" << std::endl;
  return 0;
}

bool open_capture(cv::VideoCapture& capture, const std::string& source) {
  const bool opened = is_integer(source) ? capture.open(std::stoi(source), cv::CAP_ANY)
                                         : capture.open(source, cv::CAP_ANY);
  if (opened) {
    capture.set(cv::CAP_PROP_BUFFERSIZE, 1.0);
  }
  return opened;
}

int run_stream(vlpr::Pipeline& pipeline, const vlpr::RuntimeConfig& config, EventWriter& writer) {
  cv::VideoCapture capture;
  if (!open_capture(capture, config.source)) {
    throw std::runtime_error("Cannot open camera/video/stream: " + safe_source_name(config.source));
  }
  const bool reconnectable = is_integer(config.source) || starts_with(config.source, "rtsp://") ||
                             starts_with(config.source, "http://") ||
                             starts_with(config.source, "https://");
  writer.emit("{\"ts\":\"" + vlpr::utc_timestamp() +
              "\",\"type\":\"ready\",\"version\":\"" VLPR_VERSION "\",\"source\":\"" +
              vlpr::json_escape(safe_source_name(config.source)) + "\"}");

  std::unordered_map<std::string, std::chrono::steady_clock::time_point> recent;
  auto last_heartbeat = std::chrono::steady_clock::now();
  std::uint64_t captured_frames = 0;
  std::uint64_t processed_frames = 0;
  int consecutive_failures = 0;
  while (!stop_requested.load()) {
    cv::Mat frame;
    if (!capture.read(frame) || frame.empty()) {
      ++consecutive_failures;
      if (!reconnectable) {
        break;
      }
      if (consecutive_failures >= 10) {
        writer.emit("{\"ts\":\"" + vlpr::utc_timestamp() + "\",\"type\":\"source_reconnecting\"}");
        capture.release();
        std::this_thread::sleep_for(std::chrono::seconds(1));
        open_capture(capture, config.source);
        consecutive_failures = 0;
        pipeline.reset();
      }
      continue;
    }
    consecutive_failures = 0;
    ++captured_frames;
    if ((captured_frames - 1) % static_cast<std::uint64_t>(config.frame_stride) != 0) {
      continue;
    }
    ++processed_frames;
    const auto result = pipeline.process(frame, true);
    const auto now = std::chrono::steady_clock::now();
    for (const auto& recognition : result.recognitions) {
      if (!accepted(recognition, config)) {
        continue;
      }
      const auto found = recent.find(recognition.ocr.text);
      if (found == recent.end() ||
          now - found->second >= std::chrono::seconds(config.dedupe_seconds)) {
        writer.emit(recognition_json(recognition, result, config.source, "plate"));
        recent[recognition.ocr.text] = now;
      }
    }
    for (auto iterator = recent.begin(); iterator != recent.end();) {
      if (now - iterator->second > std::chrono::seconds(60))
        iterator = recent.erase(iterator);
      else
        ++iterator;
    }
    if (now - last_heartbeat >= std::chrono::seconds(config.heartbeat_seconds)) {
      std::ostringstream heartbeat;
      heartbeat << "{\"ts\":\"" << vlpr::utc_timestamp()
                << "\",\"type\":\"heartbeat\",\"captured_frames\":" << captured_frames
                << ",\"processed_frames\":" << processed_frames << '}';
      writer.emit(heartbeat.str());
      last_heartbeat = now;
    }
  }
  writer.emit("{\"ts\":\"" + vlpr::utc_timestamp() + "\",\"type\":\"stopped\"}");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::signal(SIGINT, request_stop);
    std::signal(SIGTERM, request_stop);
    vlpr::RuntimeConfig config = parse_arguments(argc, argv);
    const auto executable_dir = executable_directory(argv[0]);
    if (config.detector_model.empty()) {
      config.detector_model = find_resource(
          executable_dir, {"models/best_vietnam_lpr.onnx", "../models/best_vietnam_lpr.onnx",
                           "../../models/best_vietnam_lpr.onnx"});
    }
    if (config.recognizer_model.empty()) {
      config.recognizer_model = find_resource(
          executable_dir, {"edge_cpp/models/PP-OCRv6_rec_small.onnx",
                           "models/PP-OCRv6_rec_small.onnx", "../models/PP-OCRv6_rec_small.onnx"});
    }
    if (config.charset_file.empty()) {
      config.charset_file = find_resource(
          executable_dir, {"edge_cpp/models/ppocrv6_chars.txt", "models/ppocrv6_chars.txt",
                           "../models/ppocrv6_chars.txt"});
    }
    if (config.detector_model.empty() || config.recognizer_model.empty() ||
        config.charset_file.empty()) {
      throw std::runtime_error(
          "Model resources were not found; pass --detector, --recognizer and --charset");
    }

    vlpr::Pipeline pipeline(config);
    EventWriter writer(config.event_log);
    if (config.dry_run) {
      writer.emit("{\"ts\":\"" + vlpr::utc_timestamp() +
                  "\",\"type\":\"models_ready\",\"version\":\"" VLPR_VERSION "\"}");
      return 0;
    }
    if (config.benchmark) {
      return run_benchmark(pipeline, config);
    }
    if (config.self_test) {
      return run_image(pipeline, config, writer);
    }
    if (!is_integer(config.source) && std::filesystem::is_regular_file(config.source)) {
      const cv::Mat image = cv::imread(config.source, cv::IMREAD_COLOR);
      if (!image.empty()) {
        return run_image(pipeline, config, writer);
      }
    }
    return run_stream(pipeline, config, writer);
  } catch (const std::exception& error) {
    std::cerr << "{\"ts\":\"" << vlpr::utc_timestamp() << "\",\"type\":\"fatal\",\"message\":\""
              << vlpr::json_escape(error.what()) << "\"}" << std::endl;
    return 1;
  }
}

#include "vlpr/text.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace vlpr {
namespace {

char letter_as_digit(char value) {
  static const std::unordered_map<char, char> mapping{
      {'A', '4'}, {'B', '8'}, {'D', '0'}, {'G', '6'}, {'I', '1'}, {'L', '1'},
      {'O', '0'}, {'Q', '0'}, {'S', '5'}, {'T', '7'}, {'U', '0'}, {'Z', '2'},
  };
  const auto found = mapping.find(value);
  return found == mapping.end() ? value : found->second;
}

char digit_as_letter(char value) {
  static const std::unordered_map<char, char> mapping{
      {'0', 'O'}, {'1', 'I'}, {'2', 'Z'}, {'4', 'A'},
      {'5', 'S'}, {'6', 'G'}, {'7', 'T'}, {'8', 'B'},
  };
  const auto found = mapping.find(value);
  return found == mapping.end() ? value : found->second;
}

std::string ascii_alnum_upper(const std::string& input) {
  std::string output;
  output.reserve(input.size());
  for (const unsigned char value : input) {
    if (value < 128 && std::isalnum(value) != 0) {
      output.push_back(static_cast<char>(std::toupper(value)));
    }
  }
  return output;
}

}  // namespace

std::string clean_plate_text(const std::string& input) {
  std::string output = ascii_alnum_upper(input);
  if (output.size() < 5) {
    return output;
  }

  for (std::size_t index = 0; index < std::min<std::size_t>(2, output.size()); ++index) {
    output[index] = letter_as_digit(output[index]);
  }
  if (output.size() > 2) {
    output[2] = digit_as_letter(output[2]);
  }

  const std::size_t digit_start =
      output.size() >= 9 && std::isalpha(static_cast<unsigned char>(output[3])) != 0 ? 4 : 3;
  if (digit_start == 4) {
    output[3] = digit_as_letter(output[3]);
  }
  for (std::size_t index = digit_start; index < output.size(); ++index) {
    output[index] = letter_as_digit(output[index]);
  }
  return output;
}

bool is_valid_vietnam_plate(const std::string& input) {
  const std::string value = ascii_alnum_upper(input);
  if (value.size() < 7 || value.size() > 10) {
    return false;
  }

  static const std::vector<std::regex> patterns{
      std::regex(R"(^\d{2}[A-Z]\d{4,6}$)"),
      std::regex(R"(^\d{2}[A-Z]{2}\d{4,6}$)"),
      std::regex(R"(^\d{2}[A-Z][1-9]\d{4,6}$)"),
      std::regex(R"(^\d{2,5}(LD|DA|KT|CD|RM|HC|MK|NG|QT|CV|NN)\d{2,6}$)"),
      std::regex(R"(^[ABHKQTPCV][A-Z]\d{4,5}$)"),
  };
  return std::any_of(patterns.begin(), patterns.end(),
                     [&](const std::regex& pattern) { return std::regex_match(value, pattern); });
}

std::string json_escape(const std::string& input) {
  std::ostringstream output;
  for (const unsigned char value : input) {
    switch (value) {
      case '\"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (value < 0x20) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<int>(value) << std::dec;
        } else {
          output << static_cast<char>(value);
        }
    }
  }
  return output.str();
}

std::string utc_timestamp() {
  const auto now = std::chrono::system_clock::now();
  const auto milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
  const std::time_t time = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
#ifdef _WIN32
  gmtime_s(&utc, &time);
#else
  gmtime_r(&time, &utc);
#endif
  std::ostringstream output;
  output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.' << std::setw(3) << std::setfill('0')
         << milliseconds.count() << 'Z';
  return output.str();
}

}  // namespace vlpr

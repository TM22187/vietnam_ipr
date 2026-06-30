#include <iostream>
#include <stdexcept>
#include <string>

#include "vlpr/text.hpp"

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main() {
  require(vlpr::clean_plate_text("51A-691.72") == "51A69172", "normalization failed");
  require(vlpr::clean_plate_text("S1A 69I7Z") == "51A69172", "OCR correction failed");
  require(vlpr::is_valid_vietnam_plate("51A69172"), "car plate should be valid");
  require(vlpr::is_valid_vietnam_plate("59U209978"), "motorcycle plate should be valid");
  require(!vlpr::is_valid_vietnam_plate("1234"), "short text should be invalid");
  require(vlpr::json_escape("a\"b\nc") == "a\\\"b\\nc", "JSON escaping failed");
  require(!vlpr::utc_timestamp().empty(), "timestamp should not be empty");
  std::cout << "vlpr_text_tests: ok\n";
  return 0;
}

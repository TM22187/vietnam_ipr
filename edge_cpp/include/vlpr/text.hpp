#pragma once

#include <string>

namespace vlpr {

std::string clean_plate_text(const std::string& input);
bool is_valid_vietnam_plate(const std::string& input);
std::string json_escape(const std::string& input);
std::string utc_timestamp();

}  // namespace vlpr

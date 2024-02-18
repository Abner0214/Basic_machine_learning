#include <cmath>

extern "C" double calculate_distance(double lat1, double lon1, double lat2, double lon2) {
    // Implement the distance calculation logic
    if (std::isnan(lat1) || std::isnan(lon1) || std::isnan(lat2) || std::isnan(lon2)) {
        return NAN;
    }
    // Using Haversine formula for simplicity, you can replace it with your preferred formula
    double dlat = lat2 - lat1;
    double dlon = lon2 - lon1;
    double a = std::sin(dlat / 2.0) * std::sin(dlat / 2.0) +
               std::cos(lat1) * std::cos(lat2) * std::sin(dlon / 2.0) * std::sin(dlon / 2.0);
    double c = 2.0 * std::atan2(std::sqrt(a), std::sqrt(1.0 - a));
    const double earth_radius = 6371000.0;  // Earth radius in meters
    return earth_radius * c;
}

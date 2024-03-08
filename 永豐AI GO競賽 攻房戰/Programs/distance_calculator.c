#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double calculate_distance(double lat1, double lon1, double lat2, double lon2) {
    // Radius of the Earth in meters
    double R = 6371000.0;

    // Convert latitude and longitude from degrees to radians
    lat1 *= M_PI / 180.0;
    lon1 *= M_PI / 180.0;
    lat2 *= M_PI / 180.0;
    lon2 *= M_PI / 180.0;

    // Calculate differences in coordinates
    double dlat = lat2 - lat1;
    double dlon = lon2 - lon1;

    // Haversine formula to calculate distance
    double a = sin(dlat / 2.0) * sin(dlat / 2.0) + cos(lat1) * cos(lat2) * sin(dlon / 2.0) * sin(dlon / 2.0);
    double c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a));

    // Distance in meters
    double distance = R * c;

    return distance;
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s lat1 lon1 lat2 lon2\n", argv[0]);
        return 1;
    }

    // Parse command-line arguments as doubles
    double lat1 = atof(argv[1]);
    double lon1 = atof(argv[2]);
    double lat2 = atof(argv[3]);
    double lon2 = atof(argv[4]);

    // Calculate distance
    double distance = calculate_distance(lat1, lon1, lat2, lon2);

    // Print the result
    printf("%f", distance);

    return 0;
}

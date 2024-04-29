#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Read data/dyn_pricing.csv
void read_pricing_csv(const char *filename, double *price) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: File not found\n");
        return;
    }

    char line[1024];
    int i = 0;

    // Skip the first line
    fgets(line, 1024, file);

    while (fgets(line, 1024, file)) {
        char *tmp = strdup(line);
        strtok(tmp, ","); // Skip the date_time column
        char *price_str = strtok(NULL, ","); // Read the price column
        if (price_str != NULL) {
            price[4*i + 0] = atof(price_str);
            price[4*i + 1] = atof(price_str);
            price[4*i + 2] = atof(price_str);
            price[4*i + 3] = atof(price_str);
            i++;
        }
        free(tmp);
    }
    fclose(file);
}

void read_solar_csv(const char *filename, double *solarGeneration) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: File not found\n");
        return;
    }

    char line[1024];
    int i = 0;

    // Skip the first line
    fgets(line, 1024, file);

    while (fgets(line, 1024, file)) {
        char *tmp = strdup(line);
        strtok(tmp, ";"); // Skip the date_time column
        char *solar_str = strtok(NULL, ";"); // Read the solar column
        if (solar_str != NULL) {
            solarGeneration[i] = atof(solar_str);
            i++;
        }
        free(tmp);
    }
    fclose(file);
}

double* read_device_csv(const char *filename, int &size) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: File not found\n");
        return NULL;
    }

    // Skip the header line
    char line[1024];
    fgets(line, 1024, file);

    // First pass: count the number of lines
    int count = 0;
    while (fgets(line, 1024, file)) {
        count++;
    }
    rewind(file); // Reset the file pointer to the beginning

    // Skip the header line again
    fgets(line, 1024, file);

    // Allocate the array
    double *dishwasher = new double[count];

    // Second pass: read the data
    int i = 0;
    while (fgets(line, 1024, file)) {
        char *tmp = strdup(line);
        strtok(tmp, ","); // Skip the timestamp column
        char *dishwasher_str = strtok(NULL, ","); // Read the value column
        if (dishwasher_str != NULL) {
            dishwasher[i] = atof(dishwasher_str);
            i++;
        }
        free(tmp);
    }
    fclose(file);

    size = count;
    return dishwasher;
}

int main(void) {
    // Initialize 96 length array
    double price[96];
    double *price_ptr = price;

    double solar[96];
    double *solar_ptr = solar;

    // Read CSV files
    read_pricing_csv("data/dyn_pricing_1.csv", price);
    read_solar_csv("data/solar_1.csv", solar);

    // READ DEVICES

    // dishwasher
    int dishwasher_size;
    double *dishwasher = read_device_csv("data/dishw_1.csv", dishwasher_size);

    // oven
    int oven_size;
    double *oven = read_device_csv("data/oven_1.csv", oven_size);


    // Print each dishwasher element
    for(int i = 0; i < oven_size; i++) {
        printf("%f\n", oven[i]);
    }
}


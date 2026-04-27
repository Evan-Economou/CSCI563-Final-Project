#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void safe_softmax(const float* x, float* y, int V);

static int read_file(std::vector<float>* data, char* fName){
    FILE* fin = std::fopen(fName, "r");
    if (!fin) { std::perror(fName); return 1; }
    
    float val;
    while (std::fscanf(fin, "%f", &val) == 1)
        data->push_back(val);
    std::fclose(fin);

    if (data->empty()) {
        std::fprintf(stderr, "error: no floats found in '%s'\n", fName);
        return 1;
    }

    return 0;
}


#endif
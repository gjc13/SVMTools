#include <iostream>
#include "svm.h"
#include "SVMTools.h"

using namespace std;

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        cout << "Usage: NeighbourSVM tagFile imageDir saveFile";
    }
    const char *tagFile = argv[1];
    const char *imageDir = argv[2];

    svm_parameter parameter = {
            .svm_type = C_SVC,
            .kernel_type = RBF,
            .gamma = 1.0,
            .cache_size = 512,
            .eps = 0.001,
            .C = 2,
            .nr_weight = 0,
            .shrinking = 1,
            .probability = 0
    };
    SVMBuilder builder(parameter, tagFile, imageDir);
    builder.load();
    builder.build();
    builder.reTest();
    svm_model model = builder.getModel();
    svm_save_model(argv[3], &model);
    SVMBuilder tester(parameter, "/Users/gjc13/Dataset/border_filter/tags1.txt",
                      "/Users/gjc13/Dataset/border_filter/neighbour_images1");
    tester.setModel(model);
    tester.load();
    tester.reTest();
    return 0;
}
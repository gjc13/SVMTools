//
// Created by 郭嘉丞 on 15/10/28.
//

#ifndef NEIGHBOURSVM_SVM_TOOLS_H
#define NEIGHBOURSVM_SVM_TOOLS_H

#include "svm.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct TaggedImage
{
    cv::Mat image;
    int tag;
};

class SVMBuilder
{
public:
    SVMBuilder(const svm_parameter &parameter, const char *tagFilename, const char *imageDir)
            : svmParameter(parameter), tagFilename(tagFilename), imageDir(imageDir)
    { }

    svm_model getModel()
    { return svmModel; }

    void setModel(const svm_model &newModel)
    { svmModel = newModel; }

    svm_parameter getParameter()
    { return svmParameter; }

    void build();

    void load();

    void reTest();

    ~SVMBuilder()
    {
        if (tags != nullptr) delete[] tags;
        for (int i = 0; i < numSamples; i++)
        {
            delete[] svm_nodes[i];
        }
        delete[] svm_nodes;
    }

private:
    svm_problem load_problem();

    std::vector<TaggedImage> getTaggedImages();

    svm_node *getSVMNode(cv::Mat image);

    svm_problem problem;
    svm_model svmModel;
    svm_parameter svmParameter;
    std::string tagFilename;
    std::string imageDir;
    int numSamples = -1;
    double *tags = nullptr;
    svm_node **svm_nodes = nullptr;
};


#endif //NEIGHBOURSVM_SVM_TOOLS_H

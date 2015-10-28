//
// Created by 郭嘉丞 on 15/10/28.
//

#include "SVMTools.h"
#include <fstream>
#include <iostream>


using namespace std;
using namespace cv;

vector<TaggedImage> SVMBuilder::getTaggedImages()
{
    vector<TaggedImage> taggedImages;
    ifstream fTag(tagFilename, fstream::in);
    string filename;
    int tag;
    fTag >> filename >> tag;
    while (fTag)
    {
        filename = imageDir + "/" + filename;
        TaggedImage taggedImage;
        taggedImage.image = imread(filename.c_str());
        taggedImage.tag = tag;
        if (taggedImage.image.rows == 0 || taggedImage.image.cols == 0)
        {
            cerr << "Cannot read image " << filename << endl;
            continue;
        }
        taggedImages.push_back(taggedImage);
        fTag >> filename >> tag;
    }
    fTag.close();
    return taggedImages;
}

svm_problem SVMBuilder::load_problem()
{
    vector<TaggedImage> taggedImages = getTaggedImages();
    numSamples = (int) taggedImages.size();
    tags = new double[numSamples];
    svm_nodes = new svm_node *[numSamples];
    for (int i = 0; i < taggedImages.size(); i++)
    {
        svm_node *node = getSVMNode(taggedImages[i].image);
        svm_nodes[i] = node;
        tags[i] = taggedImages[i].tag;
    }
    svm_problem problem =
            {
                    .l = numSamples,
                    .x = svm_nodes,
                    .y = tags
            };
    return problem;
}

svm_node *SVMBuilder::getSVMNode(cv::Mat image)
{
    //suppose image is CV_8UC3
    int dim = image.rows * image.cols;
    svm_node *node = new svm_node[dim * 3 + 1];
    Mat channels[3];
    split(image, channels);
    //we use value between 0~1 for sake of SVM
    for (int i = 0; i < 3; i++)
    {
        channels[i].convertTo(channels[i], CV_32FC1, 1.0 / 255.0);
    }
    int index = 0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < image.rows; j++)
            for (int k = 0; k < image.cols; k++)
            {
                node[index].index = index;
                node[index].value = channels[i].at<float>(j, k);
                index++;
            }
    node[index].index = -1;
    return node;
}

void SVMBuilder::load()
{
    problem = load_problem();
}

void SVMBuilder::build()
{
    const char * msg = svm_check_parameter(&problem, &svmParameter);
    if(msg)
    {
        cerr << msg << endl;
        return;
    }
    svm_model *modelPtr = svm_train(&problem, &svmParameter);
    svmModel = *modelPtr;
    delete(modelPtr);
}

void SVMBuilder::reTest()
{
    int numSuccess[2] = {0,0};
    int numFail[2] = {0, 0};
    for(int i = 0; i < numSamples; i++)
    {
        if(svm_predict(&svmModel, svm_nodes[i]) == tags[i])
            numSuccess[int(tags[i])]++;
        else
            numFail[int(tags[i])]++;
    }
    cout << "Object:" << endl;
    cout << numSuccess[1] << " " << numFail[1] << endl;
    cout << "Backgroud:" << endl;
    cout << numSuccess[0] << " " << numFail[0] << endl;
}

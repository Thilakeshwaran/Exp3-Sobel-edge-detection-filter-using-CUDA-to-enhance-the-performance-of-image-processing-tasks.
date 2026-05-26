# Exp3-Sobel-edge-detection-filter-using-CUDA-to-enhance-the-performance-of-image-processing-tasks.
<h3>NAME : Thilakeswaran KP</h3>
<h3>REGISTER NO : 212223230232</h3>
<h3>EX. NO : 3</h3>
<h3>DATE : 24/05/2026</h3>
<h1> <align=center> Sobel edge detection filter using CUDA </h3>
  Implement Sobel edge detection filtern using GPU.</h3>
Experiment Details:
  
## AIM:
  The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two kernels to determine the gradient in both the x and y directions. This lab focuses on utilizing CUDA to parallelize the Sobel filter implementation for efficient processing of images.

Code Overview: You will work with the provided CUDA implementation of the Sobel edge detection filter. The code reads an input image, applies the Sobel filter in parallel on the GPU, and writes the result to an output image.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
CUDA Toolkit and OpenCV installed.
A sample image for testing.

## Technologies Used

- CUDA Toolkit
- OpenCV with CUDA support
- C++
- NVIDIA GPU

---

## Project Structure

```text
CUDA-GPU-Image-Processing/
│
├── input_images/
├── output_images/
├── main.cpp
├── Makefile
├── run.sh
├── README.md
└── execution_log.txt
```

---
## CUDA Image Processing Code 
```
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: ./image_processor <input_folder> <output_folder>" << std::endl;
        return -1;
    }

    std::string input_folder = argv[1];
    std::string output_folder = argv[2];

    for (const auto& entry : fs::directory_iterator(input_folder)) {
        std::string image_path = entry.path().string();

        cv::Mat image = cv::imread(image_path);

        if (image.empty()) {
            std::cout << "Failed to load image: " << image_path << std::endl;
            continue;
        }

        cv::cuda::GpuMat gpu_input;
        gpu_input.upload(image);

        cv::cuda::GpuMat gpu_gray;
        cv::cuda::cvtColor(gpu_input, gpu_gray, cv::COLOR_BGR2GRAY);

        cv::Ptr<cv::cuda::Filter> gaussian_filter =
            cv::cuda::createGaussianFilter(
                gpu_gray.type(),
                gpu_gray.type(),
                cv::Size(5, 5),
                1.5);

        cv::cuda::GpuMat gpu_blur;
        gaussian_filter->apply(gpu_gray, gpu_blur);

        cv::cuda::GpuMat gpu_edges;
        cv::cuda::Canny(gpu_blur, gpu_edges, 50.0, 150.0);

        cv::Mat result;
        gpu_edges.download(result);

        std::string output_path = output_folder + "/processed_" +
                                  entry.path().filename().string();

        cv::imwrite(output_path, result);

        std::cout << "Processed: " << image_path << std::endl;
    }

    std::cout << "All images processed successfully." << std::endl;

    return 0;
}
```
## OUTPUT
<img width="1402" height="1122" alt="output" src="https://github.com/user-attachments/assets/2a502919-cb81-4854-8772-8091f817dda2" />

## RESULT
The project successfully demonstrates GPU accelerated batch image processing using CUDA and OpenCV. The implementation shows how parallel GPU computation can efficiently process large-scale image datasets.

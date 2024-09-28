# A Printed Character Recognition System for Meetei-Mayek Script Using Transfer Learning
[![Springer Link](https://img.shields.io/badge/Springer-Link-blue)](https://doi.org/10.1007/978-981-99-4713-3_50)

## 👥 Authors

- [**Vishwaksena Vishnu Simha Dingari**](https://www.linkedin.com/in/vishwaksena-dingari)
- [**Ganapathi Kosanam**](https://www.linkedin.com/in/kosanam-ganapathi/)
- [**Chavatapalli Devi Sri Shankar**](https://www.linkedin.com/in/chavatapalli-devi-sri-shankar-b6643b246/)
- [**Dr. Chingakham Neeta Devi**](https://www.nitmanipur.ac.in/Department_FacultyProfileNew.aspx?nDeptID=oq)

## 🛠️ Technologies Used

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikitlearn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Gradio-FF9900?style=flat-square&logo=gradio&logoColor=white" />
  <img src="https://img.shields.io/badge/Pillow-3776AB?style=flat-square&logo=python&logoColor=white" />
</p>


## 📝 Abstract

This paper presents a printed character recognition system for Meetei-Mayek script of Manipuri language. The input to the system is downloaded and cropped images containing printed characters of Meetei-Mayek. The images go through pre-processing which involves thresholding and Morphological Transformations. Segmentation of characters has been carried out using image processing techniques like dilation, thinning and text properties like height, width, and aspect ratio. After segmentation, the images are converted into three channeled black and white images. The work comprises development of standard database for printed characters for Meetei-Mayek. The classification task has been carried out using transfer learning using pre-trained VGG-16, VGG-19, and ResNet152-V2 convolutional neural networks (CNNs). The recognition results have been compelling, and VGG-16 has achieved better classification accuracy in comparison with other pre-trained CNN—VGG-19, ResNet152-V2.


# 🖼️ Example Outputs of OCR Application

### VGG16
![VGG16](./output_images/vgg16.png)

### VGG19
![VGG19](./output_images/vgg19.png)

### MobileNetV2
![MobileNetV2](./output_images/mobilenet.png)

## 🚀 Key Features

- Robust pre-processing and segmentation techniques
- Transfer learning with state-of-the-art CNNs
- High accuracy in Meetei-Mayek character recognition
- Scalable architecture for potential application to other scripts

## 📊 Performance Comparison


## 📚 Citation

If you use this work in your research, please cite: [https://doi.org/10.1007/978-981-99-4713-3_50](https://doi.org/10.1007/978-981-99-4713-3_50)

### bibtex
```bibtex
@inproceedings{dingari2023printed,
  title={A Printed Character Recognition System for Meetei-Mayek Script Using Transfer Learning},
  author={Dingari, Vishwaksena Vishnu Simha and Kosanam, Ganapathi and Chavatapalli, Devi Sri Shankar and Devi, Chingakham Neeta},
  booktitle={International Conference on Science, Technology and Engineering},
  pages={519--528},
  year={2023},
  organization={Springer}
}
```

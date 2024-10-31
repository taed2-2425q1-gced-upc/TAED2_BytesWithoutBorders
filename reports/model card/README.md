{\rtf1\ansi\ansicpg1252\cocoartf2818
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\froman\fcharset0 Times-Bold;\f1\froman\fcharset0 Times-Roman;\f2\fmodern\fcharset0 Courier;
}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red0\green0\blue233;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;\cssrgb\c0\c0\c93333;}
{\*\listtable{\list\listtemplateid1\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid1\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid1}
{\list\listtemplateid2\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid101\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid2}
{\list\listtemplateid3\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid201\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid3}}
{\*\listoverridetable{\listoverride\listid1\listoverridecount0\ls1}{\listoverride\listid2\listoverridecount0\ls2}{\listoverride\listid3\listoverridecount0\ls3}}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\sa321\partightenfactor0

\f0\b\fs48 \cf0 \expnd0\expndtw0\kerning0
Model Card for FashionMNIST\
\pard\pardeftab720\sa240\partightenfactor0

\fs24 \cf0 FashionMNIST
\f1\b0  is a dataset designed as a drop-in replacement for the MNIST dataset, with a focus on classifying images of clothing items.\
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 Model Summary\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 The model performs 
\f0\b multi-class image classification
\f1\b0  to distinguish between 10 different types of clothing items.\
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 Model Details\
\pard\pardeftab720\sa280\partightenfactor0

\fs28 \cf0 Model Description\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 The model is designed to classify 28x28 grayscale images into one of 10 fashion categories, based on the FashionMNIST dataset.\

\f0\b Developed by
\f1\b0 : Zalando Research\uc0\u8232 
\f0\b License
\f1\b0 : MIT License\uc0\u8232 
\f0\b Model Type
\f1\b0 : Image Classification\uc0\u8232 
\f0\b Language(s)
\f1\b0 : N/A\uc0\u8232 
\f0\b Finetuned from model
\f1\b0 : N/A\
\pard\pardeftab720\sa280\partightenfactor0

\f0\b\fs28 \cf0 Model Sources\
\pard\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls1\ilvl0
\fs24 \cf0 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Repository
\f1\b0 : {\field{\*\fldinst{HYPERLINK "https://github.com/zalandoresearch/fashion-mnist"}}{\fldrslt \cf3 \ul \ulc3 GitHub}}\
\ls1\ilvl0
\f0\b \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Paper
\f1\b0 : {\field{\*\fldinst{HYPERLINK "https://arxiv.org/pdf/1708.07747.pdf"}}{\fldrslt \cf3 \ul \ulc3 FashionMNIST Paper}}\
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 Uses\
\pard\pardeftab720\sa280\partightenfactor0

\fs28 \cf0 Direct Use\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 The model is intended for research and development, especially for benchmarking classification models.\
\pard\pardeftab720\sa280\partightenfactor0

\f0\b\fs28 \cf0 Downstream Use\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 Can be used as a baseline for more complex deep learning models on more challenging tasks.\
\pard\pardeftab720\sa280\partightenfactor0

\f0\b\fs28 \cf0 Out-of-Scope Use\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 Not suitable for real-time fashion recommendation systems without further training and optimization.\
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 Bias, Risks, and Limitations\
\pard\pardeftab720\sa280\partightenfactor0

\fs28 \cf0 Bias\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 The dataset may not fully represent fashion diversity across different cultures and regions.\
\pard\pardeftab720\sa280\partightenfactor0

\f0\b\fs28 \cf0 Risks\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 Overfitting to the training dataset is a risk when models are overly specialized in image classification tasks on small datasets.\
\pard\pardeftab720\sa280\partightenfactor0

\f0\b\fs28 \cf0 Limitations\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 The images are small (28x28 pixels), which limits real-world applications unless further finetuned.\
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 Recommendations\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 Users should be cautious about generalizing results from the dataset to real-world applications, especially in commercial systems where higher resolution images are necessary.\
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 How to Get Started with the Model\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 Use the following Python code to get started:\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 python\
\
from tensorflow.keras.datasets import fashion_mnist\
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()\
\
# Normalize pixel values\
train_images, test_images = train_images / 255.0, test_images / 255.0\
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 Training Details\
\pard\pardeftab720\sa280\partightenfactor0

\fs28 \cf0 Training Data\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 FashionMNIST dataset contains 60,000 training images and 10,000 test images.\
\pard\pardeftab720\sa280\partightenfactor0

\f0\b\fs28 \cf0 Training Procedure\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 The model can be trained using various deep learning architectures like CNNs (Convolutional Neural Networks).\
\pard\pardeftab720\sa319\partightenfactor0

\f0\b \cf0 Training Hyperparameters:\
\pard\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0\cf0 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Batch Size
\f1\b0 : 32\
\ls2\ilvl0
\f0\b \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Learning Rate
\f1\b0 : 0.001\
\ls2\ilvl0
\f0\b \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Epochs
\f1\b0 : 10-20\
\pard\pardeftab720\sa280\partightenfactor0

\f0\b\fs28 \cf0 Evaluation\
\pard\pardeftab720\sa319\partightenfactor0

\fs24 \cf0 Testing Data\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0 \cf0 The same FashionMNIST dataset's test set is used for evaluation, containing 10,000 images.\
\pard\pardeftab720\sa319\partightenfactor0

\f0\b \cf0 Metrics\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0 \cf0 The model is evaluated on accuracy, precision, recall, and F1-score.\
\pard\pardeftab720\sa280\partightenfactor0

\f0\b\fs28 \cf0 Results\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 On standard architectures like CNNs, the model typically achieves around 
\f0\b 90%
\f1\b0  accuracy on the test dataset.\
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 Environmental Impact\
\pard\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls3\ilvl0
\fs24 \cf0 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Hardware Type
\f1\b0 : GPU\
\ls3\ilvl0
\f0\b \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Hours used
\f1\b0 : ~5 hours (depending on the hardware)\
\ls3\ilvl0
\f0\b \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Cloud Provider
\f1\b0 : [More Information Needed]\
\ls3\ilvl0
\f0\b \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Carbon Emitted
\f1\b0 : [More Information Needed]\
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 Model Examination\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 The model is robust for academic purposes but may not generalize well for larger, higher-quality image datasets without modification.\
}
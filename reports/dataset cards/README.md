{\rtf1\ansi\ansicpg1252\cocoartf2818
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 HelveticaNeue;}
{\colortbl;\red255\green255\blue255;\red42\green49\blue64;\red255\green255\blue255;\red88\green94\blue109;
\red59\green67\blue80;\red24\green30\blue42;\red240\green241\blue244;\red247\green249\blue250;\red14\green18\blue29;
}
{\*\expandedcolortbl;;\cssrgb\c21569\c25490\c31765;\cssrgb\c100000\c100000\c100000;\cssrgb\c41961\c44706\c50196;
\cssrgb\c29412\c33333\c38824;\cssrgb\c12157\c16078\c21569;\cssrgb\c95294\c95686\c96471\c70196;\cssrgb\c97647\c98039\c98431;\cssrgb\c6667\c9412\c15294;
}
{\*\listtable{\list\listtemplateid1\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid1\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid1}
{\list\listtemplateid2\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid101\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid2}}
{\*\listoverridetable{\listoverride\listid1\listoverridecount0\ls1}{\listoverride\listid2\listoverridecount0\ls2}}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs40 \cf2 \cb3 \expnd0\expndtw0\kerning0
Dataset Card for FashionMNIST\cb1 \
\pard\pardeftab720\partightenfactor0

\fs36 \cf4 \
\pard\pardeftab720\partightenfactor0

\fs35\fsmilli17600 \cf2 \cb3 Dataset Summary\cb1 \
\pard\pardeftab720\sa420\partightenfactor0

\fs33\fsmilli16800 \cf5 \cb3 Fashion-MNIST is a dataset of Zalando's article images\'97consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. We intend Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.\cb1 \
\pard\pardeftab720\partightenfactor0

\fs36 \cf4 \
\pard\pardeftab720\partightenfactor0

\fs35\fsmilli17600 \cf2 \cb3 Supported Tasks and Leaderboards\cb1 \
\pard\tx220\tx720\pardeftab720\li720\fi-720\sa168\partightenfactor0
\ls1\ilvl0
\fs28 \cf6 \cb7 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
image-classification
\fs33\fsmilli16800 \cf5 \cb3 : The goal of this task is to classify a given image of Zalando's article into one of 10 classes. 
\fs36 \cf4 \cb1 \
\pard\pardeftab720\partightenfactor0

\fs35\fsmilli17600 \cf2 \cb3 Languages\cb1 \
\pard\pardeftab720\sa420\partightenfactor0

\fs33\fsmilli16800 \cf5 \cb3 [More Information Needed]\cb1 \
\pard\pardeftab720\partightenfactor0

\fs36 \cf4 \
\pard\pardeftab720\partightenfactor0

\fs37\fsmilli18560 \cf2 \cb3 Dataset Structure\cb1 \
\pard\pardeftab720\partightenfactor0

\fs36 \cf4 \
\pard\pardeftab720\partightenfactor0

\fs35\fsmilli17600 \cf2 \cb3 Data Instances\cb1 \
\pard\pardeftab720\sa420\partightenfactor0

\fs33\fsmilli16800 \cf5 \cb3 A data point comprises an image and its label.\cb1 \
\pard\pardeftab720\partightenfactor0

\fs28 \cf6 \cb8 \{\
  'image': <PIL.PngImagePlugin.PngImageFile image mode=L size=28x28 at 0x27601169DD8>,\
  'label': 9\
\}\
\pard\pardeftab720\qc\partightenfactor0

\fs33\fsmilli16800 \cf5 \cb1 \
\pard\pardeftab720\partightenfactor0

\fs36 \cf4 \
\pard\pardeftab720\partightenfactor0

\fs35\fsmilli17600 \cf2 \cb3 Data Fields\cb1 \
\pard\tx220\tx720\pardeftab720\li720\fi-720\sa168\partightenfactor0
\ls2\ilvl0
\fs28 \cf6 \cb7 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
image
\fs33\fsmilli16800 \cf5 \cb3 : A 
\fs28 \cf6 \cb7 PIL.Image.Image
\fs33\fsmilli16800 \cf5 \cb3  object containing the 28x28 image. Note that when accessing the image column: 
\fs28 \cf6 \cb7 dataset[0]["image"]
\fs33\fsmilli16800 \cf5 \cb3  the image file is automatically decoded. Decoding of a large number of image files might take a significant amount of time. Thus it is important to first query the sample index before the 
\fs28 \cf6 \cb7 "image"
\fs33\fsmilli16800 \cf5 \cb3  column, i.e. 
\fs28 \cf6 \cb7 dataset[0]["image"]
\fs33\fsmilli16800 \cf5 \cb3  should \cf6 always\cf5  be preferred over 
\fs28 \cf6 \cb7 dataset["image"][0]
\fs33\fsmilli16800 \cf5 \cb3 .\cb1 \
\pard\tx220\tx720\pardeftab720\li720\fi-720\sa168\partightenfactor0
\ls2\ilvl0
\fs28 \cf6 \cb7 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
label
\fs33\fsmilli16800 \cf5 \cb3 : an integer between 0 and 9 representing the classes with the following mapping:
\fs29\fsmilli14700 \cb1 \

\itap1\trowd \taflags1 \trgaph108\trleft-108 \tamart588 \tamarb588 \trbrdrt\brdrnil \trbrdrl\brdrnil \trbrdrr\brdrnil 
\clvertalb \clshdrawnil \clwWidth5325\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt20 \clpadl0 \clpadb168 \clpadr168 \gaph\cellx4320
\clvertalb \clshdrawnil \clwWidth10003\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt20 \clpadl168 \clpadb168 \clpadr0 \gaph\cellx8640
\pard\intbl\itap1\tx220\tx720\pardeftab720\li720\fi-720\qc\partightenfactor0
\ls2\ilvl0\cf9 \cb3 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Label\cb1 \cell 
\pard\intbl\itap1\tx220\tx720\pardeftab720\li720\fi-720\qc\partightenfactor0
\ls2\ilvl0\cf9 \cb3 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Description\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \tamart588 \tamarb588 \trbrdrl\brdrnil \trbrdrr\brdrnil 
\clvertalt\clvertalbase \clshdrawnil \clwWidth5325\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt168 \clpadl0 \clpadb168 \clpadr168 \gaph\cellx4320
\clvertalt\clvertalbase \clshdrawnil \clwWidth10003\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt168 \clpadl168 \clpadb168 \clpadr0 \gaph\cellx8640
\pard\intbl\itap1\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0\cf5 \cb3 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
0\cb1 \cell 
\pard\intbl\itap1\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0\cf5 \cb3 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
T-shirt/top\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \tamart588 \tamarb588 \trbrdrl\brdrnil \trbrdrr\brdrnil 
\clvertalt\clvertalbase \clshdrawnil \clwWidth5325\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt168 \clpadl0 \clpadb168 \clpadr168 \gaph\cellx4320
\clvertalt\clvertalbase \clshdrawnil \clwWidth10003\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt168 \clpadl168 \clpadb168 \clpadr0 \gaph\cellx8640
\pard\intbl\itap1\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0\cf5 \cb3 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
1\cb1 \cell 
\pard\intbl\itap1\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0\cf5 \cb3 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Trouser\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \tamart588 \tamarb588 \trbrdrl\brdrnil \trbrdrr\brdrnil 
\clvertalt\clvertalbase \clshdrawnil \clwWidth5325\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt168 \clpadl0 \clpadb168 \clpadr168 \gaph\cellx4320
\clvertalt\clvertalbase \clshdrawnil \clwWidth10003\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt168 \clpadl168 \clpadb168 \clpadr0 \gaph\cellx8640
\pard\intbl\itap1\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0\cf5 \cb3 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
2\cb1 \cell 
\pard\intbl\itap1\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0\cf5 \cb3 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Pullover\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \tamart588 \tamarb588 \trbrdrl\brdrnil \trbrdrr\brdrnil 
\clvertalt\clvertalbase \clshdrawnil \clwWidth5325\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt168 \clpadl0 \clpadb168 \clpadr168 \gaph\cellx4320
\clvertalt\clvertalbase \clshdrawnil \clwWidth10003\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt168 \clpadl168 \clpadb168 \clpadr0 \gaph\cellx8640
\pard\intbl\itap1\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0\cf5 \cb3 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
3\cb1 \cell 
\pard\intbl\itap1\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0\cf5 \cb3 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Dress\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \tamart588 \tamarb588 \trbrdrl\brdrnil \trbrdrr\brdrnil 
\clvertalt\clvertalbase \clshdrawnil \clwWidth5325\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt168 \clpadl0 \clpadb168 \clpadr168 \gaph\cellx4320
\clvertalt\clvertalbase \clshdrawnil \clwWidth10003\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt168 \clpadl168 \clpadb168 \clpadr0 \gaph\cellx8640
\pard\intbl\itap1\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0\cf5 \cb3 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
4\cb1 \cell 
\pard\intbl\itap1\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0\cf5 \cb3 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Coat\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \tamart588 \tamarb588 \trbrdrl\brdrnil \trbrdrr\brdrnil 
\clvertalt\clvertalbase \clshdrawnil \clwWidth5325\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt168 \clpadl0 \clpadb168 \clpadr168 \gaph\cellx4320
\clvertalt\clvertalbase \clshdrawnil \clwWidth10003\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt168 \clpadl168 \clpadb168 \clpadr0 \gaph\cellx8640
\pard\intbl\itap1\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0\cf5 \cb3 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
5\cb1 \cell 
\pard\intbl\itap1\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0\cf5 \cb3 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Sandal\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \tamart588 \tamarb588 \trbrdrl\brdrnil \trbrdrr\brdrnil 
\clvertalt\clvertalbase \clshdrawnil \clwWidth5325\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt168 \clpadl0 \clpadb168 \clpadr168 \gaph\cellx4320
\clvertalt\clvertalbase \clshdrawnil \clwWidth10003\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt168 \clpadl168 \clpadb168 \clpadr0 \gaph\cellx8640
\pard\intbl\itap1\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0\cf5 \cb3 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
6\cb1 \cell 
\pard\intbl\itap1\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0\cf5 \cb3 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Shirt\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \tamart588 \tamarb588 \trbrdrl\brdrnil \trbrdrr\brdrnil 
\clvertalt\clvertalbase \clshdrawnil \clwWidth5325\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt168 \clpadl0 \clpadb168 \clpadr168 \gaph\cellx4320
\clvertalt\clvertalbase \clshdrawnil \clwWidth10003\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt168 \clpadl168 \clpadb168 \clpadr0 \gaph\cellx8640
\pard\intbl\itap1\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0\cf5 \cb3 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
7\cb1 \cell 
\pard\intbl\itap1\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0\cf5 \cb3 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Sneaker\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \tamart588 \tamarb588 \trbrdrl\brdrnil \trbrdrr\brdrnil 
\clvertalt\clvertalbase \clshdrawnil \clwWidth5325\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt168 \clpadl0 \clpadb168 \clpadr168 \gaph\cellx4320
\clvertalt\clvertalbase \clshdrawnil \clwWidth10003\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt168 \clpadl168 \clpadb168 \clpadr0 \gaph\cellx8640
\pard\intbl\itap1\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0\cf5 \cb3 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
8\cb1 \cell 
\pard\intbl\itap1\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0\cf5 \cb3 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Bag\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \tamart588 \tamarb588 \trbrdrl\brdrnil \trbrdrt\brdrnil \trbrdrr\brdrnil 
\clvertalt\clvertalbase \clshdrawnil \clwWidth5325\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt168 \clpadl0 \clpadb168 \clpadr168 \gaph\cellx4320
\clvertalt\clvertalbase \clshdrawnil \clwWidth10003\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt168 \clpadl168 \clpadb168 \clpadr0 \gaph\cellx8640
\pard\intbl\itap1\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0\cf5 \cb3 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
9\cb1 \cell 
\pard\intbl\itap1\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0\cf5 \cb3 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Ankle boot\cb1 \cell \lastrow\row
\pard\tx220\tx720\pardeftab720\li720\fi-720\sa168\partightenfactor0
\ls2\ilvl0
\fs33\fsmilli16800 \cf5 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\
\pard\pardeftab720\partightenfactor0

\fs36 \cf4 \
\pard\pardeftab720\partightenfactor0

\fs35\fsmilli17600 \cf2 \cb3 Data Splits\cb1 \
\pard\pardeftab720\sa420\partightenfactor0

\fs33\fsmilli16800 \cf5 \cb3 The data is split into training and test set. The training set contains 60,000 images and the test set 10,000 images.\cb1 \
\pard\pardeftab720\partightenfactor0

\fs36 \cf4 \
\pard\tx220\tx720\pardeftab720\li720\fi-720\sa168\partightenfactor0

\fs33\fsmilli16800 \cf5 Evaluation Metrics\
\
Since Fashion MNIST is a classification dataset, the common evaluation metrics include:\
\
Accuracy: The percentage of correctly classified images.\
Precision, Recall, F1-Score: Used to measure the performance of individual classes, especially in multi-class classification.\
Confusion Matrix: Useful for visualizing class-wise performance.\
Recommended Benchmarks\
\
For state-of-the-art performance comparison, some common baseline models include:\
\
Logistic Regression: 0.84 accuracy\
Support Vector Machines (SVM): 0.89 accuracy\
Convolutional Neural Networks (CNN): 0.93+ accuracy\
Dataset Provenance and Usage Rights\
\
Source: Collected and curated by Zalando, a German fashion retailer, to serve as a more challenging alternative to MNIST for benchmarking machine learning models.\
License: The dataset is released under the MIT License, allowing for wide usage in both academic research and commercial projects.\
Considerations\
\
Bias: Fashion MNIST is a curated dataset with balanced classes, so there is minimal class imbalance. However, it's still important to assess if the model generalizes well to more diverse clothing datasets.\
Limitations: Since the dataset only includes grayscale images of a fixed size, it may not represent the diversity of real-world fashion items, which often vary in color, shape, and texture.\
\pard\pardeftab720\partightenfactor0

\fs36 \cf4 \
\pard\pardeftab720\partightenfactor0

\fs35\fsmilli17600 \cf2 \cb3 Personal and Sensitive Information\cb1 \
\pard\pardeftab720\sa420\partightenfactor0

\fs33\fsmilli16800 \cf5 \cb3 [More Information Needed]\cb1 \
\pard\pardeftab720\partightenfactor0

\fs36 \cf4 \
\pard\pardeftab720\partightenfactor0

\fs37\fsmilli18560 \cf2 \cb3 Considerations for Using the Data\cb1 \
\pard\pardeftab720\partightenfactor0

\fs36 \cf4 \
\pard\pardeftab720\partightenfactor0

\fs35\fsmilli17600 \cf2 \cb3 Social Impact of Dataset\cb1 \
\pard\pardeftab720\sa420\partightenfactor0

\fs33\fsmilli16800 \cf5 \cb3 [More Information Needed]\cb1 \
\pard\pardeftab720\partightenfactor0

\fs36 \cf4 \
\pard\pardeftab720\partightenfactor0

\fs35\fsmilli17600 \cf2 \cb3 Discussion of Biases\cb1 \
\pard\pardeftab720\sa420\partightenfactor0

\fs33\fsmilli16800 \cf5 \cb3 [More Information Needed]\cb1 \
\pard\pardeftab720\partightenfactor0

\fs36 \cf4 \
\pard\pardeftab720\partightenfactor0

\fs35\fsmilli17600 \cf2 \cb3 Other Known Limitations\cb1 \
\pard\pardeftab720\sa420\partightenfactor0

\fs33\fsmilli16800 \cf5 \cb3 [More Information Needed]\cb1 \
\pard\pardeftab720\partightenfactor0

\fs36 \cf4 \
\pard\pardeftab720\partightenfactor0

\fs37\fsmilli18560 \cf2 \cb3 Additional Information\cb1 \
\pard\pardeftab720\partightenfactor0

\fs36 \cf4 \
\pard\pardeftab720\partightenfactor0

\fs35\fsmilli17600 \cf2 \cb3 Dataset Curators\cb1 \
\pard\pardeftab720\sa420\partightenfactor0

\fs33\fsmilli16800 \cf5 \cb3 Han Xiao and Kashif Rasul and Roland Vollgraf\cb1 \
\pard\pardeftab720\partightenfactor0

\fs36 \cf4 \
\pard\pardeftab720\partightenfactor0

\fs35\fsmilli17600 \cf2 \cb3 Licensing Information\cb1 \
\pard\pardeftab720\sa420\partightenfactor0

\fs33\fsmilli16800 \cf5 \cb3 MIT Licence\cb1 \
\pard\pardeftab720\partightenfactor0

\fs36 \cf4 \
\pard\pardeftab720\partightenfactor0

\fs35\fsmilli17600 \cf2 \cb3 Citation Information\cb1 \
\pard\pardeftab720\partightenfactor0

\fs28 \cf6 \cb8 @article\{DBLP:journals/corr/abs-1708-07747,\
  author    = \{Han Xiao and\
               Kashif Rasul and\
               Roland Vollgraf\},\
  title     = \{Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning\
               Algorithms\},\
  journal   = \{CoRR\},\
  volume    = \{abs/1708.07747\},\
  year      = \{2017\},\
  url       = \{http://arxiv.org/abs/1708.07747\},\
  archivePrefix = \{arXiv\},\
  eprint    = \{1708.07747\},\
  timestamp = \{Mon, 13 Aug 2018 16:47:27 +0200\},\
  biburl    = \{https://dblp.org/rec/bib/journals/corr/abs-1708-07747\},\
  bibsource = \{dblp computer science bibliography, https://dblp.org\}\
\}\
\pard\pardeftab720\qc\partightenfactor0

\fs33\fsmilli16800 \cf5 \cb1 \
\pard\pardeftab720\partightenfactor0

\fs36 \cf4 \
}
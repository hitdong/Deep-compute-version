---
title: "My Jumble of Computer Vision"
category: "Computer Vision"
---
fork from joshua19881228.github.io

I am going to maintain this page to record a few things about computer vision that I have read, am doing, or will have a look at. Previously I'd like to write short notes of the papers that I have read. It is a good way to remember and understand the ideas of the authors. But gradually I found that I forget much portion of what I had learnt because in addition to paper I also derive knowledges from others' blogs, online courses and reports, not recording them at all. Besides, I need a place to keep a list of what I should have a look at but do not at the time when I discover them. This page will be much like a catalog.

## Table of Content ##

* Content
{:toc}


## PAPERS AND PROJECTS

### OBJECT/SALIENCY DETECTOIN

* Chinese Text in the Wild ([PDF](https://arxiv.org/abs/1803.00085), [Project/Code](https://ctwdataset.github.io/))
* TSSD: Temporal Single-Shot Detector Based on Attention and LSTM for Robotic Intelligent Perception ([PDF](https://arxiv.org/abs/1803.00197))
* Tiny SSD: A Tiny Single-shot Detection Deep Convolutional Neural Network for Real-time Embedded Object Detection ([PDF](https://arxiv.org/abs/1802.06488), [Reading Note](https://joshua19881228.github.io/2018-03-04-Tiny_SSD/))
* Object Detection in Videos by Short and Long Range Object Linking ([PDF](https://arxiv.org/abs/1801.09823))
* Learning a Rotation Invariant Detector with Rotatable Bounding Box ([PDF](https://arxiv.org/abs/1711.09405), [Project/Code](https://github.com//liulei01/DRBox))
* Detecting Curve Text in the Wild: New Dataset and New Solution ([PDF](https://arxiv.org/abs/1712.02170), [Project/Code](https://github.com//Yuliang-Liu/Curve-Text-Detector#curve-text-detector))
* Single Shot Text Detector with Regional Attention ([PDF](https://arxiv.org/abs/1709.00138), [Project/Code](https://github.com//BestSonny/SSTD))
* Single-Shot Refinement Neural Network for Object Detection ([PDF](https://arxiv.org/abs/1711.06897), [Project/Code](https://github.com//sfzhang15/RefineDet), [Reading Note](https://joshua19881228.github.io/2018-01-10-Single-Shot-Refinement-Neural-Network/))
* $S^3$FD: Single Shot Scale-invariant Face Detector ([PDF](https://arxiv.org/abs/1708.05237), [Reading Note](https://joshua19881228.github.io/2018-01-20-S-3_Face_Detector/))
* MegDet: A Large Mini-Batch Object Detector ([PDF](https://arxiv.org/abs/1711.07240))
* Light-Head R-CNN: In Defense of Two-Stage Object Detector ([PDF](https://arxiv.org/abs/1711.07264))
* Interpretable R-CNN ([PDF](https://arxiv.org/abs/1711.05226))
* Cascade Region Proposal and Global Context for Deep Object Detection ([PDF](https://arxiv.org/abs/1710.10749))
* PVANET: Deep but Lightweight Neural Networks for Real-time Object Detection ([PDF](http://arxiv.org/abs/1608.08021), [Project/Code](https://github.com/sanghoon/pva-faster-rcnn), [Reading Note](https://joshua19881228.github.io/2016-08-30-Reading-note/))
* Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks ([PDF](http://arxiv.org/abs/1512.04143), [Reading Note](https://joshua19881228.github.io/2016-06-01-ION/))
* Object Detection from Video Tubelets with Convolutional Neural Networks ([PDF](http://arxiv.org/abs/1604.04053), [Reading Note](https://joshua19881228.github.io/2016-05-25-TCNN/))
* R-FCN: Object Detection via Region-based Fully Convolutional Networks ([PDF](http://arxiv.org/abs/1605.06409), [Project/Code](https://github.com/daijifeng001/r-fcn), [Reading Note](https://joshua19881228.github.io/2016-05-23-RFCN/))
* SSD: Single Shot MultiBox Detector ([PDF](http://arxiv.org/abs/1512.02325v2), [Project/Code](https://github.com/weiliu89/caffe/tree/ssd), [Reading Note](https://joshua19881228.github.io/2016-04-26-SSD/))
* Pushing the Limits of Deep CNNs for Pedestrian Detection ([PDF](http://lib-arxiv-008.serverfarm.cornell.edu/abs/1603.04525), [Reading Note](https://joshua19881228.github.io/2016-03-29-CFM/))
* Object Detection by Labeling Superpixels([PDF](http://www.cv-foundation.org/openaccess/content_cvpr_2015/ext/3B_072_ext.pdf), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-object-detection-by-labeling-superpixels_74/))
* Crafting GBD-Net for Object Detection ([PDF](https://arxiv.org/abs/1610.02579), [Projct/Code](https://github.com/craftGBD/craftGBD))
	code for CUImage and CUVideo, the object detection champion of ImageNet 2016.
* Fused DNN: A deep neural network fusion approach to fast and robust pedestrian detection ([PDF](https://arxiv.org/abs/1610.03466), [Reading Note](https://joshua19881228.github.io/2016-11-21-FusedDNN/))
* Training Region-based Object Detectors with Online Hard Example Mining ([PDF](https://arxiv.org/abs/1604.03540), [Reading Note](https://joshua19881228.github.io/2016-11-03-OHEM/))
* Detecting People in Artwork with CNNs ([PDF](https://arxiv.org/abs/1610.08871), [Project/Code](https://github.com/BathVisArtData/PeopleArt))
* Deeply supervised salient object detection with short connections ([PDF](https://arxiv.org/abs/1611.04849))
* Learning to detect and localize many objects from few examples ([PDF](https://arxiv.org/abs/1611.05664))
* Multi-Scale Saliency Detection using Dictionary Learning ([PDF](https://arxiv.org/abs/1611.06307))
* Straight to Shapes: Real-time Detection of Encoded Shapes ([PDF](https://arxiv.org/abs/1611.07932))
* Weakly Supervised Cascaded Convolutional Networks ([PDF](https://arxiv.org/abs/1611.08258), [Reading Note](https://joshua19881228.github.io/2016-12-06-WCCN/))
* Speed/accuracy trade-offs for modern convolutional object detectors ([PDF](https://arxiv.org/abs/1611.10012), [Reading Note](https://joshua19881228.github.io/2016-12-02-DetectorCompare/))
* Object Detection via End-to-End Integration of Aspect Ratio and Context Aware Part-based Models and Fully Convolutional Networks ([PDF](https://arxiv.org/abs/1612.00534))
* Feature Pyramid Networks for Object Detection ([PDF](https://arxiv.org/abs/1612.03144), [Reading Note](https://joshua19881228.github.io/2016-12-15-FPN/))
* COCO-Stuff: Thing and Stuff Classes in Context ([PDF](https://arxiv.org/abs/1612.03716))
* Finding Tiny Faces ([PDF](https://arxiv.org/abs/1612.04402v1))
* Beyond Skip Connections: Top-Down Modulation for Object Detection ([PDF](https://arxiv.org/abs/1612.06851), [Reading Note](https://joshua19881228.github.io/2016-12-21-TDM/))
* YOLO9000: Better, Faster, Stronger ([PDF](https://arxiv.org/abs/1612.08242), [Project/Code](http://pjreddie.com/darknet/yolo/), [Reading Note](https://joshua19881228.github.io/2017-01-11-YOLO9000/))
* Quantitative Analysis of Automatic Image Cropping Algorithms: A Dataset and Comparative Study ([PDF](https://arxiv.org/abs/1701.01480))
* To Boost or Not to Boost? On the Limits of Boosted Trees for Object Detection ([PDF](https://arxiv.org/abs/1701.01692))
* Pixel Objectness ([PDF](https://arxiv.org/abs/1701.05349), [Project/Code](http://vision.cs.utexas.edu/projects/pixelobjectness/), [Reading Note](https://joshua19881228.github.io/2017-01-22-PixelObjectness/))
* DSSD: Deconvolutional Single Shot Detector ([PDF](https://arxiv.org/abs/1701.06659), [Reading Note](https://joshua19881228.github.io/2017-02-10-DSSD/))
* A Fast and Compact Salient Score Regression Network Based on Fully Convolutional Network ([PDF](https://arxiv.org/abs/1702.00615))
* Wide-Residual-Inception Networks for Real-time Object Detection ([PDF](https://arxiv.org/abs/1702.01243))
* Zoom Out-and-In Network with Recursive Training for Object Proposal ([PDF](https://arxiv.org/abs/1702.05711), [Project/Code](https://github.com/hli2020/zoom_network))
* Improving Object Detection with Region Similarity Learning ([PDF](https://arxiv.org/abs/1703.00234))
* Tree-Structured Reinforcement Learning for Sequential Object Localization ([PDF](https://arxiv.org/abs/1703.02710))
* Weakly Supervised Object Localization Using Things and Stuff Transfer ([PDF](https://arxiv.org/abs/1703.08000))
* Unsupervised learning from video to detect foreground objects in single images ([PDF](https://arxiv.org/abs/1703.10901))
* A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection ([PDF](http://www.cs.cmu.edu/~xiaolonw/papers/CVPR2017_Adversarial_Det.pdf), [Project/Code](https://github.com/xiaolonw/adversarial-frcnn))
* A Learning non-maximum suppression ([PDF](https://arxiv.org/abs/1705.02950))
* Real Time Image Saliency for Black Box Classifiers ([PDF](https://arxiv.org/abs/1705.07857))
* An Efficient Approach for Object Detection and Tracking of Objects in a Video with Variable Background ([PDF](https://arxiv.org/abs/1706.02672))
* RON: Reverse Connection with Objectness Prior Networks for Object Detection ([PDF](https://arxiv.org/abs/1707.01691), [Project/Code](https://github.com/taokong/RON))
* Deformable Part-based Fully Convolutional Network for Object Detection ([PDF](https://arxiv.org/abs/1707.06175), [Reading Note](https://joshua19881228.github.io/2017-07-28-DP-FCN/))
* Recurrent Scale Approximation for Object Detection in CNN ([PDF](https://arxiv.org/abs/1707.09531))
* DSOD: Learning Deeply Supervised Object Detectors from Scratch ([PDF](https://arxiv.org/abs/1708.01241), [Project/Code](https://github.com/szq0214/DSOD), [Reading Note](https://joshua19881228.github.io/2017-08-10-DSOD/))
* PPR-FCN: Weakly Supervised Visual Relation Detection via Parallel Pairwise R-FCN ([PDF](https://arxiv.org/abs/1708.01956))
* Focal Loss for Dense Object Detection ([PDF](https://arxiv.org/abs/1708.02002))
* Learning Uncertain Convolutional Features for Accurate Saliency Detection ([PDF](https://arxiv.org/abs/1708.02031))
* Optimizing Region Selection for Weakly Supervised Object Detection ([PDF](https://arxiv.org/abs/1708.01723))
* Kill Two Birds With One Stone: Boosting Both Object Detection Accuracy and Speed With adaptive Patch-of-Interest Composition ([PDF](https://arxiv.org/abs/1708.03795))
* Flow-Guided Feature Aggregation for Video Object Detection ([PDF](https://arxiv.org/abs/1703.10025))
* BlitzNet: A Real-Time Deep Network for Scene Understanding ([PDF]( BlitzNet: A Real-Time Deep Network for Scene Understanding), [Project/Code](http://thoth.inrialpes.fr/research/blitznet/))
* RON: Reverse Connection with Objectness Prior Networks for Object Detection ([PDF](https://arxiv.org/abs/1707.01691))
* Soft Proposal Networks for Weakly Supervised Object Localization ([PDF](https://arxiv.org/pdf/1709.01829.pdf), [Project/Code](https://github.com/ZhouYanzhao/SPN/))
* Feature-Fused SSD: Fast Detection for Small Objects ([PDF](https://arxiv.org/abs/1709.05054))
* Light Cascaded Convolutional Neural Networks for Accurate Player Detection ([PDF](https://arxiv.org/abs/1709.10230))
* Personalized Saliency and its Prediction ([PDF](https://arxiv.org/abs/1710.03011))
* WeText: Scene Text Detection under Weak Supervision ([PDF](https://arxiv.org/abs/1710.04826))
* VPGNet: Vanishing Point Guided Network for Lane and Road Marking Detection and Recognition ([PDF](https://arxiv.org/abs/1710.06288), [Project/Code](https://github.com/SeokjuLee/VPGNet))

### SEGMENTATION/PARSING ###

* Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation ([PDF](https://arxiv.org/abs/1802.02611), [Project/Code](https://github.com/tensorflow/models/tree/master/research/deeplab))
* Learning random-walk label propagation for weakly-supervised semantic segmentation ([PDF](https://arxiv.org/abs/1802.00470))
* Panoptic Segmentation ([PDF](https://arxiv.org/abs/1801.00868), [Reading Note](https://joshua19881228.github.io/2018-01-08-Panoptic_Segmentation/))
* Learning to Segment Every Thing ([PDF](https://arxiv.org/abs/1711.10370))
* Deep Extreme Cut: From Extreme Points to Object Segmentation ([PDF](https://arxiv.org/abs/1711.09081))
* Instance-aware Semantic Segmentation via Multi-task Network Cascades ([PDF](http://arxiv.org/abs/1512.04412), [Project/Code](https://github.com/daijifeng001/MNC))
* ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation ([PDF](http://arxiv.org/abs/1606.02147), [Reading Note](https://joshua19881228.github.io/2016-08-23-Reading-note/))
* Learning Deconvolution Network for Semantic Segmentation ([PDF](http://arxiv.org/abs/1505.04366), [Reading Note](https://joshua19881228.github.io/2016-07-04-DeconvSemanticSegmentation/))
* Semantic Object Parsing with Graph LSTM ([PDF](http://arxiv.org/abs/1603.07063), [Reading Note](https://joshua19881228.github.io/2016-05-30-GLSTM/))
* Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding ([PDF](http://arxiv.org/abs/1511.02680), [Reading Note](https://joshua19881228.github.io/2015-11-10-SegNet/))
* Learning to Segment Moving Objects in Videos ([PDF](http://arxiv.org/abs/1412.6504), [Reading Note](https://joshua19881228.github.io/2015-11-04-Learning-to-Segment-Moving-Objects-in-Videos/))
* Deep Structured Features for Semantic Segmentation ([PDF](http://arxiv.org/abs/1609.07916v1))

    >We propose a highly structured neural network architecture for semantic segmentation of images that combines i) a Haar wavelet-based tree-like convolutional neural network (CNN), ii) a random layer realizing a radial basis function kernel approximation, and iii) a linear classifier. While stages i) and ii) are completely pre-specified, only the linear classifier is learned from data. Thanks to its high degree of structure, our architecture has a very small memory footprint and thus fits onto low-power embedded and mobile platforms. We apply the proposed architecture to outdoor scene and aerial image semantic segmentation and show that the accuracy of our architecture is competitive with conventional pixel classification CNNs. Furthermore, we demonstrate that the proposed architecture is data efficient in the sense of matching the accuracy of pixel classification CNNs when trained on a much smaller data set. 

* CNN-aware Binary Map for General Semantic Segmentation ([PDF](https://arxiv.org/abs/1609.09220))
* Learning to Refine Object Segments ([PDF](https://arxiv.org/abs/1603.08695))
* Clockwork Convnets for Video Semantic Segmentation([PDF](https://arxiv.org/abs/1608.03609), [Project/Code](https://github.com/shelhamer/clockwork-fcn))
* Convolutional Gated Recurrent Networks for Video Segmentation ([PDF](https://arxiv.org/abs/1611.05435))
* Efficient Convolutional Neural Network with Binary Quantization Layer ([PDF](https://arxiv.org/abs/1611.06764))
* One-Shot Video Object Segmentation ([PDF](https://arxiv.org/abs/1611.05198))
* Fully Convolutional Instance-aware Semantic Segmentation ([PDF](https://arxiv.org/abs/1611.07709), [Projcet/Code](https://github.com/daijifeng001/TA-FCN), [Reading Note](https://joshua19881228.github.io/2016-11-28-InstanceFCN/))
* Semantic Segmentation using Adversarial Networks ([PDF](https://arxiv.org/abs/1611.08408))
* Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes ([PDF](https://arxiv.org/abs/1611.08323))
* Deep Watershed Transform for Instance Segmentation ([PDF](https://arxiv.org/abs/1611.08303))
* InstanceCut: from Edges to Instances with MultiCut ([PDF](https://arxiv.org/abs/1611.08272))
* The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation ([PDF](https://arxiv.org/abs/1611.09326))
* Improving Fully Convolution Network for Semantic Segmentation ([PDF](https://arxiv.org/abs/1611.08986))
* Video Scene Parsing with Predictive Feature Learning ([PDF](https://arxiv.org/abs/1612.00119))
* Training Bit Fully Convolutional Network for Fast Semantic Segmentation ([PDF](https://arxiv.org/abs/1612.00212))
* Pyramid Scene Parsing Network ([PDF](https://arxiv.org/abs/1612.01105), [Reading Note](https://joshua19881228.github.io/2017-09-19-PSPNet/))
* Mining Pixels: Weakly Supervised Semantic Segmentation Using Image Labels ([PDF](https://arxiv.org/abs/1612.02101))
* FastMask: Segment Object Multi-scale Candidates in One Shot ([PDF](https://arxiv.org/abs/1612.08843), [Project/Code](https://github.com/voidrank/FastMask), [Reading Note](https://joshua19881228.github.io/2017-04-18-FastMask/))
* A New Convolutional Network-in-Network Structure and Its Applications in Skin Detection, Semantic Segmentation, and Artifact Reduction ([PDF](https://arxiv.org/abs/1701.06190), [Reading Note](https://joshua19881228.github.io/2017-01-24-NINApplication/))
* FusionSeg: Learning to combine motion and appearance for fully automatic segmention of generic objects in videos ([PDF](https://arxiv.org/abs/1701.05384))
* Visual Saliency Prediction Using a Mixture of Deep Neural Networks ([PDF](https://arxiv.org/abs/1702.00372))
* PixelNet: Representation of the pixels, by the pixels, and for the pixels ([PDF](https://arxiv.org/abs/1702.06506), [Project/Code](http://www.cs.cmu.edu/~aayushb/pixelNet/))
* Super-Trajectory for Video Segmentation ([PDF](https://arxiv.org/abs/1702.08634))
* Understanding Convolution for Semantic Segmentation ([PDF](https://arxiv.org/abs/1702.08502), [Reading Note](https://joshua19881228.github.io/2017-03-04-DUC-HDC/))
* Adversarial Examples for Semantic Image Segmentation ([PDF](https://arxiv.org/abs/1703.01101))
* Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network ([PDF](https://arxiv.org/abs/1703.02719))
* Deep Image Matting ([PDF](https://arxiv.org/abs/1703.03872), [Reading Note](https://joshua19881228.github.io/2017-03-16-Matting/))
* Mask R-CNN ([PDF](https://arxiv.org/abs/1703.06870), [Caffe Implementation](https://github.com/jasjeetIM/Mask-RCNN), [TuSimple Implementation on MXNet](https://github.com/TuSimple/mx-maskrcnn), [TensorFlow Implementation](https://github.com/matterport/Mask_RCNN), [Reading Note](https://joshua19881228.github.io/2017-05-02-MaskRCNN/))
* Predicting Deeper into the Future of Semantic Segmentation ([PDF](https://arxiv.org/abs/1703.07684))
* Convolutional Oriented Boundaries: From Image Segmentation to High-Level Tasks ([PDF](https://arxiv.org/abs/1701.04658), [Project/Code](http://www.vision.ee.ethz.ch/~cvlsegmentation/cob/))
* One-Shot Video Object Segmentation ([PDF](https://arxiv.org/abs/1611.05198), [Project/Code](https://github.com/kmaninis/OSVOS-caffe))
* Semantic Instance Segmentation via Deep Metric Learning ([PDF](https://arxiv.org/abs/1703.10277))
* Not All Pixels Are Equal: Difficulty-aware Semantic Segmentation via Deep Layer Cascade ([PDF](https://arxiv.org/abs/1704.01344))
* Semantically-Guided Video Object Segmentation ([PDF](https://arxiv.org/abs/1704.01926))
* Recurrent Multimodal Interaction for Referring Image Segmentation ([PDF](https://arxiv.org/abs/1703.07939))
* Loss Max-Pooling for Semantic Image Segmentation ([PDF](https://arxiv.org/abs/1704.02966)) 
* Reformulating Level Sets as Deep Recurrent Neural Network Approach to Semantic Segmentation ([PDF](https://arxiv.org/abs/1704.03593))
* Learning Video Object Segmentation with Visual Memory ([PDF](https://arxiv.org/abs/1704.05737))
* A Review on Deep Learning Techniques Applied to Semantic Segmentation ([PDF](https://arxiv.org/abs/1704.06857))
* BiSeg: Simultaneous Instance Segmentation and Semantic Segmentation with Fully Convolutional Networks ([PDF](https://arxiv.org/abs/1706.02135))
* Rethinking Atrous Convolution for Semantic Image Segmentation ([PDF](https://arxiv.org/abs/1706.05587))
* Discriminative Localization in CNNs for Weakly-Supervised Segmentation of Pulmonary Nodules ([PDF](https://arxiv.org/abs/1707.01086))
* Superpixel-based semantic segmentation trained by statistical process control ([PDF](https://arxiv.org/abs/1706.10071))
* The Devil is in the Decoder ([PDF](https://arxiv.org/abs/1707.05847))
* Semantic Segmentation with Reverse Attention ([PDF](https://arxiv.org/abs/1707.06426))
* Learning Deconvolution Network for Semantic Segmentation ([PDF](https://arxiv.org/abs/1505.04366), [Project/Code](https://github.com/HyeonwooNoh/DeconvNet))
* Depth Adaptive Deep Neural Network for Semantic Segmentation ([PDF](https://arxiv.org/abs/1708.01818))
* Semantic Instance Segmentation with a Discriminative Loss Function ([PDF](https://arxiv.org/abs/1708.02551))
* A Cost-Sensitive Visual Question-Answer Framework for Mining a Deep And-OR Object Semantics from Web Images ([PDF](https://arxiv.org/abs/1708.03911))
* ICNet for Real-Time Semantic Segmentation on High-Resolution Images ([PDF](https://arxiv.org/abs/1704.08545), [Project/Code](https://hszhao.github.io/projects/icnet/))
* Pyramid Scene Parsing Network ([PDF](https://arxiv.org/abs/1612.01105), [Project/Code](https://github.com/hszhao/PSPNet), [Reading Note](https://joshua19881228.github.io/2017-09-19-PSPNet/))
* Learning to Segment Instances in Videos with Spatial Propagation Network ([PDF](http://davischallenge.org/challenge2017/papers/DAVIS-Challenge-6th-Team.pdf), [Project/Code](https://github.com/JingchunCheng/Seg-with-SPN))
* Learning Affinity via Spatial Propagation Networks ([PDF](https://arxiv.org/abs/1710.01020), [Project/Code](https://github.com/Liusifei/caffe-spn))

### TRACKING ###

* Machine Learning Methods for Solving Assignment Problems in Multi-Target Tracking ([PDF](https://arxiv.org/abs/1802.06897))
* Multi-Target, Multi-Camera Tracking by Hierarchical Clustering: Recent Progress on DukeMTMC Project ([PDF](https://arxiv.org/abs/1712.09531))
* Detect-and-Track: Efficient Pose Estimation in Videos ([PDF](https://arxiv.org/abs/1712.09184))
* Track, then Decide: Category-Agnostic Vision-based Multi-Object Tracking ([PDF](https://arxiv.org/abs/1712.07920))
* Spatially Supervised Recurrent Convolutional Neural Networks for Visual Object Tracking ([PDF](http://arxiv.org/abs/1607.05781), [Reading Note](https://joshua19881228.github.io/2016-07-21-ROLO/))
* Joint Tracking and Segmentation of Multiple Targets ([PDF](http://milanton.de/files/cvpr2015/cvpr2015-ext-abstr-anton.pdf), [Reading Note](https://joshua19881228.github.io/2015-11-05-Joint-Tracking-and-Segmentation-of-Multiple-Targets/))
* Deep Tracking on the Move: Learning to Track the World from a Moving Vehicle using Recurrent Neural Networks ([PDF](https://arxiv.org/abs/1609.09365))
* Convolutional Regression for Visual Tracking ([PDF](https://arxiv.org/abs/1611.04215))
* Kernelized Correlation Filters([Project](http://www.robots.ox.ac.uk/%7Ejoao/circulant/) [CODE1](https://github.com/foolwood/KCF) [CODE2](https://github.com/vojirt/kcf))
* Online Visual Multi-Object Tracking via Labeled Random Finite Set Filtering ([PDF](https://arxiv.org/abs/1611.06011))
* SANet: Structure-Aware Network for Visual Tracking ([PDF](https://arxiv.org/abs/1611.06878))
* Semantic tracking: Single-target tracking with inter-supervised convolutional networks ([PDF](https://arxiv.org/abs/1611.06395))
* On The Stability of Video Detection and Tracking ([PDF](https://arxiv.org/abs/1611.06467))
* Dual Deep Network for Visual Tracking ([PDF](https://arxiv.org/abs/1612.06053))
* Deep Motion Features for Visual Tracking ([PDF](https://arxiv.org/abs/1612.06615))
* Robust and Real-time Deep Tracking Via Multi-Scale Domain Adaptation ([PDF](https://arxiv.org/abs/1701.00561), [Project/Code](https://bitbucket.org/xinke_wang/msdat/src))
* Instance Flow Based Online Multiple Object Tracking ([PDF](https://arxiv.org/abs/1703.01289))
* PathTrack: Fast Trajectory Annotation with Path Supervision ([PDF](https://arxiv.org/abs/1703.02437))
* Good Features to Correlate for Visual Tracking ([PDF](https://arxiv.org/abs/1704.06326))
* Re3 : Real-Time Recurrent Regression Networks for Object Tracking ([PDF](https://arxiv.org/abs/1705.06368))
* Action-Decision Networks for Visual Tracking with Deep Reinforcement Learning ([PDF](https://drive.google.com/file/d/0B34VXh5mZ22cZUs2Umc1cjlBMFU/view), [Project/Code](https://sites.google.com/view/cvpr2017-adnet))
* Simple Online and Realtime Tracking with a Deep Association Metric ([PDF](https://arxiv.org/abs/1703.07402))
* Learning Policies for Adaptive Tracking with Deep Feature Cascades ([PDF](https://arxiv.org/abs/1708.02973))
* Recurrent Filter Learning for Visual Tracking ([PDF](https://arxiv.org/abs/1708.03874))
* Tracking Persons-of-Interest via Unsupervised Representation Adaptation  ([PDF](https://arxiv.org/abs/1710.02139))
* Detect to Track and Track to Detect ([PDF](https://arxiv.org/abs/1710.03958), [Project/Code](https://www.robots.ox.ac.uk/~vgg/research/detect-track/), [Reading Note](https://joshua19881228.github.io/2017-10-25-D2TT2D/))

### POSE ESTIMATION ###

* DensePose: Dense Human Pose Estimation In The Wild ([PDF](https://arxiv.org/abs/1802.00434), [Project/Code](http://densepose.org/))
* Cascaded Pyramid Network for Multi-Person Pose Estimation ([PDF](https://arxiv.org/abs/1711.07319))
* Chained Predictions Using Convolutional Neural Networks ([PDF](http://arxiv.org/abs/1605.02346), [Reading Note](https://joshua19881228.github.io/2016-05-24-chainedPredictions/))
* CRF-CNN: Modeling Structured Information in Human Pose Estimation ([PDF](https://arxiv.org/abs/1611.00468))
* Convolutional Pose Machines ([PDF](https://arxiv.org/abs/1602.00134), [Project/Code](https://github.com/shihenw/convolutional-pose-machines-release), [Reading Note](https://joshua19881228.github.io/2016-12-27-CPM/))
* Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields ([PDF](https://arxiv.org/abs/1611.08050), [Project/Code](https://github.com/ZheC/Multi-Person-Pose-Estimation), [Reading Note](https://joshua19881228.github.io/2016-12-29-PAF/))
* Towards Accurate Multi-person Pose Estimation in the Wild ([PDF](https://arxiv.org/abs/1701.01779), [Reading Note](https://joshua19881228.github.io/2017-01-19-TAMPEW/))
* Adversarial PoseNet: A Structure-aware Convolutional Network for Human Pose Estimation ([PDF](https://arxiv.org/abs/1705.00389))
* Coarse-to-Fine Volumetric Prediction for Single-Image 3D Human Pose ([PDF](https://arxiv.org/abs/1611.07828), [Project/Code](https://www.seas.upenn.edu/~pavlakos/projects/volumetric/))
* Learning Feature Pyramids for Human Pose Estimation ([PDF](https://arxiv.org/abs/1708.01101), [Project/Code](https://github.com/bearpaw/PyraNet))
* Joint Multi-Person Pose Estimation and Semantic Part Segmentation ([PDF](https://arxiv.org/abs/1708.03383))
* DeepPrior++: Improving Fast and Accurate 3D Hand Pose Estimation ([PDF](https://arxiv.org/abs/1708.08325))
* Lifting from the Deep: Convolutional 3D Pose Estimation from a Single Image ([PDF](https://arxiv.org/abs/1701.00295))
* Human Pose Regression by Combining Indirect Part Detection and Contextual Information ([PDF](https://arxiv.org/abs/1710.02322))
* Dual Path Networks for Multi-Person Human Pose Estimation ([PDF](https://arxiv.org/abs/1710.10192))

### ACTION RECOGNITION/EVENT DETECTION/VIDEO ###

* 2D/3D Pose Estimation and Action Recognition using Multitask Deep Learning ([PDF](https://arxiv.org/abs/1802.09232))
* Real-Time End-to-End Action Detection with Two-Stream Networks ([PDF](https://arxiv.org/abs/1802.08362))
* Learning Video-Story Composition via Recurrent Neural Network ([PDF](https://arxiv.org/abs/1801.10281))
* Real-world Anomaly Detection in Surveillance Videos ([PDF](https://arxiv.org/abs/1801.04264))
* Fully-Coupled Two-Stream Spatiotemporal Networks for Extremely Low Resolution Action Recognition ([PDF](https://arxiv.org/abs/1801.03983))
* Deep Reinforcement Learning for Unsupervised Video Summarization with Diversity-Representativeness Reward ([PDF](https://arxiv.org/abs/1801.00054), [Project/Code](https://github.com//KaiyangZhou/vsumm-reinforce))
* Making a long story short: A Multi-Importance Semantic for Fast-Forwarding Egocentric Videos ([PDF](https://arxiv.org/abs/1711.03473))
* Attentional Pooling for Action Recognition ([PDF](https://arxiv.org/abs/1711.01467), [Project/Code](https://github.com/rohitgirdhar/AttentionalPoolingAction))
* Pooling the Convolutional Layers in Deep ConvNets for Action Recognition ([PDF](http://arxiv.org/abs/1511.02126), [Reading Note](https://joshua19881228.github.io/2015-11-15-PoolConvNet/))
* Two-Stream Convolutional Networks for Action Recognition in Videos ([PDF](http://arxiv.org/abs/1406.2199), [Reading Note](https://joshua19881228.github.io/2015-11-06-Two-Stream/))
* YouTube-8M: A Large-Scale Video Classification Benchmark ([PDF](https://arxiv.org/abs/1609.08675), [Project/Code](http://research.google.com/youtube8m))
* Spatiotemporal Residual Networks for Video Action Recognition ([PDF](https://arxiv.org/abs/1611.02155))
* An End-to-End Spatio-Temporal Attention Model for Human Action Recognition from Skeleton Data ([PDF](https://arxiv.org/abs/1611.06067))
* Fast Video Classification via Adaptive Cascading of Deep Models ([PDF](https://arxiv.org/abs/1611.06453))
* Video Pixel Networks ([PDF](https://arxiv.org/abs/1610.00527))
* Plug-and-Play CNN for Crowd Motion Analysis: An Application in Abnormal Event Detection ([PDF](https://arxiv.org/abs/1610.00307))
* EM-Based Mixture Models Applied to Video Event Detection ([PDF](https://arxiv.org/abs/1610.02923))
* Video Captioning and Retrieval Models with Semantic Attention ([PDF](https://arxiv.org/abs/1610.02947))
* Title Generation for User Generated Videos ([PDF](https://arxiv.org/abs/1608.07068))
* Review of Action Recognition and Detection Methods ([PDF](https://arxiv.org/abs/1610.06906))
* RECURRENT MIXTURE DENSITY NETWORK FOR SPATIOTEMPORAL VISUAL ATTENTION ([PDF](http://openreview.net/pdf?id=SJRpRfKxx))
* Self-Supervised Video Representation Learning With Odd-One-Out Networks ([PDF](https://arxiv.org/abs/1611.06646))
* Recurrent Memory Addressing for describing videos ([PDF](https://arxiv.org/abs/1611.06492))
* Online Real time Multiple Spatiotemporal Action Localisation and Prediction on a Single Platform ([PDF](https://arxiv.org/abs/1611.08563))
* Real-Time Video Highlights for Yahoo Esports ([PDF](https://arxiv.org/abs/1611.08780))
* Surveillance Video Parsing with Single Frame Supervision ([PDF](https://arxiv.org/abs/1611.09587))
* Anomaly Detection in Video Using Predictive Convolutional Long Short-Term Memory Networks ([PDF](https://arxiv.org/abs/1612.00390))
* Action Recognition with Dynamic Image Networks ([PDF](https://arxiv.org/abs/1612.00738))
* ActionFlowNet: Learning Motion Representation for Action Recognition ([PDF](https://arxiv.org/abs/1612.03052))
* Video Propagation Networks ([PDF](https://arxiv.org/abs/1612.05478))
* Detecting events and key actors in multi-person videos ([PDF](https://arxiv.org/abs/1511.02917))
* A Pursuit of Temporal Accuracy in General Activity Detection ([PDF](https://arxiv.org/abs/1703.02716), [Reading Note](https://joshua19881228.github.io/2017-03-15-TAG/))
* Tube Convolutional Neural Network (T-CNN) for Action Detection in Videos ([PDF](https://arxiv.org/abs/1703.10664))
* Deceiving Google's Cloud Video Intelligence API Built for Summarizing Videos ([PDF](https://arxiv.org/abs/1703.09793))
* Incremental Tube Construction for Human Action Detection ([PDF](https://arxiv.org/abs/1704.01358))
* Unsupervised Action Proposal Ranking through Proposal Recombination ([PDF](https://arxiv.org/abs/1704.00758))
* CERN: Confidence-Energy Recurrent Network for Group Activity Recognition ([PDF](https://arxiv.org/abs/1704.03058))
* Forecasting Human Dynamics from Static Images ([PDF](https://arxiv.org/abs/1704.03432))
* Interpretable 3D Human Action Analysis with Temporal Convolutional Networks ([PDF](https://arxiv.org/abs/1704.04516))
* Training object class detectors with click supervision ([PDF](https://arxiv.org/abs/1704.06228))
* Skeleton-based Action Recognition with Convolutional Neural Networks ([PDF](https://arxiv.org/abs/1704.07595))
* Online growing neural gas for anomaly detection in changing surveillance scenes ([PDF](https://www.researchgate.net/publication/309101586_Online_growing_neural_gas_for_anomaly_detection_in_changing_surveillance_scenes))
* Learning Person Trajectory Representations for Team Activity Analysis ([PDF](https://arxiv.org/abs/1706.00893))
* Concurrence-Aware Long Short-Term Sub-Memories for Person-Person Action Recognition ([PDF](https://arxiv.org/abs/1706.00931))
* Video Imagination from a Single Image with Transformation Generation ([PDF](https://arxiv.org/abs/1706.04124), [Project/Code](https://github.com/gitpub327/VideoImagination))
* Optimizing Deep CNN-Based Queries over Video Streams at Scale ([PDF](https://arxiv.org/abs/1703.02529), [Project/Code](https://github.com/stanford-futuredata/noscope), [Reading Note](https://joshua19881228.github.io/2017-07-10-NoScope/))
* Extreme Low Resolution Activity Recognition with Multi-Siamese Embedding Learning ([PDF](https://arxiv.org/abs/1708.00999))
* Predicting Human Activities Using Stochastic Grammar ([PDF](https://arxiv.org/abs/1708.00945))
* Discriminative convolutional Fisher vector network for action recognition ([PDF](https://arxiv.org/abs/1707.06119))
* Extreme Low Resolution Activity Recognition with Multi-Siamese Embedding Learning ([PDF](https://arxiv.org/abs/1708.00999))
* Exploiting Semantic Contextualization for Interpretation of Human Activity in Videos ([PDF](https://arxiv.org/abs/1708.03725))
* Lattice Long Short-Term Memory for Human Action Recognition ([PDF](https://arxiv.org/abs/1708.03958))
* Kinship Verification from Videos using Spatio-Temporal Texture Features and Deep Learning ([PDF](https://arxiv.org/abs/1708.04069))
* Fast-Forward Video Based on Semantic Extraction ([PDF](https://arxiv.org/abs/1708.04160))
* Emotion Detection on TV Show Transcripts with Sequence-based Convolutional Neural Networks ([PDF](https://arxiv.org/abs/1708.04299))
* ConvNet Architecture Search for Spatiotemporal Feature Learning ([PDF](https://arxiv.org/abs/1708.05038), [Project/Code](http://vlg.cs.dartmouth.edu/c3d/), [Github](https://github.com/facebook/C3D))
* Fully Context-Aware Video Prediction ([PDF](https://arxiv.org/abs/1710.08518))

### FACE ###

* Detecting and counting tiny faces ([PDF](https://arxiv.org/abs/1801.06504), [Project/Code](https://github.com/alexattia/ExtendedTinyFaces))
* Training Deep Face Recognition Systems with Synthetic Data ([PDF](https://arxiv.org/abs/1802.05891))
* Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification ([PDF](http://proceedings.mlr.press/v81/buolamwini18a/buolamwini18a.pdf), [Project/Code](http://gendershades.org/))
* Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks ([PDF](https://arxiv.org/abs/1604.02878), [Project/Code](https://github.com/kpzhang93/MTCNN_face_detection_alignment), [Code Caffe](https://github.com/foreverYoungGitHub/MTCNN))
* Deep Architectures for Face Attributes ([PDF](http://arxiv.org/abs/1609.09018))
* Face Detection with End-to-End Integration of a ConvNet and a 3D Model ([PDF](https://www.arxiv.org/abs/1606.00850), [Reading Note](https://joshua19881228.github.io/2016-09-29-Reading-note/), [Project/Code](https://github.com/tfwu/FaceDetection-ConvNet-3D))
* A CNN Cascade for Landmark Guided Semantic Part Segmentation ([PDF](https://arxiv.org/pdf/1609.09642.pdf), [Project/Code](http://www.cs.nott.ac.uk/~psxasj/papers/jackson2016guided/))
* Kernel Selection using Multiple Kernel Learning and Domain Adaptation in Reproducing Kernel Hilbert Space, for Face Recognition under Surveillance Scenario ([PDF](https://arxiv.org/abs/1610.00660))
* An All-In-One Convolutional Neural Network for Face Analysis ([PDF](https://arxiv.org/abs/1611.00851))
* Fast Face-swap Using Convolutional Neural Networks ([PDF](https://arxiv.org/abs/1611.09577))
* Cross-Age Reference Coding for Age-Invariant Face Recognition and Retrieval ([Project/Code](http://bcsiriuschen.github.io/CARC/))
* CMS-RCNN: Contextual Multi-Scale Region-based CNN for Unconstrained Face Detection ([Project/Code](https://arxiv.org/abs/1606.05413))
* Face Synthesis from Facial Identity Features ([PDF](https://arxiv.org/abs/1701.04851))
* DeepFace: Face Generation using Deep Learning ([PDF](https://arxiv.org/abs/1701.01876))
* Emotion Recognition in the Wild via Convolutional Neural Networks and Mapped Binary Patterns ([PDF](http://www.openu.ac.il/home/hassner/projects/cnn_emotions/LeviHassnerICMI15.pdf), [Project/Code](http://www.openu.ac.il/home/hassner/projects/cnn_emotions/))
* EmotioNet Challenge: Recognition of facial expressions of emotion in the wild ([PDF](https://arxiv.org/abs/1703.01210))
* Unrestricted Facial Geometry Reconstruction Using Image-to-Image Translation ([PDF](https://arxiv.org/abs/1703.10131))
* Semi and Weakly Supervised Semantic Segmentation Using Generative Adversarial Network ([PDF](https://arxiv.org/abs/1703.09695))
* Deep Alignment Network: A convolutional neural network for robust face alignment ([PDF](https://arxiv.org/abs/1706.01789), [Project/Code](https://github.com/MarekKowalski/DeepAlignmentNetwork))
* Scale-Aware Face Detection ([PDF](https://arxiv.org/abs/1706.09876))
* SSH: Single Stage Headless Face Detector ([PDF](https://arxiv.org/abs/1708.03979))
* AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild ([PDF](https://arxiv.org/abs/1708.03985))
* SphereFace: Deep Hypersphere Embedding for Face Recognition ([PDF](https://arxiv.org/abs/1704.08063), [Project/Code](https://github.com/wy1iu/sphereface))
* Age Group and Gender Estimation in the Wild with Deep RoR Architecture ([PDF](https://arxiv.org/abs/1710.02985))
* Island Loss for Learning Discriminative Features in Facial Expression Recognition ([PDF](https://arxiv.org/abs/1710.03144))
* Temporal Non-Volume Preserving Approach to Facial Age-Progression and Age-Invariant Face Recognition ([PDF](https://arxiv.org/abs/1703.08617))

### OPTICAL FLOW ###

* DeepFlow: Large displacement optical flow with deep matching ([PDF](https://hal.inria.fr/hal-00873592), [Project/Code](http://lear.inrialpes.fr/src/deepflow/))
* Guided Optical Flow Learning ([PDF](https://arxiv.org/abs/1702.02295))

### IMAGE PROCESSING ###

* Neural Aesthetic Image Reviewer ([PDF](https://arxiv.org/abs/1802.10240), [Reading Note](https://joshua19881228.github.io/2018-03-05-Neural_Aesthetic_Image_Reviewer/))
* Automatic Image Cropping for Visual Aesthetic Enhancement Using Deep Neural Networks and Cascaded Regression ([PDF](https://arxiv.org/abs/1712.09048))
* Learning Intelligent Dialogs for Bounding Box Annotation ([PDF](https://arxiv.org/abs/1712.08087))
* Real-time video stabilization and mosaicking for monitoring and surveillance ([PDF](http://ieeexplore.ieee.org/document/7886813/?reload=true), [Project/Code](https://github.com//a-jahani/Real-time-Video-Mosaic))
* Learning Recursive Filter for Low-Level Vision via a Hybrid Neural Network ([PDF](http://faculty.ucmerced.edu/mhyang/papers/eccv16_rnn_filter.pdf), [Project/Code](https://github.com/Liusifei/caffe-lowlevel))
* Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding([PDF](https://arxiv.org/abs/1510.00149), [Project/Code](https://github.com/songhan/Deep-Compression-AlexNet))
* A Learned Representation For Artistic Style([PDF](https://arxiv.org/abs/1610.07629))
* Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification ([PDF](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/data/colorization_sig2016.pdf), [Project/Code](https://github.com/satoshiiizuka/siggraph2016_colorization))
* Pixel Recurrent Neural Networks ([PDF](https://arxiv.org/abs/1601.06759))
* Conditional Image Generation with PixelCNN Decoders ([PDF](https://arxiv.org/abs/1606.05328), [Project/Code](https://github.com/dritchie/pixelCNN))
* RAISR: Rapid and Accurate Image Super Resolution ([PDF](https://arxiv.org/abs/1606.01299))
* Photo-Quality Evaluation based on Computational Aesthetics: Review of Feature Extraction Techniques ([PDF](https://arxiv.org/abs/1612.06259))
* Fast color transfer from multiple images ([PDF](https://arxiv.org/abs/1612.08927))
* Bringing Impressionism to Life with Neural Style Transfer in Come Swim ([PDF](https://arxiv.org/abs/1701.04928))
* PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications ([PDF](https://arxiv.org/abs/1701.05517), (Project/CODE)[https://github.com/openai/pixel-cnn])
* Deep Photo Style Transfer ([PDF](https://arxiv.org/abs/1703.07511))
* A Neural Representation of Sketch Drawings ([PDF](https://arxiv.org/abs/1704.03477))
* Visual Attribute Transfer through Deep Image Analogy ([PDF](https://arxiv.org/abs/1705.01088))
* Deep Semantics-Aware Photo Adjustment ([PDF](https://arxiv.org/abs/1706.08260))
* Diversified Texture Synthesis with Feed-forward Networks ([PDF](https://arxiv.org/abs/1703.01664), [Project/Code](https://github.com/Yijunmaverick/MultiTextureSynthesis))
* Real-Time Neural Style Transfer for Videos ([PDF](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Real-Time_Neural_Style_CVPR_2017_paper.pdf))
* Creatism: A deep-learning photographer capable of creating professional work ([PDF](https://arxiv.org/abs/1707.03491))
* Deep Image Harmonization ([PDF](https://arxiv.org/abs/1703.00069), [Project/Code](https://github.com/wasidennis/DeepHarmonization))
* Neural Color Transfer between Images ([PDF](https://arxiv.org/abs/1710.00756))
* Deeper, Broader and Artier Domain Generalization ([PDF](https://arxiv.org/abs/1710.03077))

### CNN AND DEEP LEARNING ###

* Understanding the Loss Surface of Neural Networks for Binary Classification ([PDF](https://arxiv.org/abs/1803.00909))
* Tell Me Where to Look: Guided Attention Inference Network ([PDF](https://arxiv.org/abs/1802.10171))
* Convolutional Neural Networks with Alternately Updated Clique ([PDF](https://arxiv.org/abs/1802.10419), [Project/Code](https://github.com/iboing/CliqueNet))
* Visual Interpretability for Deep Learning: a Survey ([PDF](https://arxiv.org/abs/1802.00614))
* Threat of Adversarial Attacks on Deep Learning in Computer Vision: A Survey ([PDF](https://arxiv.org/abs/1801.00553))
* CNNs are Globally Optimal Given Multi-Layer Support ([PDF](https://arxiv.org/abs/1712.02501))
* Take it in your stride: Do we need striding in CNNs? ([PDF](https://arxiv.org/abs/1712.02502))
* Gradients explode - Deep Networks are shallow - ResNet explained ([PDF](https://arxiv.org/abs/1712.05577))
* Super-Convergence: Very Fast Training of Residual Networks Using Large Learning Rates ([PDF](https://arxiv.org/abs/1708.07120), [Project/Code](https://github.com/lnsmith54/super-convergence))
* Data Distillation: Towards Omni-Supervised Learning ([PDF](https://arxiv.org/abs/1712.04440))
* Peephole: Predicting Network Performance Before Training ([PDF](https://arxiv.org/abs/1712.03351))
* AdaBatch: Adaptive Batch Sizes for Training Deep Neural Networks ([PDF](https://arxiv.org/abs/1712.02029))
* Gradual Tuning: a better way of Fine Tuning the parameters of a Deep Neural Network ([PDF](https://arxiv.org/abs/1711.10177))
* CondenseNet: An Efficient DenseNet using Learned Group Convolutions ([PDF](http://www.cs.cornell.edu/~gaohuang/papers/condensenet.pdf), [Project/Code](https://github.com//ShichenLiu/CondenseNet))
* Population Based Training of Neural Networks ([PDF](https://arxiv.org/abs/1711.09846))
* Knowledge Concentration: Learning 100K Object Classifiers in a Single CNN ([PDF](https://arxiv.org/abs/1711.07607))
* Shift: A Zero FLOP, Zero Parameter Alternative to Spatial Convolutions ([PDF](https://arxiv.org/abs/1711.08141))
* Unleashing the Potential of CNNs for Interpretable Few-Shot Learning ([PDF](https://arxiv.org/abs/1711.08277))
* Non-local Neural Networks ([PDF](https://arxiv.org/abs/1711.07971))
* Log-DenseNet: How to Sparsify a DenseNet ([PDF](https://arxiv.org/abs/1711.00002g))
* Don't Decay the Learning Rate, Increase the Batch Size ([PDF](https://arxiv.org/abs/1711.00489))
* Guarding Against Adversarial Domain Shifts with Counterfactual Regularization ([PDF](https://arxiv.org/abs/1710.11469))
* UberNet: Training a 'Universal' Convolutional Neural Network for Low-, Mid-, and High-Level Vision using Diverse Datasets and Limited Memory ([PDF](http://arxiv.org/abs/1609.02132), [Project/Code](http://cvn.ecp.fr/ubernet/))
* What makes ImageNet good for transfer learning? ([PDF](http://arxiv.org/abs/1608.08614), [Project/Code](http://minyounghuh.com/papers/analysis/), [Reading Note](https://joshua19881228.github.io/2016-09-06-Reading-note/))

    >The tremendous success of features learnt using the ImageNet classification task on a wide range of transfer tasks begs the question: what are the intrinsic properties of the ImageNet dataset that are critical for learning good, general-purpose features? This work provides an empirical investigation of various facets of this question: Is more pre-training data always better? How does feature quality depend on the number of training examples per class? Does adding more object classes improve performance? For the same data budget, how should the data be split into classes? Is fine-grained recognition necessary for learning good features? Given the same number of training classes, is it better to have coarse classes or fine-grained classes? Which is better: more classes or more examples per class?

* Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units ([PDF](http://arxiv.org/abs/1603.05201))
* Densely Connected Convolutional Networks ([PDF](http://arxiv.org/abs/1608.06993), [Project/Code](https://github.com/liuzhuang13/DenseNet), [Reading Note](https://joshua19881228.github.io/2016-11-24-DenseNet/))
* Decoupled Neural Interfaces using Synthetic Gradients ([PDF](https://arxiv.org/pdf/1608.05343.pdf))

    >Training directed neural networks typically requires forward-propagating data through a computation graph, followed by backpropagating error signal, to produce weight updates. All layers, or more generally, modules, of the network are therefore locked, in the sense that they must wait for the remainder of the network to execute forwards and propagate error backwards before they can be updated. In this work we break this constraint by decoupling modules by introducing a model of the future computation of the network graph.  These models predict what the result of the modeled sub-graph will produce using only local information. In particular we focus on modeling error gradients: by using the modeled synthetic gradient in place of true backpropagated error gradients we decouple subgraphs, and can update them independently and asynchronously.

* Rethinking the Inception Architecture for Computer Vision ([PDF](http://arxiv.org/abs/1512.00567), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-rethinking-the-inception-architecture-for-computer-vision_136/))

    In this paper, several network designing choices are discussed, including *factorizing convolutions into smaller kernels and asymmetric kernels*, *utility of auxiliary classifiers* and *reducing grid size using convolution stride rather than pooling*.

* Factorized Convolutional Neural Networks ([PDF](http://arxiv.org/abs/1608.04337), [Reading Note](https://joshua19881228.github.io/2016-08-19-Reading-note/))
* Do semantic parts emerge in Convolutional Neural Networks? ([PDF](http://arxiv.org/abs/1607.03738), [Reading Note](https://joshua19881228.github.io/2016-08-09-Reading-Note/))
* A Critical Review of Recurrent Neural Networks for Sequence Learning ([PDF](https://arxiv.org/abs/1506.00019))
* Image Compression with Neural Networks ([Project/Code](https://github.com/tensorflow/models/tree/master/compression))
* Graph Convolutional Networks ([Project/Code](http://tkipf.github.io/graph-convolutional-networks/))
* Understanding intermediate layers using linear classifier probes ([PDF](https://arxiv.org/abs/1610.01644), [Reading Note](https://joshua19881228.github.io/2016-10-28-linear-classifier-probe/))
* Learning What and Where to Draw ([PDF](http://www.scottreed.info/files/nips2016.pdf), [Project/Code](https://github.com/reedscot/nips2016))
* On the interplay of network structure and gradient convergence in deep  learning ([PDF](https://arxiv.org/abs/1511.05297))
* Deep Learning with Separable Convolutions ([PDF](https://arxiv.org/abs/1610.02357))
* Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization ([PDF](https://arxiv.org/abs/1610.02391), [Project/Code](https://github.com/ramprs/grad-cam/))
* Optimization of Convolutional Neural Network using Microcanonical Annealing Algorithm ([PDF](https://arxiv.org/abs/1610.02306))
* Deep Pyramidal Residual Networks ([PDF](https://arxiv.org/abs/1610.02915))
* Impatient DNNs - Deep Neural Networks with Dynamic Time Budgets ([PDF](https://arxiv.org/abs/1610.02850))
* Uncertainty in Deep Learning ([PDF](http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf), [Project/Code](http://mlg.eng.cam.ac.uk/yarin/blog_2248.html))
	This is the PhD Thesis of Yarin Gal.
* Tensorial Mixture Models ([PDF](https://arxiv.org/abs/1610.04167), [Project/Code](https://github.com/HUJI-Deep/TMM))
* Multifaceted Feature Visualization: Uncovering the Different Types of Features Learned By Each Neuron in Deep Neural Networks ([PDF](https://arxiv.org/abs/1602.03616))
* Why Deep Neural Networks? ([PDF](https://arxiv.org/abs/1610.04161))
* Local Similarity-Aware Deep Feature Embedding ([PDF](https://arxiv.org/abs/1610.08904))
* A Review of 40 Years of Cognitive Architecture Research: Focus on Perception, Attention, Learning and Applications ([PDF](https://arxiv.org/abs/1610.08602))
* Professor Forcing: A New Algorithm for Training Recurrent Networks ([PDF](https://arxiv.org/abs/1610.09038))
* On the expressive power of deep neural networks([PDF](https://arxiv.org/abs/1606.05336))
* What Is the Best Practice for CNNs Applied to Visual Instance Retrieval? ([PDF](https://arxiv.org/abs/1611.01640))
* Deep Convolutional Neural Network Design Patterns ([PDF](https://arxiv.org/abs/1611.00847), [Project/Code](https://github.com/iPhysicist/CNNDesignPatterns))
* Tricks from Deep Learning ([PDF](https://arxiv.org/abs/1611.03777))
* A Connection between Generative Adversarial Networks, Inverse Reinforcement Learning, and Energy-Based Models ([PDF](https://arxiv.org/abs/1611.03852))
* Multi-Shot Mining Semantic Part Concepts in CNNs ([PDF](https://arxiv.org/abs/1611.04246))
* Aggregated Residual Transformations for Deep Neural Networks ([PDF](https://arxiv.org/abs/1611.05431), [Reading Note](https://joshua19881228.github.io/2016-11-17-ResNeXt/))
* PolyNet: A Pursuit of Structural Diversity in Very Deep Networks ([PDF](https://arxiv.org/abs/1611.05725))
* On the Exploration of Convolutional Fusion Networks for Visual Recognition ([PDF](https://arxiv.org/abs/1611.05503))
* ResFeats: Residual Network Based Features for Image Classification ([PDF](https://arxiv.org/abs/1611.06656))
* Object Recognition with and without Objects ([PDF](https://arxiv.org/abs/1611.06596))
* LCNN: Lookup-based Convolutional Neural Network ([PDF](https://arxiv.org/abs/1611.06473), [Reading Note](https://joshua19881228.github.io/2016-11-23-LCNN/))
* Inductive Bias of Deep Convolutional Networks through Pooling Geometry ([PDF](https://arxiv.org/abs/1605.06743), [Project/Code](https://github.com/HUJI-Deep/inductive-pooling))
* Wider or Deeper: Revisiting the ResNet Model for Visual Recognition ([PDF](https://arxiv.org/abs/1611.10080), [Reading Note](https://joshua19881228.github.io/2016-12-07-WiderResNet/))
* Multi-Scale Context Aggregation by Dilated Convolutions ([PDF](https://arxiv.org/abs/1511.07122), [Project/Code](https://github.com/fyu/dilation))
* Large-Margin Softmax Loss for Convolutional Neural Networks ([PDF](https://arxiv.org/abs/1612.02295), [mxnet Code](https://github.com/luoyetx/mx-lsoftmax), [Caffe Code](https://github.com/wy1iu/LargeMargin_Softmax_Loss))
* Adversarial Examples Detection in Deep Networks with Convolutional Filter Statistics ([PDF](https://arxiv.org/abs/1612.07767))
* Feedback Networks ([PDF](https://arxiv.org/abs/1612.09508))
* Visualizing Residual Networks ([PDF](https://arxiv.org/abs/1701.02362))
* Convolutional Oriented Boundaries: From Image Segmentation to High-Level Tasks ([PDF](https://arxiv.org/abs/1701.04658), [Project/Code](http://www.vision.ee.ethz.ch/~cvlsegmentation/cob/))
* Understanding trained CNNs by indexing neuron selectivity ([PDF](https://arxiv.org/abs/1702.00382))
* Benchmarking State-of-the-Art Deep Learning Software Tools ([PDF](https://arxiv.org/abs/1608.07249), [Project/Code](http://dlbench.comp.hkbu.edu.hk/))
* Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models ([PDF](https://arxiv.org/abs/1702.03275))
* Visualizing Deep Neural Network Decisions: Prediction Difference Analysis ([PDF](https://arxiv.org/abs/1702.04595), [Project/Code](https://github.com/lmzintgraf/DeepVis-PredDiff))
* ShaResNet: reducing residual network parameter number by sharing weights ([PDF](https://arxiv.org/abs/1702.08782))
* Deep Forest: Towards An Alternative to Deep Neural Networks ([PDF](https://arxiv.org/abs/1702.08835), [Project/Code](https://github.com/kingfengji/gcForest))
* All You Need is Beyond a Good Init: Exploring Better Solution for Training Extremely Deep Convolutional Neural Networks with Orthonormality and Modulation ([PDF](https://arxiv.org/abs/1703.01827))
* Genetic CNN ([PDF](https://arxiv.org/abs/1703.01513))
* Deformable Convolutional Networks ([PDF](https://arxiv.org/abs/1703.06211))
* Quality Resilient Deep Neural Networks ([PDF](https://arxiv.org/abs/1703.08119))
* How ConvNets model Non-linear Transformations ([PDF](https://arxiv.org/abs/1702.07664))
* Active Convolution: Learning the Shape of Convolution for Image Classification ([PDF](https://arxiv.org/abs/1703.09076))
* Multi-Scale Dense Convolutional Networks for Efficient Prediction ([PDF](https://arxiv.org/abs/1703.09844), [Project/Code](https://github.com/gaohuang/MSDNet))
* Coordinating Filters for Faster Deep Neural Networks ([PDF](https://arxiv.org/abs/1703.09746), [Project/Code](https://github.com/wenwei202/caffe/tree/sfm))
* A Genetic Programming Approach to Designing Convolutional Neural Network Architectures ([PDF](https://arxiv.org/abs/1704.00764))
* On Generalization and Regularization in Deep Learning ([PDF](https://arxiv.org/abs/1704.01312))
* Interpretable Explanations of Black Boxes by Meaningful Perturbation ([PDF](https://arxiv.org/abs/1704.03296))
* Energy Propagation in Deep Convolutional Neural Networks ([PDF](https://arxiv.org/abs/1704.03636))
* Introspection: Accelerating Neural Network Training By Learning Weight Evolution ([PDF](https://arxiv.org/abs/1704.04959))
* Deeply-Supervised Nets ([PDF](https://arxiv.org/abs/1409.5185))
* Speeding up Convolutional Neural Networks By Exploiting the Sparsity of Rectifier Units ([PDF](https://arxiv.org/abs/1704.07724))
* Inception Recurrent Convolutional Neural Network for Object Recognition ([PDF](https://arxiv.org/abs/1704.07709))
* Residual Attention Network for Image Classification ([PDF](https://arxiv.org/abs/1704.06904))
* The Landscape of Deep Learning Algorithms ([PDF](https://arxiv.org/abs/1705.07038))
* Pixel Deconvolutional Networks ([PDF](https://arxiv.org/abs/1705.06820))
* Dilated Residual Networks ([PDF](https://arxiv.org/abs/1705.09914))
* A Kernel Redundancy Removing Policy for Convolutional Neural Network ([PDF](https://arxiv.org/abs/1705.10748))
* Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour ([PDF](https://arxiv.org/abs/1706.02677))
* Learning Spatial Regularization with Image-level Supervisions for Multi-label Image Classification ([PDF](https://arxiv.org/abs/1702.05891), [Project/Code](https://github.com/zhufengx/SRN_multilabel/), [Reading Note](https://joshua19881228.github.io/2017-07-06-SRN/))
* VisualBackProp: efficient visualization of CNNs ([PDF](https://arxiv.org/abs/1611.05418))
* Pruning Convolutional Neural Networks for Resource Efficient Inference ([PDF](https://arxiv.org/abs/1611.06440), [Project/Code](https://github.com/jacobgil/pytorch-pruning))
* Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly ([PDF](https://arxiv.org/abs/1707.00600), [Project/Code](http://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/))
* ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices ([PDF](https://arxiv.org/abs/1707.01083), [Caffe Implementation](https://github.com/camel007/Caffe-ShuffleNet))
* Submanifold Sparse Convolutional Networks ([PDF](https://arxiv.org/abs/1706.01307), [Project/Code](https://github.com/facebookresearch/SparseConvNet))
* Dual Path Networks ([PDF](https://arxiv.org/abs/1707.01629))
* ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression ([PDF](https://arxiv.org/abs/1707.06342), [Project/Code](https://github.com/Roll920/ThiNet), [Reading Note](https://joshua19881228.github.io/2017-08-03-ThiNet/))
* Memory-Efficient Implementation of DenseNets ([PDF](https://arxiv.org/abs/1707.06990))
* Residual Attention Network for Image Classification ([PDF](https://arxiv.org/abs/1704.06904), [Project/Code](https://github.com/buptwangfei/residual-attention-network))
* An Effective Training Method For Deep Convolutional Neural Network ([PDF](https://arxiv.org/abs/1708.01666))
* Learning to Transfer ([PDF](https://arxiv.org/abs/1708.05629))
* Learning Efficient Convolutional Networks through Network Slimming ([PDF](https://arxiv.org/abs/1708.06519), [Project/Code](https://github.com/liuzhuang13/slimming))
* Super-Convergence: Very Fast Training of Residual Networks Using Large Learning Rates ([PDF](https://arxiv.org/pdf/1506.01186.pdf), [Project/Code](https://github.com/lnsmith54/super-convergence))
* Hierarchical loss for classification ([PDF](https://arxiv.org/abs/1709.01062))
* Convolutional Gaussian Processes ([PDF](https://arxiv.org/abs/1709.01894), [Code/Project](https://github.com/markvdw/convgp))
* Interpretable Convolutional Neural Networks ([PDF](https://arxiv.org/abs/1710.00935))
* What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? ([PDF](https://arxiv.org/abs/1703.04977))
* Porcupine Neural Networks: (Almost) All Local Optima are Global ([PDF](https://arxiv.org/abs/1710.02196))
* Generalization in Deep Learning ([PDF](https://arxiv.org/abs/1710.05468))
* A systematic study of the class imbalance problem in convolutional neural networks ([PDF](https://arxiv.org/abs/1710.05381))
* Interpretable Transformations with Encoder-Decoder Networks ([PDF](https://arxiv.org/abs/1710.07307), [Project/Code](http://visual.cs.ucl.ac.uk/pubs/interpTransform/))
* One pixel attack for fooling deep neural networks ([PDF](https://arxiv.org/abs/1710.08864))

### GAN ###

* Face Aging with Contextual Generative Adversarial Nets ([PDF](https://arxiv.org/abs/1802.00237), [Project/Code](https://www.arxiv-vanity.com/papers/1802.00237/))
* Deformable GANs for Pose-based Human Image Generation ([PDF](https://arxiv.org/abs/1801.00055), [Project/Code](https://github.com//AliaksandrSiarohin/pose-gan))
* ComboGAN: Unrestrained Scalability for Image Domain Translation ([PDF](https://arxiv.org/abs/1712.06909), [Project/Code](https://github.com//AAnoosheh/ComboGAN))
* Eye In-Painting with Exemplar Generative Adversarial Networks ([PDF](https://arxiv.org/abs/1712.03999))
* Disentangled Person Image Generation ([PDF](https://arxiv.org/abs/1712.02621))
* Fader Networks: Manipulating Images by Sliding Attributes ([PDF](https://arxiv.org/abs/1706.00409), [Code/Project](https://github.com/facebookresearch/FaderNetworks))
* Are GANs Created Equal? A Large-Scale Study ([PDF](https://arxiv.org/abs/1711.10337))
* StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation ([PDF](https://arxiv.org/abs/1711.09020), [Project/Code](https://github.com//yunjey/StarGAN))
* Two Birds with One Stone: Iteratively Learn Facial Attributes with GANs ([PDF](https://arxiv.org/abs/1711.06078), [Project/Code](https://github.com//punkcure/Iterative-GAN))
* Spectral Normalization for Generative Adversarial Networks ([PDF](https://openreview.net/pdf?id=B1QRgziT-))
* XGAN: Unsupervised Image-to-Image Translation for many-to-many Mappings ([PDF](https://arxiv.org/abs/1711.05139))
* How Generative Adversarial Nets and its variants Work: An Overview of GAN ([PDF](https://arxiv.org/abs/1711.05914))
* DNA-GAN: Learning Disentangled Representations from Multi-Attribute Images ([PDF](https://arxiv.org/abs/1711.05415), [Project/Code](https://github.com//Prinsphield/DNA-GAN))
* Sobolev GAN ([PDF](https://arxiv.org/abs/1711.04894))
* Data Augmentation Generative Adversarial Networks ([PDF](https://arxiv.org/abs/1711.04340))
* Conditional Autoencoders with Adversarial Information Factorization ([PDF](https://arxiv.org/abs/1711.05175), [Project/Code](https://github.com/ToniCreswell/attribute-cVAEGAN))
* Progressive Growing of GANs for Improved Quality, Stability, and Variation ([PDF](https://arxiv.org/abs/1710.10196), [Project/Code](https://github.com/tkarras/progressive_growing_of_gans), [Torch](https://github.com/nashory/progressive-growing-torch), [PyTorch](https://github.com//github-pengge/PyTorch-progressive_growing_of_gans), [Reading Note](https://joshua19881228.github.io/2017-11-12-ProgressiveGrowing/))
* Bayesian GAN ([PDF](https://arxiv.org/abs/1705.09558), [Project/Code](https://github.com//andrewgordonwilson/bayesgan/))
* Metric Learning-based Generative Adversarial Network ([PDF](https://arxiv.org/abs/1711.02792))
* Flexible Prior Distributions for Deep Generative Models ([PDF](https://arxiv.org/abs/1710.11383))
* Data Augmentation in Classification using GAN ([PDF](https://arxiv.org/abs/1711.00648))
* Semantically Decomposing the Latent Spaces of Generative Adversarial Networks ([PDF](https://arxiv.org/abs/1705.07904))
* Multi-View Data Generation Without View Supervision ([PDF](https://arxiv.org/abs/1711.00305))
* StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks ([PDF](https://arxiv.org/abs/1710.10916))
* Generative Adversarial Networks ([PDF](https://arxiv.org/abs/1406.2661))
* Stacked Generative Adversarial Networks ([PDF](https://arxiv.org/abs/1612.04357))
* Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks ([PDF](https://arxiv.org/abs/1612.05424))
* Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks ([PDF](https://arxiv.org/abs/1511.06434))
* Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks ([PDF](https://arxiv.org/abs/1506.05751))
* NIPS 2016 Tutorial: Generative Adversarial Networks ([PDF](https://arxiv.org/abs/1701.00160))
* Wasserstein GAN ([PDF](https://arxiv.org/abs/1701.07875))
* Adversarial Discriminative Domain Adaptation ([PDF](https://arxiv.org/abs/1702.05464), [Reading Note](https://joshua19881228.github.io/2017-02-22-ADDA/))
* Generative Adversarial Nets with Labeled Data by Activation Maximization ([PDF](https://arxiv.org/abs/1703.02000))
* Triple Generative Adversarial Nets ([PDF](https://arxiv.org/abs/1703.02291))
* On the Quantative Evaluation of Deep Generative Models ([PDF](http://www.cs.cmu.edu/~rsalakhu/talk_Eval.pdf))
* Adversarial Transformation Networks: Learning to Generate Adversarial Examples ([PDF](https://arxiv.org/abs/1703.09387))
* Improved Training of Wasserstein GANs ([PDF](https://arxiv.org/abs/1704.00028), [Project/Code](https://github.com/igul222/improved_wgan_trainin))
* Generate To Adapt: Aligning Domains using Generative Adversarial Networks ([PDF](https://arxiv.org/abs/1704.01705))
* Adversarial Generator-Encoder Networks ([PDF](https://arxiv.org/abs/1704.02304), [Project/Code](https://github.com/DmitryUlyanov/AGE))
* Training Triplet Networks with GAN ([PDF](https://arxiv.org/abs/1704.02227))
* Multi-Agent Diverse Generative Adversarial Networks ([PDF](https://arxiv.org/abs/1704.02906))
* GP-GAN: Towards Realistic High-Resolution Image Blending ([PDF](https://arxiv.org/abs/1703.07195), [Project/Code](https://github.com/wuhuikai/GP-GAN))
* BEGAN: Boundary Equilibrium Generative Adversarial Networks ([PDF](https://www.arxiv.org/abs/1703.10717))
* MAGAN: Margin Adaptation for Generative Adversarial Networks ([PDF](https://arxiv.org/abs/1704.03817))
* Pose Guided Person Image Generation ([PDF](https://arxiv.org/abs/1705.09368))
* On the Effects of Batch and Weight Normalization in Generative Adversarial Networks ([PDF](https://arxiv.org/abs/1704.03971), [Project/Code](https://github.com/stormraiser/GAN-weight-norm))
* Aesthetic-Driven Image Enhancement by Adversarial Learning ([PDF](https://arxiv.org/abs/1707.05251))
* VEEGAN: Reducing Mode Collapse in GANs using Implicit Variational Learning ([PDF](https://arxiv.org/abs/1705.07761), [Project/Code](https://akashgit.github.io/VEEGAN/)
* MoCoGAN: Decomposing Motion and Content for Video Generation ([PDF](https://arxiv.org/abs/1707.04993), [Project/Code](https://github.com/sergeytulyakov/mocogan))
* Generative Adversarial Networks: An Overview ((PDF)[https://arxiv.org/abs/1710.07035])
* SalGAN: Visual Saliency Prediction with Generative Adversarial Networks ([PDF](https://arxiv.org/abs/1701.01081), [Project/Code](https://github.com/imatge-upc/saliency-salgan-2017))

### MACHINE LEARNING ###

* Metric Learning with Adaptive Density Discrimination ([PDF](https://arxiv.org/abs/1511.05939), [PyTorch](https://github.com/vithursant/MagnetLoss-PyTorch/), [TF](https://github.com/pumpikano/tf-magnet-loss))
* Accelerated Gradient Descent Escapes Saddle Points Faster than Gradient Descent ([PDF](https://arxiv.org/abs/1711.10456))
* [计算机视觉与机器学习 【随机森林】](https://joshua19881228.github.io/2016-08-03-CV-ML/)
* [计算机视觉与机器学习 【深度学习中的激活函数】](https://joshua19881228.github.io/2016-08-01-CV-ML/)
* [我爱机器学习](https://www.52ml.net/) 机器学习干货站
* [Bayesian Reasoning and Machine Learning](http://web4.cs.ucl.ac.uk/staff/D.Barber/pmwiki/pmwiki.php?n=Brml.HomePage)
* Stochastic Gradient Descent as Approximate Bayesian Inference ([PDF](https://arxiv.org/abs/1704.04289))

### LIGHT-WEIGHT MODEL/EMBEDDED/MOBILE/MODEL COMPRESSION ###

* Building Efficient ConvNets using Redundant Feature Pruning ([PDF](https://arxiv.org/abs/1802.07653), [Project/Code](https://github.com/babajide07/Redundant-Feature-Pruning-Pytorch-Implementation))
* Multi-Scale Dense Networks for Resource Efficient Image Classification ([PDF](https://arxiv.org/abs/1703.09844))
* Net-Trim: Convex Pruning of Deep Neural Networks with Performance Guarantee ([pdf](https://arxiv.org/abs/1611.05162))
* NISP: Pruning Networks using Neuron Importance Score Propagation ([PDF](https://arxiv.org/abs/1711.05908))
* Caffeinated FPGAs: FPGA Framework For Convolutional Neural Networks ([PDF](https://arxiv.org/abs/1609.09671))
* Comprehensive Evaluation of OpenCL-based Convolutional Neural Network Accelerators in Xilinx and Altera FPGAs  ([PDF](https://arxiv.org/abs/1609.09296))
* FINN: A Framework for Fast, Scalable Binarized Neural Network Inference ([PDF](http://www.idi.ntnu.no/~yamanu/2017-fpga-finn-preprint.pdf))
* Two-Bit Networks for Deep Learning on Resource-Constrained Embedded Devices ([PDF](https://arxiv.org/abs/1701.00485))
* SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size ([PDF](https://arxiv.org/abs/1602.07360), [Project/Code](https://github.com/DeepScale/SqueezeNet))
* MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications ([PDF](https://arxiv.org/abs/1704.04861), [Caffe Implementation](https://github.com/farmingyard/caffe-mobilenet), [Reading Note](https://joshua19881228.github.io/2017-07-19-MobileNet/))
* Binarized Convolutional Neural Networks with Separable Filters for Efficient Hardware Acceleration ([PDF](https://arxiv.org/abs/1707.04693))
* Channel Pruning for Accelerating Very Deep Neural Networks ([PDF](https://arxiv.org/abs/1707.06168), [Project/Code](https://github.com/yihui-he/channel-pruning))
* Quantized Convolutional Neural Networks for Mobile Devices ([PDF](https://arxiv.org/abs/1512.06473), [Project/Code](https://github.com/jiaxiang-wu/quantized-cnn))
* Squeeze-and-Excitation Networks ([PDF](https://arxiv.org/abs/1709.01507))
* Domain-adaptive deep network compression ([PDF](https://arxiv.org/abs/1709.01041))
* Embedded Binarized Neural Networks ([PDF](https://arxiv.org/abs/1709.02260))
* Keynote: Small Neural Nets Are Beautiful: Enabling Embedded Systems with Small Deep-Neural-Network Architectures ([PDF](https://arxiv.org/abs/1710.02759))
* A Survey of Model Compression and Acceleration for Deep Neural Networks ([https://arxiv.org/abs/1710.09282])

### ReID ###

* Video Person Re-identification by Temporal Residual Learning ([PDF](https://arxiv.org/abs/1802.07918))
* Harmonious Attention Network for Person Re-Identification ([PDF](https://arxiv.org/abs/1802.08122))
* In Defense of the Triplet Loss for Person Re-Identification ([PDF](https://arxiv.org/abs/1703.07737))
* Deep Spatial Feature Reconstruction for Partial Person Re-identification: Alignment-Free Approach ([PDF](https://arxiv.org/abs/1801.00881))
* AlignedReID: Surpassing Human-Level Performance in Person Re-Identification ([PDF](https://arxiv.org/abs/1711.08184))
* A Discriminatively Learned CNN Embedding for Person Re-identification ([PDF](https://arxiv.org/abs/1611.05666), [Project/Code](https://github.com/layumi/2016_person_re-ID))
* Learning Deep Neural Networks for Vehicle Re-ID with Visual-spatio-temporal Path Proposals ([PDF](https://arxiv.org/abs/1708.03918))
* Beyond triplet loss: a deep quadruplet network for person re-identification ([PDF](https://arxiv.org/abs/1704.01719))
* Person Re-identification by Local Maximal Occurrence Representation and Metric Learning ([PDF](http://www.cbsr.ia.ac.cn/users/scliao/papers/Liao-CVPR15-LOMO-XQDA.pdf), [Project/Code](http://www.cbsr.ia.ac.cn/users/scliao/projects/lomo_xqda/))
* Person Re-identification: Past, Present and Future ([PDF](https://arxiv.org/abs/1610.02984))
* Unsupervised Person Re-identification: Clustering and Fine-tuning ([PDF](https://arxiv.org/abs/1705.10444), [Project/Code](https://github.com/hehefan/Unsupervised-Person-Re-identification-Clustering-and-Fine-tuning))
* Jointly Attentive Spatial-Temporal Pooling Networks for Video-based Person Re-Identification ([PDF](https://arxiv.org/abs/1708.02286))
* Divide and Fuse: A Re-ranking Approach for Person Re-identification ([PDF](https://arxiv.org/abs/1708.04169))
* Learning Deep Context-aware Features over Body and Latent Parts for Person Re-identification ([PDF](https://arxiv.org/abs/1710.06555))
* HydraPlus-Net: Attentive Deep Features for Pedestrian Analysis ([PDF](https://arxiv.org/abs/1709.09930), [Project/Code](https://github.com/xh-liu/HydraPlus-Net))

### FASHION ###

* Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba ([PDF](https://arxiv.org/abs/1803.02349))
* Visually-Aware Fashion Recommendation and Design with Generative Image Models ([PDF](Visually-Aware Fashion Recommendation and Design with Generative Image Models))
* Be Your Own Prada: Fashion Synthesis with Structural Coherence ([PDF](https://arxiv.org/abs/1710.07346), [Project/Code](http://mmlab.ie.cuhk.edu.hk/projects/FashionGAN/), [Reading Note](https://joshua19881228.github.io/2017-10-31-DeepFashion/))
* Style2Vec: Representation Learning for Fashion Items from Style Sets ([PDF](https://arxiv.org/abs/1708.04014))
* Dress like a Star: Retrieving Fashion Products from Videos ([PDF](https://arxiv.org/abs/1710.07198))
* The Conditional Analogy GAN: Swapping Fashion Articles on People Images ([PDF](https://arxiv.org/abs/1709.04695))

### OTHER ###

* End-to-end weakly-supervised semantic alignment ([PDF](https://arxiv.org/abs/1712.06861), [Project/Code](http://www.di.ens.fr/willow/research/weakalign/))
* Dense Optical Flow based Change Detection Network Robust to Difference of Camera Viewpoints ([PDF](https://arxiv.org/abs/1712.02941))
* Dual-Path Convolutional Image-Text Embedding ([PDF](https://arxiv.org/abs/1711.05535), [Project/Code](https://github.com//layumi/Image-Text-Embedding))
* The Promise and Peril of Human Evaluation for Model Interpretability ([PDF](https://arxiv.org/abs/1711.07414))
* Semantic Image Retrieval via Active Grounding of Visual Situations ([PDF](https://arxiv.org/abs/1711.00088))
* LIFT: Learned Invariant Feature Transform ([PDF](https://arxiv.org/abs/1603.09114))
* Learning Aligned Cross-Modal Representations from Weakly Aligned Data ([PDF](http://cmplaces.csail.mit.edu/content/paper.pdf), [Project/Code](http://cmplaces.csail.mit.edu/))
* Multi-Task Curriculum Transfer Deep Learning of Clothing Attributes ([PDF](https://arxiv.org/abs/1610.03670))
* End-to-end Learning of Deep Visual Representations for Image Retrieval ([PDF](https://arxiv.org/abs/1610.07940))
* SoundNet: Learning Sound Representations from Unlabeled Video ([PDF](http://web.mit.edu/vondrick/soundnet.pdf))
* Bags of Local Convolutional Features for Scalable Instance Search ([PDF](https://arxiv.org/abs/1604.04653), [Project/Code](https://imatge-upc.github.io/retrieval-2016-icmr/))
* Universal Correspondence Network ([PDF](http://cvgl.stanford.edu/projects/ucn/choy_nips16.pdf), [Project/Code](http://cvgl.stanford.edu/projects/ucn/))
* Judging a Book By its Cover ([PDF](https://arxiv.org/abs/1610.09204))
* Generalisation and Sharing in Triplet Convnets for Sketch based Visual Search ([PDF](https://arxiv.org/abs/1611.05301))
* Analysis and Optimization of Loss Functions for Multiclass, Top-k, and Multilabel Classification ([PDF](https://arxiv.org/abs/1612.03663))
* Automatic generation of large-scale handwriting fonts via style learning ([PDF](http://delivery.acm.org/10.1145/3010000/3005371/a12-lian.pdf?ip=101.36.73.155&id=3005371&acc=OPEN&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E6D218144511F3437&CFID=878397374&CFTOKEN=60453893&__acm__=1482192254_e947f14035db372a97d831530a7c05b1))
* Image Retrieval with Deep Local Features and Attention-based Keypoints ([PDF](https://arxiv.org/abs/1612.06321))
* Visual Discovery at Pinterest ([PDF](https://arxiv.org/abs/1702.04680))
* Learning to Detect Human-Object Interactions ([PDF](https://arxiv.org/abs/1702.05448), [Project/Code](http://www-personal.umich.edu/~ywchao/hico/), [Reading Note](https://joshua19881228.github.io/2017-02-28-HORCNN/))
* Learning Deep Features via Congenerous Cosine Loss for Person Recognition ([PDF](https://arxiv.org/abs/1702.06890))
* Large-Scale Evolution of Image Classifiers ([PDF](https://arxiv.org/abs/1703.01041))
* Deep Variation-structured Reinforcement Learning for Visual Relationship and Attribute Detection ([PDF](https://arxiv.org/abs/1703.03054))
* Twitter100k: A Real-world Dataset for Weakly Supervised Cross-Media Retrieval ([PDF](https://arxiv.org/abs/1703.06618), [Project/Code](https://github.com/huyt16/Twitter100k/))
* Mixture of Counting CNNs: Adaptive Integration of CNNs Specialized to Specific Appearance for Crowd Counting ([PDF](https://arxiv.org/abs/1703.09393))
* Computer Vision for Autonomous Vehicles: Problems, Datasets and State-of-the-Art ([PDF](https://arxiv.org/abs/1704.05519), [Project/Code](http://www.cvlibs.net/projects/autonomous_vision_survey/))
* Learning Features by Watching Objects Move ([PDF](https://arxiv.org/abs/1612.06370), [Project/Code](https://people.eecs.berkeley.edu/~pathak/unsupervised_video/))
* GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence ([PDF](http://jwbian.net/Papers/GMS_CVPR17.pdf), [Project/Code](http://jwbian.net/gms))
* ResnetCrowd: A Residual Deep Learning Architecture for Crowd Counting, Violent Behaviour Detection and Crowd Density Level Classification ([PDF](https://arxiv.org/abs/1705.10698))
* Learning Cross-modal Embeddings for Cooking Recipes and Food Images ([PDF](http://im2recipe.csail.mit.edu/im2recipe.pdf), [Project/Code](http://im2recipe.csail.mit.edu/))
* Convolutional neural network architecture for geometric matching ([PDF](https://arxiv.org/abs/1703.05593), [Project/Code](http://www.di.ens.fr/willow/research/cnngeometric/))
* Semantic Compositional Networks for Visual Captioning ([PDF](https://arxiv.org/abs/1611.08002), [Project/Code](https://github.com/zhegan27/SCN_for_video_captioning))
* CNN-based Cascaded Multi-task Learning of High-level Prior and Density Estimation for Crowd Counting ([PDF](https://arxiv.org/abs/1707.09605))
* Understanding Black-box Predictions via Influence Functions ([PDF](https://arxiv.org/abs/1703.04730))
* Learning a Repression Network for Precise Vehicle Search ([PDF](https://arxiv.org/abs/1708.02386))
* Visual Graph Mining ([PDF](https://arxiv.org/abs/1708.03921))
* A Deep Multimodal Approach for Cold-start Music Recommendation ([PDF](https://arxiv.org/abs/1706.09739))
* A Multilayer-Based Framework for Online Background Subtraction with Freely Moving Cameras ([PDF](https://arxiv.org/abs/1709.01140))
* A self-organizing neural network architecture for learning human-object interactions ([PDF](https://arxiv.org/abs/1710.01916))

## INTERESTING FINDS ##

### RESOURCES/PERSPECTIVES ###

* [A Gentle Introduction to Transfer Learning for Image Classification](https://github.com/miguelgfierro/sciblog_support/blob/master/A_Gentle_Introduction_to_Transfer_Learning/Intro_Transfer_Learning.ipynb)
* [GAN Timeline](https://github.com//dongb5/GAN-Timeline)
    A timeline showing the development of Generative Adversarial Networks (GAN).
* [arXiv(Computer Vision and Pattern Recognition)](http://arxiv.org/list/cs.CV/recent) 
    A good place to explore latest papers.
* [Awesome Computer Vision](https://github.com/jbhuang0604/awesome-computer-vision)
    A curated list of awesome computer vision resources.
* [Awesome Deep Vision](https://github.com/kjw0612/awesome-deep-vision)
    A curated list of deep learning resources for computer vision.
* [Awesome MXNet](https://github.com/dmlc/mxnet/blob/master/example/README.md)
    This page contains a curated list of awesome MXnet examples, tutorials and blogs.
* [Awesome TensorFlow](https://github.com/jtoy/awesome-tensorflow)
    A curated list of awesome TensorFlow experiments, libraries, and projects.
* [gans-awesome-applications](https://github.com/nashory/gans-awesome-applications)
    Curated list of awesome GAN applications and demonstrations.
* [Deep Reinforcement Learning survey](https://github.com/andrewliao11/Deep-Reinforcement-Learning-Survey)
    This paper list is a bit different from others. The author puts some opinion and summary on it. However, to understand the whole paper, you still have to read it by yourself!
* [TensorFlow 官方文档中文版](https://github.com/jikexueyuanwiki/tensorflow-zh)
* [TensorTalk](https://tensortalk.com/)
    A place to find latest work's codes.
* [OTB Results](https://github.com/foolwood/benchmark_results)
    Object tracking benchmark
* [Adversarial Nets Papers](https://github.com/zhangqianhui/AdversarialNetsPapers)
* [Creating Human-Level AI](https://www.youtube.com/watch?v=ZHYXp3gJCaI)
* [cv-tricks.com](http://cv-tricks.com/)
* [Find deep learning models for your mobile platform](http://mlmodelzoo.com/)
* [ICCV 2017 Open Access Repository](http://openaccess.thecvf.com/ICCV2017.py)

### PROJECTS ###

* [Neural Network Tools: Converter, Constructor and Analyser](https://github.com//hahnyuan/nn_tools)
    For caffe, pytorch, tensorflow, draknet and so on.
* [TensorFlow Examples](https://github.com/aymericdamien/TensorFlow-Examples)
    TensorFlow Tutorial with popular machine learning algorithms implementation. This tutorial was designed for easily diving into TensorFlow, through examples.It is suitable for beginners who want to find clear and concise examples about TensorFlow. For readability, the tutorial includes both notebook and code with explanations.
* [TensorFlow Tutorials](https://github.com/Hvass-Labs/TensorFlow-Tutorials)
    These tutorials are intended for beginners in Deep Learning and TensorFlow. Each tutorial covers a single topic. The source-code is well-documented. There is a YouTube video for each tutorial.
* [Home Surveilance with Facial Recognition](https://github.com/BrandonJoffe/home_surveillance)
* [Deep Learning algorithms with TensorFlow](https://github.com/blackecho/Deep-Learning-TensorFlow)
	This repository is a collection of various Deep Learning algorithms implemented using the TensorFlow library. This package is intended as a command line utility you can use to quickly train and evaluate popular Deep Learning models and maybe use them as benchmark/baseline in comparison to your custom models/datasets. 
* [TensorLayer](https://github.com/zsdonghao/tensorlayer) 
    TensorLayer is designed to use by both Researchers and Engineers, it is a transparent library built on the top of Google TensorFlow. It is designed to provide a higher-level API to TensorFlow in order to speed-up experimentations and developments. TensorLayer is easy to be extended and modified. In addition, we provide many examples and tutorials to help you to go through deep learning and reinforcement learning.
* [Easily Create High Quality Object Detectors with Deep Learning](http://blog.dlib.net/2016/10/easily-create-high-quality-object.html)
	Using [dlib](http://dlib.net/) to train a CNN to detect.
* [Command Line Neural Network](https://github.com/hugorut/neural-cli)
	Neuralcli provides a simple command line interface to a python implementation of a simple classification neural network. Neuralcli allows a quick way and easy to get instant feedback on a hypothesis or to play around with one of the most popular concepts in machine learning today.
* [LSTM for Human Activity Recognition](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition/)
	Human activity recognition using smartphones dataset and an LSTM RNN. The project is based on Tesorflow. A MXNet implementation is [MXNET-Scala Human Activity Recognition](https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Mxnet-Scala/HumanActivityRecognition).
* [YOLO in caffe](https://github.com/xingwangsfu/caffe-yolo)
    This is a caffe implementation of the YOLO:Real-Time Object Detection.
* [SSD: Single Shot MultiBox Object Detector in mxnet](https://github.com/zhreshold/mxnet-ssd)
* [MTCNN face detection and alignment in MXNet](https://github.com/pangyupo/mxnet_mtcnn_face_detection)
	This is a python/mxnet implementation of [Zhang's](https://github.com/kpzhang93/MTCNN_face_detection_alignment) work .
* [CNTK Examples: Image/Detection/Fast R-CNN](https://github.com/Microsoft/CNTK/tree/master/Examples/Image/Detection/FastRCNN)
* [Self Driving (Toy) Ferrari](https://github.com/RyanZotti/Self-Driving-Car)
* [Finding Lane Lines on the Road](https://github.com/udacity/CarND-LaneLines-P1)
* [Magenta](https://github.com/tensorflow/magenta)
	Magenta is a project from the Google Brain team that asks: Can we use machine learning to create compelling art and music? If so, how? If not, why not?
* [Adversarial Nets Papers](https://github.com/zhangqianhui/AdversarialNetsPapers)
	The classical Papers about adversarial nets 
* [Mushreco](http://mushreco.ml/en/)
	Make a photo of a mushroom and see which species it is. Determine over 200 different species. 
* [Neural Enhance](https://github.com/alexjc/neural-enhance#3-background--research)
    The neural network is hallucinating details based on its training from example images. It's not reconstructing your photo exactly as it would have been if it was HD. That's only possible in Hollywood — but using deep learning as "Creative AI" works and it is just as cool!
* [CNN Models by CVGJ](https://github.com/cvjena/cnn-models)
    This repository contains convolutional neural network (CNN) models trained on ImageNet by Marcel Simon at the Computer Vision Group Jena (CVGJ) using the Caffe framework. Each model is in a separate subfolder and contains everything needed to reproduce the results. This repository focuses currently contains the batch-normalization-variants of AlexNet and VGG19 as well as the training code for Residual Networks (Resnet). 
* [YOLO2](http://pjreddie.com/darknet/yolo/)

    >YOLOv2 uses a few tricks to improve training and increase performance. Like Overfeat and SSD we use a fully-convolutional model, but we still train on whole images, not hard negatives. Like Faster R-CNN we adjust priors on bounding boxes instead of predicting the width and height outright. However, we still predict the x and y coordinates directly. The full details are in our paper soon to be released on Arxiv, stay tuned!
    
* [Lightened CNN for Deep Face Representation](https://github.com/AlfredXiangWu/face_verification_experiment)
    The Deep Face Representation Experiment is based on Convolution Neural Network to learn a robust feature for face verification task.    
* [Recurrent dreams and filling in](http://blog.piekniewski.info/2016/12/05/recurrent-dreams-and-filling-in/)
* [MTCNN in MXnet](https://github.com/Seanlinx/mtcnn)
* [openai-gemm](https://github.com/openai/openai-gemm)

    Open single and half precision gemm implementations. The main speedups over cublas are with small minibatch and in fp16 data formats.

* [Neural Style](https://github.com/zhaw/neural_style)

    style transfer with mxnet

* [Can Convolutional Neural Networks Crack Sudoku Puzzles?](https://github.com/Kyubyong/sudoku)
* [cleverhans](https://github.com/openai/cleverhans)

    This repository contains the source code for  cleverhans , a Python library to benchmark machine learning systems' vulnerability to adversarial examples.

* [A deep learning traffic light detector using dlib and a few images from Google street view](https://sagivtech.com/2016/11/10/post-2/)
* [Paints Chainer](https://github.com/taizan/PaintsChainer)
* [Calculate deep convolution neurAl network on Cell Unit](https://github.com/luhaofang/CACU)
* [Deep Video Analytics](https://github.com/AKSHAYUBHAT/DeepVideoAnalytics)
    Deep Video Analytics provides a platform for indexing and extracting information from videos and images. Deep learning detection and recognition algorithms are used for indexing individual frames / images along with detected objects. The goal of Deep Video analytics is to become a quickly customizable platform for developing visual & video analytics applications, while benefiting from seamless integration with state or the art models released by the vision research community.
* [Yolo_mark](https://github.com/AlexeyAB/Yolo_mark) 
    Windows GUI for marking bounded boxes of objects in images for training Yolo v2
* [Yolo-Windows v2 - Windows version of Yolo Convolutional Neural Networks](https://github.com/AlexeyAB/darknet)
* [An Unsupervised Distance Learning Framework for Multimedia Retrieva](https://github.com/lucasPV/UDLF)
* [awesome-deep-vision-web-demo](https://github.com/hwalsuklee/awesome-deep-vision-web-demo)
* [Mini Caffe](https://github.com/luoyetx/mini-caffe)
	Minimal runtime core of Caffe, Forward only, GPU support and Memory efficiency.
* [Picasso: A free open-source visualizer for Convolutional Neural Networks](https://medium.com/merantix/picasso-a-free-open-source-visualizer-for-cnns-d8ed3a35cfc5)
    Picasso is a free open-source (Eclipse Public License) DNN visualization tool that gives you partial occlusion and saliency maps with minimal fuss.
* [pix2code: Generating Code from a Graphical User Interface Screenshot](https://uizard.io/research#pix2code)
* [MTCNN-light](https://github.com/AlphaQi/MTCNN-light)
    this repository is the implementation of MTCNN with no framework, Just need opencv and openblas. "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks", implemented with C++，no framework
* [MobileNet-MXNet](https://github.com/KeyKy/mobilenet-mxnet)
    This is a MXNet implementation of Google's MobileNets.
* [NoScope: 1000x Faster Deep Learning Queries over Video](http://dawn.cs.stanford.edu/2017/06/22/noscope/)
* [Caffe2 C++ Tutorials and Examples](https://github.com/leonardvandriel/caffe2_cpp_tutorial)
* [Web Image Downloader Tools](https://github.com/kencoken/imsearch-tools)

### NEWS/BLOGS ###

* [The Building Blocks of Interpretability]([PDF](https://distill.pub/2018/building-blocks/))
* [Setting the learning rate of your neural network.](https://www.jeremyjordan.me/nn-learning-rate/)
* [The Root Cause of Slow Neural Net Training](https://medium.com/south-park-commons/otodscinos-the-root-cause-of-slow-neural-net-training-fec7295c364c)
* [Why is it hard to train deep neural networks? Degeneracy, not vanishing gradients, is the key](https://severelytheoretical.wordpress.com/2018/01/01/why-is-it-hard-to-train-deep-neural-networks-degeneracy-not-vanishing-gradients-is-the-key/)
* [ResNet, AlexNet, VGG, Inception: Understanding various architectures of Convolutional Networks](http://cv-tricks.com/cnn/understand-resnet-alexnet-vgg-inception/)
* [Neural Networks For Recommender Systems](https://www.bilibili.com/video/av16499811/)
* [MIT Technology Review](https://www.technologyreview.com/)
    A good place to keep up the trends.
* [LAB41](https://gab41.lab41.org/)
    Lab41 is a Silicon Valley challenge lab where experts from the U.S. Intelligence Community (IC), academia, industry, and In-Q-Tel come together to gain a better understanding of how to work with — and ultimately use — big data.
* [Partnership on AI](http://www.partnershiponai.org/)
    Amazon, DeepMind/Google, Facebook, IBM, and Microsoft announced that they will create a non-profit organization that will work to advance public understanding of artificial intelligence technologies (AI) and formulate best practices on the challenges and opportunities within the field. Academics, non-profits, and specialists in policy and ethics will be invited to join the Board of the organization, named the Partnership on Artificial Intelligence to Benefit People and Society (Partnership on AI).
* [爱可可-爱生活](http://weibo.com/fly51fly?from=profile&wvr=6&is_all=1) 老师的推荐十分值得一看
* [Guide to deploying deep-learning inference networks and realtime object recognition tutorial for NVIDIA Jetson TX1](https://github.com/dusty-nv/jetson-inference)
* [A Return to Machine Learning](https://medium.com/@kcimc/a-return-to-machine-learning-2de3728558eb)
    This post is aimed at artists and other creative people who are interested in a survey of recent developments in machine learning research that intersect with art and culture. If you’ve been following ML research recently, you might find some of the experiments interesting but will want to skip most of the explanations.
* [ResNets, HighwayNets, and DenseNets, Oh My!](https://medium.com/@awjuliani/resnets-highwaynets-and-densenets-oh-my-9bb15918ee32#.1d3mpy4hd)
	This post walks through the logic behind three recent deep learning architectures: ResNet, HighwayNet, and DenseNet. Each make it more possible to successfully trainable deep networks by overcoming the limitations of traditional network design.
* [How to build a robot that “sees” with $100 and TensorFlow](https://www.oreilly.com/learning/how-to-build-a-robot-that-sees-with-100-and-tensorflow)
	
	>I wanted to build a robot that could recognize objects. Years of experience building computer programs and doing test-driven development have turned me into a menace working on physical projects. In the real world, testing your buggy device can burn down your house, or at least fry your motor and force you to wait a couple of days for replacement parts to arrive.

* [Navigating the unsupervised learning landscape](https://culurciello.github.io//tech/2016/06/10/unsup.html)
	Unsupervised learning is the Holy Grail of Deep Learning. The goal of unsupervised learning is to create general systems that can be trained with little data. Very little data.
* [Deconvolution and Checkerboard Artifacts](http://distill.pub/2016/deconv-checkerboard/)
* [Facial Recognition on a Jetson TX1 in Tensorflow](http://www.mattkrzus.com/face.html)
	Here's a way to hack facial recognition system together in relatively short time on NVIDIA's Jetson TX1.
* [Deep Learning with Generative and Generative Adverserial Networks – ICLR 2017 Discoveries](https://amundtveit.com/2016/11/12/deep-learning-with-generative-and-generative-adverserial-networks-iclr-2017-discoveries/)
	This blog post gives an overview of Deep Learning with Generative and Adverserial Networks related papers submitted to ICLR 2017.
* [Unsupervised Deep Learning – ICLR 2017 Discoveries](https://amundtveit.com/2016/11/12/unsupervised-deep-learning-iclr-2017-discoveries/)
	This blog post gives an overview of papers related to Unsupervised Deep Learning submitted to ICLR 2017.
* [You Only Look Twice — Multi-Scale Object Detection in Satellite Imagery With Convolutional Neural Networks](https://medium.com/the-downlinq/you-only-look-twice-multi-scale-object-detection-in-satellite-imagery-with-convolutional-neural-38dad1cf7571)
* [Deep Learning isn’t the brain](https://hackernoon.com/deep-learning-isnt-the-brain-e1d800ebb5a9?gi=1ef40a41a579)
* [iSee: Using deep learning to remove eyeglasses from faces](https://blog.insightdatascience.com/isee-removing-eyeglasses-from-faces-using-deep-learning-d4e7d935376f#.kmld0zlxz)
* [Decoding The Thought Vector](http://gabgoh.github.io/ThoughtVectors/)
* [Algorithmia will help you make your own AI-powered photo filters](http://venturebeat.com/2016/11/29/algorithmia-will-help-you-make-ai-powered-photo-filters/)
* [Deep Learning Enables You to Hide Screen when Your Boss is Approaching](http://ahogrammer.com/2016/11/15/deep-learning-enables-you-to-hide-screen-when-your-boss-is-approaching/)
* [对偶学习：一种新的机器学习范式](http://weibo.com/ttarticle/p/show?id=2309404050111117306404)
* [How to Train a GAN? Tips and tricks to make GANs work](https://github.com/soumith/ganhacks)

    >While research in Generative Adversarial Networks (GANs) continues to improve the fundamental stability of these models, we use a bunch of tricks to train them and make them stable day to day.
    
* [Highlights of IEEE Big Data 2016: Nearest Neighbours, Outliers and Deep Learning](https://medium.com/seldon-open-source-machine-learning/highlights-of-ieee-big-data-2016-nearest-neighbours-outliers-and-deep-learning-696d014d8cdf#.8xnfbuofv)
* [Some CNN visualization tools and techniques](http://www.erogol.com/cnn-visualization-tools-techniques/)
    
    Besides this post, the others written by the author are also worthy of reading.
* [Deep Learning 2016: The Year in Review](http://www.deeplearningweekly.com/blog/deep-learning-2016-the-year-in-review)
* [GANs will change the world](https://medium.com/@Moscow25/gans-will-change-the-world-7ed6ae8515ca#.uod2m7gpp)
* [colah's blog](http://colah.github.io/)
* [Analysis of Dropout](https://pgaleone.eu/deep-learning/regularization/2017/01/10/anaysis-of-dropout/)
* [NIPS 2016 Review](https://gab41.lab41.org/nips-2016-review-day-1-6e504bcf1451#.g2wwg6g6a)
* [【榜单】GitHub 最受欢迎深度学习应用项目 Top 16（持续更新）](http://mp.weixin.qq.com/s/-JxpVh_sAIXWueSBfzxbsA)
* [Why use SVM?](http://blog.yhat.com/posts/why-support-vector-machine.html)
* [TensorFlow Image Recognition on a Raspberry Pi](http://svds.com/tensorflow-image-recognition-raspberry-pi/)
* [Building Your Own Deep Learning Box](https://medium.com/@bfortuner/building-your-own-deep-learning-box-47b918aea1eb#.gabyza8vg)
* [Vehicle tracking using a support vector machine vs. YOLO](https://medium.com/@ksakmann/vehicle-detection-and-tracking-using-hog-features-svm-vs-yolo-73e1ccb35866#.b6215aaoi)
* [Understanding, generalisation, and transfer learning in deep neural networks](https://blog.acolyer.org/2017/02/27/understanding-generalisation-and-transfer-learning-in-deep-neural-networks/)
* [NVIDIA Announces The Jetson TX2, Powered By NVIDIA's "Denver 2" CPU & Pascal Graphics](http://www.phoronix.com/scan.php?page=article&item=nvidia-jtx2-denver&num=1)
* [Can FPGAs Beat GPUs in Accelerating Next-Generation Deep Learning?](https://www.nextplatform.com/2017/03/21/can-fpgas-beat-gpus-accelerating-next-generation-deep-learning/)
* [Flexible Image Tagging with Fast0Tag](https://gab41.lab41.org/flexible-image-tagging-with-fast0tag-681c6283c9b7)
* [Eye Fidelity: How Deep Learning Will Help Your Smartphone Track Your Gaze](https://blogs.nvidia.com/blog/2016/08/30/eye-tracking-deep-learning/)
* [Using Deep Learning to Find Similar Dresses](https://www.slideshare.net/HJvanVeen/using-deep-learning-to-find-similar-dresses)
* [Rules of Machine Learning: Best Practices for ML Engineering](http://211.136.65.144/cache/martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf?ich_args2=63-30201313013522_2c061a5e256a1540143166bdb81879f4_10001002_9c886d2cdfc4f4d89e3e518939a83798_1fc45e3ebbbb5d0180278ba0111aceaf)
* [Neural Network Architectures](https://culurciello.github.io/tech/2016/06/04/nets.html)
* [A Brief History of CNNs in Image Segmentation: From R-CNN to Mask R-CNN](https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4)
* [晓雷机器学习笔记](https://zhuanlan.zhihu.com/xiaoleimlnote)
* [Image Classification with 5 methods](https://github.com/Fdevmsy/Image_Classification_with_5_methods)
* [How do Convolutional Neural Networks work?](http://brohrer.github.io/how_convolutional_neural_networks_work.html)
* [基于深度卷积神经网络进行人脸识别的原理是什么](https://www.zhihu.com/question/60759296/answer/180176680)
* [10 Deep Learning projects based on Apache MXNet](https://medium.com/@julsimon/10-deep-learning-projects-based-on-apache-mxnet-8231109f3f64)
* [Off the Convex Path](http://www.offconvex.org/)
* [图像风格迁移(Neural Style)简史](https://zhuanlan.zhihu.com/p/26746283)
* [ML notes: Why the log-likelihood?](https://blog.metaflow.fr/ml-notes-why-the-log-likelihood-24f7b6c40f83)
* [Generative Adversarial Networks (GANs): Engine and Applications](https://blog.statsbot.co/generative-adversarial-networks-gans-engine-and-applications-f96291965b47)
* [Compressing deep neural nets](http://machinethink.net/blog/compressing-deep-neural-nets/)
* [Sigmoidal](https://sigmoidal.io/)
* [A Gentle Introduction to the Bag-of-Words Model](https://machinelearningmastery.com/gentle-introduction-bag-words-model/)
* [Fantastic GANs and where to find them](http://guimperarnau.com/blog/2017/03/Fantastic-GANs-and-where-to-find-them)
* [Fantastic GANs and where to find them II](http://guimperarnau.com/blog/2017/11/Fantastic-GANs-and-where-to-find-them-II)

### BENCHMARK/LEADERBOARD/DATASET ###

* [Large-Scale Landmark Recognition: A Challenge](https://landmarkscvprw18.github.io/)
* [Low-Power Image Recognition Challenge](https://rebootingcomputing.ieee.org/lpirc)
* [Open Images Dataset](https://github.com//openimages/dataset)
    Open Images is a dataset of ~9 million URLs to images that have been annotated with image-level labels and bounding boxes spanning thousands of classes.
* [Visual Tracker Benchmark](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html)
    This website contains data and code of the benchmark evaluation of online visual tracking algorithms. Join visual-tracking Google groups for further updates, discussions, or QnAs.
* [Multiple Object Tracking Benchmark](https://motchallenge.net/)
	With this benchmark we would like to pave the way for a unified framework towards more meaningful quantification of multi-target tracking.
* [Leaderboards for the Evaluations on PASCAL VOC Data](http://host.robots.ox.ac.uk:8080/leaderboard/main_bootstrap.php)
* [Open Images dataset](https://github.com/openimages/dataset)
    Open Images is a dataset of ~9 million URLs to images that have been annotated with labels spanning over 6000 categories.
* [Open Sourcing 223GB of Driving Data](https://github.com/udacity/self-driving-car)
	223GB of image frames and log data from 70 minutes of driving in Mountain View on two separate days, with one day being sunny, and the other overcast. 
* [MS COCO](http://mscoco.org)
* [UMDFaces Dataset](http://umdfaces.io/)
	UMDFaces is a face dataset which has 367,920 faces of 8,501 subjects. From this page you can download the entire dataset and the trained model for predicting the localization of the 21 keypoints. 
* [VideoNet](http://videonet.team/)
    VideoNet is a new initiative to bring together the community of researchers that have put effort into creating benchmarks for video tasks. 
* [YouTube-BoundingBoxes: A Large High-Precision Human-Annotated Data Set for Object Detection in Video](https://arxiv.org/abs/1702.00824)
* [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/index.php)
* [Duke: A New Large-scale Person Re-identification Dataset derived from DukeMTMC](https://github.com/layumi/Duke_evaluation)
    Duke is a subset of the DukeMTMC for image-based re-ID, in the format of the Market-1501 dataset. The original dataset contains 85-minute high-resolution videos from 8 different cameras. Hand-drawn pedestrain bounding boxes are available.
* [Releasing the World’s Largest Street-level Imagery Dataset for Teaching Machines to See](http://blog.mapillary.com/product/2017/05/03/mapillary-vistas-dataset.html)
	>Today we present the Mapillary Vistas Dataset—the world’s largest and most diverse publicly available, pixel-accurately and instance-specifically annotated street-level imagery dataset for empowering autonomous mobility and transport at the global scale.
* [WEBVISION DATASET](http://www.vision.ee.ethz.ch/webvision/) 
    >The WebVision dataset is designed to facilitate the research on learning visual representation from noisy web data. Our goal is to disentangle the deep learning techniques from huge human labor on annotating large-scale vision dataset. We release this large scale web images dataset as a benchmark to advance the research on learning from web data, including weakly supervised visual representation learning, visual transfer learning, text and vision, etc. 
* [DukeMTMC4ReID](https://github.com/NEU-Gou/DukeReID)
    >DukeMTMC4ReID dataset is new large-scale real-world person re-id dataset based on DukeMTMC.
* [Person Re-identification Datasets](http://robustsystems.coe.neu.edu/sites/robustsystems.coe.neu.edu/files/systems/projectpages/reiddataset.html)
    >Person re-identification has drawn intensive attention in the computer vision society in recent decades. As far as we know, this page collects all public datasets that have been tested by person re-identification algorithms.
* [MIT Saliency Benchmark](http://saliency.mit.edu/home.html)
* [How far are we from solving the 2D & 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)](https://www.adrianbulat.com/face-alignment/)
* [PoseTrack: A Benchmark for Human Pose Estimation and Tracking](https://posetrack.net/workshops/iccv2017/posetrack-challenge-results.html)

### TOOLKITS ###

* [Caffe](http://caffe.berkeleyvision.org/)
    Caffe is a deep learning framework made with expression, speed, and modularity in mind. It is developed by the Berkeley Vision and Learning Center (BVLC) and by community contributors. Yangqing Jia created the project during his PhD at UC Berkeley. Caffe is released under the BSD 2-Clause license.
* [Caffe2](https://github.com/caffe2/caffe2)
    Caffe2 is a deep learning framework made with expression, speed, and modularity in mind. It is an experimental refactoring of Caffe, and allows a more flexible way to organize computation.
* [Caffe on Intel](https://github.com/intel/caffe)
    This fork of BVLC/Caffe is dedicated to improving performance of this deep learning framework when running on CPU, in particular Intel® Xeon processors (HSW+) and Intel® Xeon Phi processors
* [TensorFlow](https://github.com/tensorflow/tensorflow)
    TensorFlow is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) that flow between them. This flexible architecture lets you deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device without rewriting code. TensorFlow also includes TensorBoard, a data visualization toolkit.
* [MXNet](http://mxnet.io/)
    MXNet is a deep learning framework designed for both efficiency and flexibility. It allows you to mix the flavours of symbolic programming and imperative programming to maximize efficiency and productivity. In its core, a dynamic dependency scheduler that automatically parallelizes both symbolic and imperative operations on the fly. A graph optimization layer on top of that makes symbolic execution fast and memory efficient. The library is portable and lightweight, and it scales to multiple GPUs and multiple machines.
* [neon](https://github.com/NervanaSystems/neon)
    neon is Nervana's Python based Deep Learning framework and achieves the fastest performance on modern deep neural networks such as AlexNet, VGG and GoogLeNet. Designed for ease-of-use and extensibility.
* [Piotr's Computer Vision Matlab Toolbox](https://pdollar.github.io/toolbox/)
    This toolbox is meant to facilitate the manipulation of images and video in Matlab. Its purpose is to complement, not replace, Matlab's Image Processing Toolbox, and in fact it requires that the Matlab Image Toolbox be installed. Emphasis has been placed on code efficiency and code reuse. Thanks to everyone who has given me feedback - you've helped make this toolbox more useful and easier to use. 
* [NVIDIA Developer](https://developer.nvidia.com/)
* [nvCaffe](https://github.com/NVIDIA/caffe/tree/experimental/fp16)
    A special branch of caffe is used on TX1 which includes support for FP16.
* [dlib](http://dlib.net/)
    Dlib is a modern C++ toolkit containing machine learning algorithms and tools for creating complex software in C++ to solve real world problems. It is used in both industry and academia in a wide range of domains including robotics, embedded devices, mobile phones, and large high performance computing environments. Dlib's open source licensing allows you to use it in any application, free of charge. 
* [OpenCV](http://opencv.org/)
    OpenCV is released under a BSD license and hence it’s free for both academic and commercial use. It has C++, C, Python and Java interfaces and supports Windows, Linux, Mac OS, iOS and Android. OpenCV was designed for computational efficiency and with a strong focus on real-time applications.
* [CNNdroid](https://github.com/ENCP/CNNdroid)
	CNNdroid is an open source library for execution of trained convolutional neural networks on Android devices. 
* [tiny dnn](https://github.com/tiny-dnn/tiny-dnn)
    tiny-dnn is a C++11 implementation of deep learning. It is suitable for deep learning on limited computational resource, embedded systems and IoT devices.

    An introduction to this toolkit at《[Deep learning with C++ - an introduction to tiny-dnn》by Taiga Nomi](http://www.slideshare.net/ssuser756ec5/deep-learning-with-c-an-introduction-to-tinydnn)

* [CaffeMex](https://github.com/sciencefans/CaffeMex_v2)
    A multi-GPU & memory-reduced MAT-Caffe on LINUX and WINDOWS
* [ARCore](https://developers.google.com/ar/) ARCore is a platform for building augmented reality apps on Android. ARCore uses three key technologies to integrate virtual content with the real world as seen through your phone's camera
* [CNTK](https://github.com/Microsoft/CNTK) Microsoft Cognitive Toolkit (CNTK), an open source deep-learning toolkit.
* [ONNX](http://onnx.ai/) ONNX is a open format to represent deep learning models. With ONNX, AI developers can more easily move models between state-of-the-art tools and choose the combination that is best for them. ONNX is developed and supported by a community of partners.
* [PyToune](http://pytoune.org/en/latest/) is a Keras-like framework for PyTorch and handles much of the boilerplating code needed to train neural networks.
* [Deep Learning Studio - Desktop | DeepCognition.ai] (http://deepcognition.ai/desktop/) is a single user solution that runs locally on your hardware.  Desktop version allows you to train models on your GPU(s) without uploading data to the cloud.   The platform supports transparent multi-GPU training for up to 4 GPUs.  Additional GPUs are supported in Deep Learning Studio – Enterprise.

### LEARNING/TRICKS ###

* [Deep Learning Courses](https://machinelearningmastery.com/deep-learning-courses/)
* [Backpropagation Algorithm](http://deeplearning.stanford.edu/wiki/index.php/Backpropagation_Algorithm)
    A website that explain how Backpropagation Algorithm works.
* [Deep Learning (textbook authored by Ian Goodfellow and Yoshua Bengio and Aaron Courville)](http://www.deeplearningbook.org/)
    The Deep Learning textbook is a resource intended to help students and practitioners enter the field of machine learning in general and deep learning in particular. 
* [Neural Networks and Deep Learning (online book authored by Michael Nielsen)](http://neuralnetworksanddeeplearning.com/index.html)
    Neural Networks and Deep Learning is a free online book. The book will teach you about 1) Neural networks, a beautiful biologically-inspired programming paradigm which enables a computer to learn from observational data and 2) Deep learning, a powerful set of techniques for learning in neural networks. Neural networks and deep learning currently provide the best solutions to many problems in image recognition, speech recognition, and natural language processing. This book will teach you many of the core concepts behind neural networks and deep learning.
* [Computer Vision: Algorithms and Applications](http://szeliski.org/Book/)
    This book is largely based on the computer vision courses that Richard Szeliski has co-taught at the University of Washington (2008, 2005, 2001) and Stanford (2003) with Steve Seitz and David Fleet.
* [Must Know Tips/Tricks in Deep Neural Networks ](http://210.28.132.67/weixs/project/CNNTricks/CNNTricks.html)
    Many implementation details for DCNNs are collected and concluded. Extensive implementation details are introduced, i.e., tricks or tips, for building and training your own deep networks.
* [The zen of gradient descent](http://blog.mrtz.org/2013/09/07/the-zen-of-gradient-descent.html)
* [Deriving the Gradient for the Backward Pass of Batch Normalization](https://kevinzakka.github.io/2016/09/14/batch_normalization/)
* [Reinforcement Learning: An Introduction](https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html)
* [An overview of gradient descent optimization algorithms](http://sebastianruder.com/optimizing-gradient-descent/)
* [Regularizing neural networks by penalizing confident predictions](https://pan.baidu.com/s/1kUUtxdl)
* [What you need to know about data augmentation for machine learning](https://cartesianfaith.com/2016/10/06/what-you-need-to-know-about-data-augmentation-for-machine-learning/)
    Plentiful high-quality data is the key to great machine learning models. But good data doesn’t grow on trees, and that scarcity can impede the development of a model. One way to get around a lack of data is to augment your dataset. Smart approaches to programmatic data augmentation can increase the size of your training set 10-fold or more. Even better, your model will often be more robust (and prevent overfitting) and can even be simpler due to a better training set.
* [Guide to deploying deep-learning inference networks and realtime object recognition tutorial for NVIDIA Jetson TX1]
* [The Effect of Resolution on Deep Neural Network Image Classification Accuracy](https://medium.com/the-downlinq/the-effect-of-resolution-on-deep-neural-network-image-classification-accuracy-d1338e2782c5#.5iwdz2te8)
	The author explored the impact of both spatial resolution and training dataset size on the classification performance of deep neural networks in this post.
* [深度学习调参的技巧](https://www.zhihu.com/question/25097993)
* [CNN怎么调参数](https://www.zhihu.com/question/27962483)
* [视频多目标跟踪当前（2014,2015,2016）比较好的算法有哪些](https://www.zhihu.com/question/40545681)
* [5 algorithms to train a neural network](https://www.neuraldesigner.com/blog/5_algorithms_to_train_a_neural_network.html)
* [Towards Good Practices for Recognition & Detection](http://mp.weixin.qq.com/s?__biz=MzI1NTE4NTUwOQ==&mid=2650325586&idx=1&sn=69a7e8482d884dda869290581a50a6d6&chksm=f235a558c5422c4eabe75f6db92fd13a3ca662d648676461914c9bfd1f4e22affe12430afce3&mpshare=1&scene=2&srcid=1024IdcoRukOMBvDA5dPbRoV&from=timeline&isappinstalled=0#wechat_redirect)
	海康威视研究院ImageNet2016竞赛经验分享
* [What are the differences between Random Forest and Gradient Tree Boosting algorithms](https://www.quora.com/What-are-the-differences-between-Random-Forest-and-Gradient-Tree-Boosting-algorithms)
* [为什么现在的CNN模型都是在GoogleNet、VGGNet或者AlexNet上调整的](https://www.zhihu.com/question/43370067)
* [神经网络与深度学习](https://nndl.github.io/)
* [ILSVRC2016目标检测任务回顾(上)——图像目标检测(DET)](http://mp.weixin.qq.com/s/t3U_gUfe5KekrH-jDSDk_w)
* [ILSVRC2016目标检测任务回顾(下)——视频目标检测(VID)](http://mp.weixin.qq.com/s/mQ78KNuaHUTox3ql6rU-Nw)
* [How to Train a GAN? Tips and tricks to make GANs work](https://github.com/soumith/ganhacks)
* [令人拍案叫绝的Wasserstein GAN](https://zhuanlan.zhihu.com/p/25071913)
* [Mathematics for Computer Science](https://courses.csail.mit.edu/6.042/spring17/mcs.pdf)
* [生成式对抗网络 GAN 的研究进展与展望](https://mp.weixin.qq.com/s?__biz=MzI3MTA0MTk1MA==&mid=2651995256&idx=1&sn=f508a61ebd4792b4b407ac7418c4f1ab)
* [A guide to receptive field arithmetic for Convolutional Neural Networks](https://www.medium.com/@nikasa1889/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807)
* [见微知著：细粒度图像分析进展](http://geek.csdn.net/news/detail/191718)
* [目标跟踪相关资源](https://zhuanlan.zhihu.com/p/27292838)
* [Intro to Neural Networks and Machine Learning](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/)
* [Tips for Training Recurrent Neural Networks](http://danijar.com/tips-for-training-recurrent-neural-networks/)
* [A Gentle Introduction to Mini-Batch Gradient Descent and How to Configure Batch Size](http://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/)
* [Learning To See（机器学习计算机视觉入门）（英文字幕）](https://www.bilibili.com/video/av12532910/)
* [37 Reasons why your Neural Network is not working](https://medium.com/@slavivanov/4020854bd607)
* [My Neural Network isn't working! What should I do?](http://theorangeduck.com/page/neural-network-not-working)
* [Tutorial on Deep Generative Models](http://www.shakirm.com/slides/DeepGenModelsTutorial.pdfg)

## SKILLS ##

### ABOUT CAFFE ###

* [Set Up Caffe on Ubuntu14.04 64bit+NVIDIA GTX970M+CUDA7.0](https://joshua19881228.github.io/2015-08-30-Setting-Up-Caffe/)
* [VS2013配置Caffe卷积神经网络工具（64位Windows 7）——建立工程](http://blog.csdn.net/joshua_1988/article/details/45048871)
* [VS2013配置Caffe卷积神经网络工具（64位Windows 7）——准备依赖库](http://blog.csdn.net/joshua_1988/article/details/45036993)

### SETTING UP ###

* [Installation of NVIDIA GPU Driver and CUDA Toolkit](https://joshua19881228.github.io/2015-08-29-Ndriver-Cuda/)
* [Tensorflow v0.10 installed from scratch on Ubuntu 16.04, CUDA 8.0RC+Patch, cuDNN v5.1 with a 1080GTX](https://marcnu.github.io/2016-08-17/Tensorflow-v0.10-installed-from-scratch-Ubuntu-16.04-CUDA8.0RC-cuDNN5.1-1080GTX/)
* [DL小钢炮攒机心得 帮你踩坑](https://mp.weixin.qq.com/s?__biz=MzIzMzI0NjkwMw==&mid=2652210076&idx=1&sn=f7150cc62d68eb9fc29d3a65f9834104)

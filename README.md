[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)]()
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![GitHub license](https://img.shields.io/github/license/HuaizhengZhang/Awesome-System-for-Machine-Learning.svg?color=blue)](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/blob/master/LICENSE)


# Awesome-3D-Object-Detection
A curated list of research in 3D Object Detection(**Lidar-based Method**). 

You are very welcome to pull request to update this list. :smiley:   
![3D Object Detection](https://github.com/TianhaoFu/Awesome-3D-Object-Detection/blob/main/3d.png)

## Dataset
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
  - 3,712 training samples
  - 3,769 validation samples
  - 7,518 testing samples

- [nuScenes Dataset](https://www.nuscenes.org/)
  - 28k training samples
  - 6k validation samples
  - 6k testing samples

- [Lyft Dataset](https://level-5.global/data/perception/)
- [Waymo Open Dataset](https://waymo.com/open/download/)
  - 798 training sequences with around 158, 361 LiDAR samples
  - 202 validation sequences with 40, 077 LiDAR samples.

## Top conference & workshop
### Conferene
- Conference on Computer Vision and Pattern Recognition(CVPR)
- International Conference on Computer Vision(ICCV)
- European Conference on Computer Vision(ECCV)
### Workshop
- CVPR 2019 Workshop on Autonomous Driving([nuScenes 3D detection](http://cvpr2019.wad.vision/))
- CVPR 2020 Workshop on Autonomous Driving([BDD1k 3D tracking](http://cvpr2020.wad.vision/))
- CVPR 2021 Workshop on Autonomous Driving([waymo 3D detection](http://cvpr2021.wad.vision/))
- CVPR 2022 Workshop on Autonomous Driving([waymo 3D detection](http://cvpr2022.wad.vision/))
- [CVPR 2021 Workshop on 3D Vision and Robotics](https://sites.google.com/view/cvpr2021-3d-vision-robotics)
- [CVPR 2021 Workshop on 3D Scene Understanding for Vision, Graphics, and Robotics](https://scene-understanding.com/)

- [ICCV 2019 Workshop on Autonomous Driving](http://wad.ai/)
- [ICCV 2021 Workshop on Autonomous Vehicle Vision (AVVision)](https://avvision.xyz/iccv21/), [note](https://openaccess.thecvf.com/content/ICCV2021W/AVVision/papers/Fan_Autonomous_Vehicle_Vision_2021_ICCV_Workshop_Summary_ICCVW_2021_paper.pdf)
- [ICCV 2021 Workshop SSLAD Track 2 - 3D Object Detection](https://competitions.codalab.org/competitions/33236#learn_the_details)
- [ECCV 2020 Workshop on Commands for Autonomous Vehicles](https://c4av-2020.github.io/)
- [ECCV 2020 Workshop on Perception for Autonomous Driving](https://sites.google.com/view/pad2020)
## Paper (Lidar-based method)
- End-to-End Multi-View Fusion for 3D Object Detection in LiDAR Point Clouds [paper](https://github.com/Tom-Hardy-3D-Vision-Workshop/awesome-3D-object-detection/blob/master)
- Vehicle Detection from 3D Lidar Using Fully Convolutional Network(baidu) [paper](https://arxiv.org/abs/1608.07916)
- VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection [paper](https://arxiv.org/pdf/1711.06396.pdf)
- Object Detection and Classification in Occupancy Grid Maps using Deep Convolutional Networks [paper](https://arxiv.org/pdf/1805.08689.pdf)
- RT3D: Real-Time 3-D Vehicle Detection in LiDAR Point Cloud for Autonomous Driving [paper](https://www.onacademic.com/detail/journal_1000040467923610_4dfe.html)
- BirdNet: a 3D Object Detection Framework from LiDAR information [paper](https://arxiv.org/pdf/1805.01195.pdf)
- LMNet: Real-time Multiclass Object Detection on CPU using 3D LiDAR [paper](https://arxiv.org/pdf/1805.04902.pdf)
- HDNET: Exploit HD Maps for 3D Object Detection [paper](https://link.zhihu.com/?target=http%3A//proceedings.mlr.press/v87/yang18b/yang18b.pdf)
- PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation [paper](https://arxiv.org/pdf/1612.00593.pdf)
- PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space [paper](https://arxiv.org/abs/1706.02413)
- IPOD: Intensive Point-based Object Detector for Point Cloud [paper](https://arxiv.org/abs/1812.05276v1)
- PIXOR: Real-time 3D Object Detection from Point Clouds [paper](http://www.cs.toronto.edu/~wenjie/papers/cvpr18/pixor.pdf)
- DepthCN: Vehicle Detection Using 3D-LIDAR and ConvNet [paper](https://www.baidu.com/link?url=EaE2zYjHkWvF33nsET2eNvbFGFu8-D3wWPia04uyKm95jMetHsSv3Zk-tODPGm5clsgCUgtVULsZ6IQqv0EYS_Z8El7Zzh57XzlJroSkaOuC8yv7r1XXL4bUrM2tWrTgjwqzfMV2tMTnFNbMOmHLTkUobgMg7HKoS6WW6PfQzkG&wd=&eqid=8f320cfa0005b878000000055e528b6d)
- Voxel-FPN: multi-scale voxel feature aggregation in 3D object detection from point clouds [paper](https://arxiv.org/ftp/arxiv/papers/1907/1907.05286.pdf)
- STD: Sparse-to-Dense 3D Object Detector for Point Cloud [paper](https://arxiv.org/abs/1907.10471)
- Fast Point R-CNN [paper](https://arxiv.org/abs/1908.02990)
- StarNet: Targeted Computation for Object Detection in Point Clouds [paper](https://arxiv.org/abs/1908.11069)
- Class-balanced Grouping and Sampling for Point Cloud 3D Object Detection [paper](https://arxiv.org/abs/1908.09492v1)
- LaserNet: An Efficient Probabilistic 3D Object Detector for Autonomous Driving [paper](https://arxiv.org/abs/1903.08701v1)
- FVNet: 3D Front-View Proposal Generation for Real-Time Object Detection from Point Clouds[ paper](https://arxiv.org/abs/1903.10750v1)
- Part-A^2 Net: 3D Part-Aware and Aggregation Neural Network for Object Detection from Point Cloud [paper](https://arxiv.org/abs/1907.03670v1)
- PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud [paper](https://arxiv.org/abs/1812.04244)
- Complex-YOLO: Real-time 3D Object Detection on Point Clouds [paper](https://arxiv.org/abs/1803.06199)
- YOLO4D: A ST Approach for RT Multi-object Detection and Classification from LiDAR Point Clouds [paper](https://github.com/Tom-Hardy-3D-Vision-Workshop/awesome-3D-object-detection/blob/master)
- YOLO3D: End-to-end real-time 3D Oriented Object Bounding Box Detection from LiDAR Point Cloud [paper](https://arxiv.org/abs/1808.02350)
- Monocular 3D Object Detection with Pseudo-LiDAR Point Cloud [paper](https://arxiv.org/pdf/1903.09847.pdf)
- Structure Aware Single-stage 3D Object Detection from Point Cloud（CVPR2020) [paper](http://openaccess.thecvf.com/content_CVPR_2020/html/He_Structure_Aware_Single-Stage_3D_Object_Detection_From_Point_Cloud_CVPR_2020_paper.html) [code](https://github.com/skyhehe123/SA-SSD)
- MLCVNet: Multi-Level Context VoteNet for 3D Object Detection（CVPR2020) [paper](https://arxiv.org/abs/2004.05679) [code](https://github.com/NUAAXQ/MLCVNet)
- 3DSSD: Point-based 3D Single Stage Object Detector（CVPR2020） [paper](https://arxiv.org/abs/2002.10187) [code](https://github.com/tomztyang/3DSSD)
- LiDAR-based Online 3D Video Object Detection with Graph-based Message Passing and Spatiotemporal Transformer Attention（CVPR2020） [paper](https://arxiv.org/abs/2004.01389) [code](https://github.com/yinjunbo/3DVID)
- PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection(CVPR2020) [paper](https://arxiv.org/abs/1912.13192) [code](https://github.com/sshaoshuai/PV-RCNN)
- Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud（CVPR2020） [paper](https://arxiv.org/abs/2003.01251) [code](https://github.com/WeijingShi/Point-GNN)
- MLCVNet: Multi-Level Context VoteNet for 3D Object Detection（CVPR2020） [paper](https://arxiv.org/pdf/2004.05679)
- Density Based Clustering for 3D Object Detection in Point Clouds（CVPR2020） [paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Ahmed_Density-Based_Clustering_for_3D_Object_Detection_in_Point_Clouds_CVPR_2020_paper.pdf)
- What You See is What You Get: Exploiting Visibility for 3D Object Detection（CVPR2020) [paper](https://arxiv.org/pdf/1912.04986.pdf)
- PointPainting: Sequential Fusion for 3D Object Detection(CVPR2020) [paper](https://arxiv.org/pdf/1911.10150.pdf)
- HVNet: Hybrid Voxel Network for LiDAR Based 3D Object Detection（CVPR2020) [paper](https://arxiv.org/pdf/2003.00186)
- LiDAR R-CNN: An Efficient and Universal 3D Object Detector（CVPR2021) [paper](https://arxiv.org/abs/2103.15297)
- Center-based 3D Object Detection and Tracking(CVPR2021) [paper](https://arxiv.org/abs/2006.11275)
- 3DIoUMatch: Leveraging IoU Prediction for Semi-Supervised 3D Object Detection(CVPR2021) [paper](https://arxiv.org/pdf/2012.04355.pdf)
- Embracing Single Stride 3D Object Detector with Sparse Transformer(CVPR2022) [paper](https://arxiv.org/pdf/2112.06375.pdf), [code](https://github.com/TuSimple/SST)
- Point Density-Aware Voxels for LiDAR 3D Object Detection(CVPR2022) [paper](https://arxiv.org/abs/2203.05662), [code](https://github.com/TRAILab/PDV)
- A Unified Query-based Paradigm for Point Cloud Understanding(CVPR2022) [paper](https://arxiv.org/abs/2203.01252#:~:text=Abstract%3A%203D%20point%20cloud%20understanding,including%20detection%2C%20segmentation%20and%20classification.)
- Beyond 3D Siamese Tracking: A Motion-Centric Paradigm for 3D Single Object Tracking in Point Clouds(CVPR2022) [paper](https://arxiv.org/abs/2203.01252#:~:text=Abstract%3A%203D%20point%20cloud%20understanding,including%20detection%2C%20segmentation%20and%20classification.), [code](https://github.com/Ghostish/Open3DSOT)
- Not All Points Are Equal: Learning Highly Efficient Point-based Detectors for 3D LiDAR Point Clouds(CVPR2022) [paper](https://arxiv.org/abs/2203.11139), [code](https://github.com/yifanzhang713/IA-SSD)
- Back To Reality: Weakly-supervised 3D Object Detection with Shape-guided Label Enhancement(CVPR2022) [paper](http://arxiv.org/abs/2203.05238), [code](https://github.com/xuxw98/BackToReality)
- Voxel Set Transformer: A Set-to-Set Approach to 3D Object Detection from Point Clouds(CVPR2022) [paper](https://www4.comp.polyu.edu.hk/~cslzhang/paper/VoxSeT_cvpr22.pdf), [code](https://github.com/skyhehe123/VoxSeT)
- BoxeR: Box-Attention for 2D and 3D Transformers(CVPR2022) [paper](https://arxiv.org/abs/2111.13087), [code](https://github.com/kienduynguyen/boxer), [中文介绍](https://mp.weixin.qq.com/s/UnUJJBwcAsRgz6TnQf_b7w)
- Canonical Voting: Towards Robust Oriented Bounding Box Detection in 3D Scenes(CVPR2022) [paper](https://arxiv.org/abs/2011.12001), [code](https://github.com/qq456cvb/CanonicalVoting)

## Competition Solution
## Engineering

## Survey
- 2021.04 Point-cloud based 3D object detection and classification methods for self-driving applications: A survey and taxonomy [paper](https://www.sciencedirect.com/science/article/abs/pii/S1566253520304097)
- 2021.07 3D Object Detection for Autonomous Driving: A Survey [paper](https://arxiv.org/abs/2106.10823)
- 2021.07 Multi-Modal 3D Object Detection in Autonomous Driving: a Survey [paper](https://arxiv.org/abs/2106.12735)
- 2021.10 A comprehensive survey of LIDAR-based 3D object detection methods with deep learning for autonomous driving [paper](https://www.sciencedirect.com/science/article/abs/pii/S0097849321001321)
- 2021.12 Deep Learning for 3D Point Clouds: A Survey [paper](https://ieeexplore.ieee.org/abstract/document/9127813)
## Book
- 3D Object Detection Algorithms Based on Lidar and Camera: Design and Simulation [book](https://www.amazon.com/Object-Detection-Algorithms-Based-Camera/dp/6200536538)
## Video
- Aivia online workshop: 3D object detection and tracking [video](https://www.youtube.com/watch?v=P0TrkwAdFYQ)
- 3D Object Retrieval 2021 workshop [video](https://3dor2021.github.io/programme.html)
- 3D Deep Learning Tutorial from SU lab at UCSD [video](https://www.youtube.com/watch?v=vfL6uJYFrp4)
- Lecture: Self-Driving Cars (Prof. Andreas Geiger, University of Tübingen) [video](https://www.youtube.com/watch?v=vfL6uJYFrp4)
- Current Approaches and Future Directions for Point Cloud Object (2021.04) [video](https://www.youtube.com/watch?v=xFFCQVwYeec)
- Latest 3D OBJECT DETECTION with 30+ FPS on CPU - MediaPipe and OpenCV Python (2021.05) [video](https://www.youtube.com/watch?v=f-Ibri14KMY)
- MIT autonomous driving seminar (2019.11) [video](https://space.bilibili.com/174493426/channel/series)
## Course
- [University of Toronto, csc2541](http://www.cs.toronto.edu/~urtasun/courses/CSC2541/06_3D_detection.pdf)
- [University of Tübingen, Self-Driving Cars](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/autonomous-vision/lectures/self-driving-cars/) *(Strong Recommendation)*
- [baidu](https://apollo.auto/devcenter/devcenter.html)
- [baidu-apollo](http://bit.baidu.com/Subject/index/id/16.html)
- [University of Toronto, coursera](https://www.coursera.org/specializations/self-driving-cars?ranMID=40328&ranEAID=9IqCvd3EEQc&ranSiteID=9IqCvd3EEQc-MlZGCwEU2294XsVYWDNwzw&siteID=9IqCvd3EEQc-MlZGCwEU2294XsVYWDNwzw&utm_content=10&utm_medium=partners&utm_source=linkshare&utm_campaign=9IqCvd3EEQc)

## Blog
- [Waymo Blog](https://blog.waymo.com/)
- [apollo介绍之Perception模块](https://zhuanlan.zhihu.com/p/142401769)
- [Apollo notes (Apollo学习笔记) - Apollo learning notes for beginners.](https://github.com/daohu527/Dig-into-Apollo#ledger-%E7%9B%AE%E5%BD%95)
- [PointNet系列论文解读](https://zhuanlan.zhihu.com/p/44809266)
- [Deep3dBox: 3D Bounding Box Estimation Using Deep Learning and Geometry](https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/deep3dbox.html)
- [SECOND算法解析](https://zhuanlan.zhihu.com/p/356892010)
- [PointRCNN深度解读](https://zhuanlan.zhihu.com/p/361973979)
- [Fast PointRCNN论文解读](https://zhuanlan.zhihu.com/p/363926237)
- [PointPillars论文和代码解析](https://zhuanlan.zhihu.com/p/357626425)
- [VoxelNet论文和代码解析](https://zhuanlan.zhihu.com/p/352419316)
- [CenterPoint源码分析](https://zhuanlan.zhihu.com/p/444447881)
- [PV-RCNN: 3D目标检测 Waymo挑战赛+KITTI榜 单模态第一算法](https://zhuanlan.zhihu.com/p/148942116)
- [LiDAR R-CNN：一种快速、通用的二阶段3D检测器](https://zhuanlan.zhihu.com/p/359800738)
- [混合体素网络（HVNet)](https://zhuanlan.zhihu.com/p/122426949)
- [自动驾驶感知| Range Image paper分享](https://zhuanlan.zhihu.com/p/420708905)
- [SST：单步长稀疏Transformer 3D物体检测器](https://zhuanlan.zhihu.com/p/476056546)
## Famous Research Group/Scholar
- [Naiyan Wang@Tusimple](https://scholar.google.com/citations?user=yAWtq6QAAAAJ&hl=en)
- [Hongsheng Li@CUHK](https://scholar.google.com/citations?user=BN2Ze-QAAAAJ&hl=en)
- [Oncel Tuzel@Apple](https://scholar.google.com/citations?user=Fe7NTe0AAAAJ&hl=en)
- [Oscar Beijbom@nuTonomy](https://scholar.google.com/citations?user=XP_Hxm4AAAAJ&hl=en)
- [Raquel Urtasun@University of Toronto](https://scholar.google.com/citations?user=jyxO2akAAAAJ&hl=en)
- [Philipp Krähenbühl@UT Austin](https://scholar.google.com/citations?hl=en&user=dzOd2hgAAAAJ&view_op=list_works&sortby=pubdate)
- [Deva Ramanan@CMU](https://scholar.google.com/citations?hl=en&user=9B8PoXUAAAAJ&view_op=list_works&sortby=pubdate)
- [Jiaya Jia@CUHK](https://jiaya.me/)
- [Thomas Funkhouser@princeton](https://www.cs.princeton.edu/~funk/)
- [Leonidas Guibas@Stanford](https://scholar.google.com/citations?hl=en&user=5JlEyTAAAAAJ&view_op=list_works&sortby=pubdate)
- [Steven Waslander@University of Toronto](https://www.trailab.utias.utoronto.ca/)
- [Ouais Alsharif@Google Brain](https://scholar.google.com/citations?hl=en&user=nFefEI8AAAAJ&view_op=list_works&sortby=pubdate)
- [Yuning CHAI(former)@waymo](https://scholar.google.com/citations?hl=en&user=i7U4YogAAAAJ&view_op=list_works&sortby=pubdate)
- [Yulan Guo@NUDT](http://yulanguo.me/)
- [Lei Zhang@The Hong Kong Polytechnic University](https://www4.comp.polyu.edu.hk/~cslzhang/)
## Famous CodeBase
- [Point Cloud Library (PCL)](https://github.com/PointCloudLibrary/pcl)
- [Spconv](https://github.com/traveller59/spconv)
- [Det3D](https://github.com/poodarchu/Det3D)
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- [Centerpoint](https://github.com/tianweiy/CenterPoint)
- [Apollo Auto - Baidu open autonomous driving platform](https://github.com/ApolloAuto)
- [AutoWare - The University of Tokyo autonomous driving platform](https://www.autoware.org/)

## Famous Toolkit
- [ZED Box](https://www.stereolabs.com/docs/object-detection/)

# Acknowlegement
[Awesome System for Machine Learning](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning)

[awesome-3D-object-detection](https://github.com/Tom-Hardy-3D-Vision-Workshop/awesome-3D-object-detection)

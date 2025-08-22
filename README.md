# TrackID3x3: A Dataset for 3x3 Basketball Player Tracking and Identification

This project provides 3x3 fixed camera footage and a baseline for extracting tracking data from them, and has been accepted for MMSports'25. The paper can be read [here](https://arxiv.org/html/2503.18282v2).


## TrackID3x3 dataset 

The dataset consists of three subsets: Indoor, captured by indoor fixed cameras; Outdoor, captured by outdoor fixed cameras; and Drone, captured by outdoor drone cameras.
The videos have bounding boxes of 6 on-court players for all frames and 10 pose keypoints of each player for some frames only.
The video files and some intermediate product of proposed baseline can be downloaded [here](https://drive.google.com/drive/folders/1aWqMwQKr5xKMjqms7-raYluSlxPsGvwX).

<!-- GIF EMBEDS START -->
![Indoor](videos/gif/Indoor.gif)
![Outdoor](videos/gif/Outdoor.gif)
![Drone](videos/gif/Drone.gif)
<!-- GIF EMBEDS END -->

## Track-ID task
This task is a simplified version of Game State Reconstruction and its objective is to extract player tracking data from fixed camera video. In 3x3, only the localization of the court is done manually in this task, the rest is done automatically, and the location and identification of 6 on-court players are extracted.

<!-- GIF EMBEDS START -->
![Indoor_minimap_drawn](videos/gif/Indoor_minimap_drawn.gif)
![Outdoor](videos/gif/Outdoor_minimap_drawn.gif)
<!-- GIF EMBEDS END -->

## Track-ID Baseline

### Installation & Setup

This repository incorporates the following three external projects with modifications.  
(Please refer to the README of each project for detailed configuration and usage.)

- [CAMELTrack](https://github.com/TrackingLaboratory/CAMELTrack)  
- [jersey-number-pipeline](https://github.com/mkoshkina/jersey-number-pipeline)  
- [BoT-SORT](https://github.com/NirAharon/BoT-SORT)  

---

**Step 1: Clone this repository**

```bash
git clone https://github.com/your_username/TrackID3x3.git
cd TrackID3x3
```


**Step 2: Setup of each sub-project**

*Recommendation: Create a Python 3.10 virtual environment for each sub-project and install the necessary libraries.*
- CAMELTrack: Please install libraries according to corresponding README.
- jersey-number-pipeline: Please install jersey-number-pipeline/requirement.txt and follow the instructions in the corresponding README to download the model weights (in the paper, the PARSeq and LegibilityClassifier weights were trained in Hockey)
- BoT-SORT: Please download the YOLOX weights from the [ByteTrack repository](https://github.com/FoundationVision/ByteTrack) and place them in the pretrained directory (the paper reports results using the bytetrack_x_mot17.pth.tar).



### Execution 
- CAMELTrack:
  -  Execute run.sh in CAMELTrack directory
      - run.sh is for execution for multiple videos
      - By default, YOLOX built into BoT-SORT is used for person detection
      - The execution results at the baseline in the paper can be downloaded from [here](https://drive.google.com/drive/folders/1aWqMwQKr5xKMjqms7-raYluSlxPsGvwX).
- Rule-based Heuristic Methods:
  - Execute script/Indoor/rule-based_heuristic_methods.ipynb or script/Outdoor/rule-based_heuristic_methods.ipynb 
  - By default, processing is performed on output/CAMELTrack_outputs (i.e., those that can be downloaded from Google Drive).
- jersey-number-pipeline (Only Outdoor):
  - Execute run.sh in jersey-number-pipeline directory
  - Note that each tracklet's bbox cropped .png file must be organized into a directory for each video in jersey-number-pipeline/data/Outdoor/test/images. (e.g., jersey-number-pipeline/data/Outdoor/test/images/IMG_0104_1/track1/IMG_0104_1_frame1_track1_cropped.png).
    - You can crop using script/Outdoor/crop_images.py
  - The execution results (except .png files) in the paper can be downloaded from [here](https://drive.google.com/drive/folders/1aWqMwQKr5xKMjqms7-raYluSlxPsGvwX)
- Attributes Identification:
  - Indoor:
    - Just run script/Indoor/add_attributes_pred.ipynb
  - Outdoor:
    - First, run script/Outdoor/organization_torso_images.py to organize the images of the torso region.
    - Next, run script/Outdoor/calculate_color_hist.py to calculate the color histogram for each tracklet.
      - the .npy files containing the color histograms in the paper can be downloaded from [here](https://drive.google.com/drive/folders/1aWqMwQKr5xKMjqms7-raYluSlxPsGvwX)
    - Finally, use the calculated color histogram and final_results.json (output from jersey-number-pipeline) to run script/Outdoor/add_attributes.ipynb
- Calculation of TI-HOTA:
  - Just run script/compute_TI-HOTA.ipynb




## Citation

If you use this repository for your research or wish to refer to our contributions, please cite the following paper:

[**Analyzing coordinated group behavior through role-sharing: a pilot study in female 3-on-3 basketball with practical application**](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1513982/full)

[**Enhanced Multi-Object Tracking Using Pose-based Virtual Markers in 3x3 Basketball**](https://arxiv.org/abs/2412.06258)

[**TrackID3x3: A Dataset for 3x3 Basketball Player Tracking and Identification**](https://arxiv.org/abs/2503.18282)

```bibtex
@article{ichikawa2024analyzing,
  title={Analyzing coordinated group behavior through role-sharing: A pilot study in female 3-on-3 basketball with practical application},
  author={Ichikawa, Jun and Yamada, Masatoshi and Fujii, Keisuke},
  journal={Frontiers in Sports and Active Living},
  volume={7},
  pages={1513982},
  year={2024},
  publisher={Frontiers}
}
@article{yin2024enhanced,
  title={Enhanced Multi-Object Tracking Using Pose-based Virtual Markers in 3x3 Basketball},
  author={Yin, Li and Yeung, Calvin and Hu, Qingrui and Ichikawa, Jun and Azechi, Hirotsugu and Takahashi, Susumu and Fujii, Keisuke},
  journal={arXiv preprint arXiv:2412.06258},
  year={2024}
}
@article{yamada2025trackid3x3,
  title={TrackID3x3: A Dataset for 3x3 Basketball Player Tracking and Identification},
  author={Yamada, Kazuhiro and Yin, Li and Hu, Qingrui and Ding, Ning and Iwashita, Shunsuke and Ichikawa, Jun and Kotani, Kiwamu and Yeung, Calvin and Fujii, Keisuke},
  journal={arXiv preprint arXiv:2503.18282v2},
  year={2025}
}
```



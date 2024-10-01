# Eye Socket Recognition

This repository contains the code implementation for the paper titled "[Enhancing Eye Socket Recognition Performance using Inverse Histogram Fusion Images in Gabor Transform](https://onlinelibrary.wiley.com/doi/10.4218/etrij.2023-0395)" authored by Harisu Abdullahi Shehu, Ibrahim Furkan Ince, and Faruk Bulut.


## Overview

The paper presents a novel approach to improve eye socket recognition performance by leveraging inverse histogram fusion images in Gabor transform. The code provided here includes Java implementations (`EyeSocketRecognition.pde`, `GaborFeature.pde`, and `Utils.pde`) for eye socket recognition and Gabor feature extraction. These features are then utilized by our proposed method, implemented in Python (`model.py`), for accurate eye socket recognition.

Additionally, we provide code (`CK_processed`) for pre-processing the CK+ dataset. Moreover, we conduct comparative analyses with other filter-based approaches, including wavelength-based filtering, gabor filter, guided filter, and bilateral filter (`wavelength_basedFilter.py`, `EyeSocketRecognition.pde`, `guidedFilter.py`, and `bilateralFilter.py`). For evaluating the proposed method against deep learning methods, we provide implementations of VGG19, InceptionV3, ResNet50, Face Mesh Deep Neural Network, and Multi-task Neural Network (`VGG19.py`, `InceptionV3.py`, `ResNet50.py`, `FaceMesh_DNN.py`, and `MTN.py`).

For visualizing the histogram obtained before and after equalization, we provide the script `visual_histogram_plusequalization.py`.

## Datasets

Researchers can also access the datasets used in our study:
- [CK+](http://www.jeffcohn.net/Resources/)
- [MaskedAT&T](https://data.mendeley.com/datasets/v992cb6bw7/6) 
- [Flickr30](https://www.flickr.com/photos/thefacewemake/albums)
- [BioID](https://www.bioid.com/About/BioID-Face-Database)

## Citation

If you utilize our code or findings, please cite our paper:

```bash
@article{shehu2024enhancing,
  title={Enhancing Eye Socket Recognition Performance using Inverse Histogram Fusion Images in Gabor Transform},
  author={Shehu, Harisu Abdullahi and Ince, Ibrahim Furkan and Bulut, Faruk},
  journal={Etri journal},
  year={2024},
  doi={10.4218/etrij.2023-0395}
}
```

For any inquiries, please contact Harisu Abdullahi Shehu at harisu.shehu@ecs.vuw.ac.nz

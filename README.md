<div align="center">

<!-- 🌈 Top Banner -->
<div style="background: linear-gradient(135deg, #0f766e 0%, #0891b2 50%, #1e40af 100%); padding: 45px 30px; border-radius: 20px; color: white; margin-bottom: 25px; box-shadow: 0 10px 40px rgba(8,145,178,0.3);">

# 🛰️ SFSNet

### 🌐 Scale-aware Fusion and Spatial-Frequency Collaborative Network

**Remote Sensing Imagery Semantic Segmentation**

<br/>

<table style="margin: 0 auto;">
<tr>
<td align="center">
<br/>

🔍<br/>**Small-Scale Targets**<br/><sub>Shadow-robust detection</sub>

</td>
<td align="center">
<br/>

📐<br/>**Scale-Varying Regions**<br/><sub>Multi-scale weighting</sub>

</td>
<td align="center">
<br/>

✂️<br/>**Complex Boundaries**<br/><sub>Low-contrast refinement</sub>

</td>
<td align="center">
<br/>

🌊<br/>**Spatial-Frequency**<br/><sub>FFT collaboration</sub>

</td>
</tr>
</table>

</div>

<!-- 🏷️ Badges -->

![Journal](https://img.shields.io/badge/IEEE-JSTARS-00629B?logo=ieee&logoColor=white)
![Status](https://img.shields.io/badge/Status-Published-brightgreen)
![Topic](https://img.shields.io/badge/Remote%20Sensing-Semantic%20Segmentation-orange?logo=NASA&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Backbone](https://img.shields.io/badge/Backbone-ResNet--18-626262?logo=meta&logoColor=white)
![FFT](https://img.shields.io/badge/Fourier-Transform-7C3AED?logo=scipy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

</div>

---

## 📰 Paper Information

| | |
|:---:|:---|
| 📄 **Title** | Scale-aware Fusion and Spatial-Frequency Collaborative Network for Remote Sensing Imagery Semantic Segmentation |
| 🏛️ **Journal** | IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (**JSTARS**) |
| 📅 **Year** | 2026 |
| 👥 **Authors** | Lu Wang, Chenxu Lou, Jing Wang, Junbo Yu, Lihua Zhao, Wenbin Wang, Dong Ren |

---

## 🎯 Motivation

High-resolution remote sensing imagery presents unique challenges for semantic segmentation: objects exhibit **drastic scale variations**, **small targets are easily obscured by shadows**, and **boundaries are winding with low contrast** between foreground and background. As shown below, traditional methods suffer from weight dispersion in scale-varying regions and struggle to focus near complex boundaries.

<div align="center">

![Motivation](https://aka.doubaocdn.com/s/SoX4CChr3p)

<i>Fig. 1. Motivation visualization. Compared with traditional methods, SFSNet achieves more focused attention on small-scale shadow-obscured targets, better weighting in scale-varying regions, and refined boundary attention in low-contrast areas.</i>

</div>

### 🔍 Key Observations

- 🚗 **(a) Small-scale targets & shadows**: Cars and small objects are easily obscured by building shadows. Traditional methods produce dispersed attention weights, while SFSNet focuses precisely on target regions.
- 🌳 **(b) Scale-varying regions**: Trees and vegetation show drastic scale changes across the scene. Traditional methods spread attention evenly; SFSNet adaptively weights scale-change regions.
- 🏢 **(c) Complex & low-contrast boundaries**: Buildings have winding boundaries with low contrast against the background. Traditional methods mix weights near boundaries; SFSNet enhances boundary-focused attention.

---

## 🏗️ Method Overview

SFSNet addresses these challenges through two core designs:

### 🌊 Spatial-Channel Enhanced Attention Module (SCEAM)

- **Low-frequency path**: A Non-Local block with Adaptive Low-Pass Filtering (ALPF) captures long-range spatial dependencies.
- **High-frequency path**: A Fourier Unit performs FFT-based spatial and channel attention, emphasizing high-frequency details such as boundaries and textures.
- **Fusion**: The low- and high-frequency branches are combined to achieve complementary spatial-frequency collaboration.

### 📐 Scale-aware Fusion Module (SAFM)

- **CAIF**: Channel Attention with Information Fusion, leveraging both standard deviation and average pooling for robust channel weighting.
- **RSSC-CSCF**: Multi-dilated residual feature fusion to capture multi-scale context at receptive fields of 1, 2, 3, and 3 (dilated).
- Progressive encoder-decoder interaction for refined multi-scale feature aggregation.

---

## 📁 Repository Structure

```
SFSNet/
├── 📄 README.md          # This file
├── 📄 LICENSE            # MIT License
└── 🧠 model/
    └── SFSNet.py         # SFSNet model implementation
```

---

## 🙏 Acknowledgments

This project is built upon the excellent **[GeoSeg](https://github.com/WangLibo1995/GeoSeg)** framework. We sincerely thank the authors for providing a well-structured and reproducible remote sensing segmentation codebase, which greatly facilitated our implementation and experiments.

> **GeoSeg**: A general remote sensing semantic segmentation repository by Libo Wang. [[GitHub]](https://github.com/WangLibo1995/GeoSeg)

---

## 📝 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{wang2026sfsnet,
  title={Scale-aware Fusion and Spatial-Frequency Collaborative Network for Remote Sensing Imagery Semantic Segmentation},
  author={Wang, Lu and Lou, Chenxu and Wang, Jing and Yu, Junbo and Zhao, Lihua and Wang, Wenbin and Ren, Dong},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2026},
  publisher={IEEE}
}
```

---

<div align="center">

### 🌟 If this repo helps your research, please give it a star! 🌟

<br/>

<sub>Built with ❤️ using PyTorch · GeoSeg · Fourier Transform</sub>

</div>

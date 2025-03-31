<p align="center">
<h1 align="center"<strong>TokenHSI: Unified Synthesis of Physical Human-Scene Interactions through Task Tokenization</strong></h1>
  <p align="center">
    <a href='https://liangpan99.github.io/' target='_blank'>Liang Pan</a><sup>1,2</sup>&emsp;
    <a href='https://zeshiyang.github.io/' target='_blank'>Zeshi Yang</a> <sup>3</sup>&emsp;
    <a href='https://frank-zy-dou.github.io/' target='_blank'>Zhiyang Dou</a><sup>2</sup>&emsp;
    <a href='https://wenjiawang0312.github.io/' target='_blank'>Wenjia Wang</a><sup>2</sup>&emsp;
    <a href='https://www.buzhenhuang.com/about/' target='_blank'>Buzhen Huang</a><sup>4</sup>&emsp;
    <br>
    <a href='https://scholar.google.com/citations?user=KNWTvgEAAAAJ&hl=en' target='_blank'>Bo Dai</a><sup>2,5</sup>&emsp;
    <a href='https://i.cs.hku.hk/~taku/' target='_blank'>Taku Komura</a><sup>2</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=GStTsxAAAAAJ&hl=en&oi=ao' target='_blank'>Jingbo Wang</a><sup>1</sup>&emsp;
    <br>
    <sup>1</sup>Shanghai AI Lab <sup>2</sup>The University of Hong Kong <sup>3</sup>Independent Researcher <sup>4</sup>Southeast University <sup>5</sup>Feeling AI
    <br>
    <strong>CVPR 2025</strong>
  </p>
</p>
<p align="center">
  <a href='https://arxiv.org/abs/2503.19901'>
    <img src='https://img.shields.io/badge/Arxiv-2502.20390-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a>
  <a href='https://arxiv.org/pdf/2503.19901'>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'></a>
  <a href='https://liangpan99.github.io/TokenHSI/'>
    <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green'></a>
</p>

## 🏠 About
<div style="text-align: center;">
    <img src="https://github.com/liangpan99/TokenHSI/blob/page/static/images/teaser.png" width=100% >
</div>
Introducing TokenHSI, a unified model that enables physics-based characters to perform diverse human-scene interaction tasks. It excels at seamlessly unifying multiple <b>foundational HSI skills</b> within a single transformer network and flexibly adapting learned skills to <b>challenging new tasks</b>, including skill composition, object/terrain shape variation, and long-horizon task completion.
</br>

## 📹 Demo
<p align="center">
    <img src="assets/longterm_demo_isaacgym.gif" align="center" width=60% >
    <br>
    Long-horizon Task Completion in a Complex Dynamic Environment
</p>

<!-- ## 🕹 Pipeline
<div style="text-align: center;">
    <img src="https://github.com/liangpan99/TokenHSI/blob/page/static/images/pipeline.jpg" width=100% >
</div> -->

## 🔥 News  
- **[2025-03-31]** We've released the codebase and checkpoint for the foundational skill learning part.

## 📝 TODO List  
- [x] Release foundational skill learning 
- [ ] Release policy adaptation - skill composition  
- [ ] Release policy adaptation - object/terrain shape variation
- [ ] Release policy adaptation - long-horizon task completion

## 📖 Getting Started

coming soon

## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@inproceedings{pan2025tokenhsi,
  title={TokenHSI: Unified Synthesis of Physical Human-Scene Interactions through Task Tokenization},
  author={Pan, Liang and Yang, Zeshi and Dou, Zhiyang and Wang, Wenjia and Huang, Buzhen and Dai, Bo and Komura, Taku and Wang, Jingbo},
  booktitle={CVPR},
  year={2025},
}

@inproceedings{pan2024synthesizing,
  title={Synthesizing physically plausible human motions in 3d scenes},
  author={Pan, Liang and Wang, Jingbo and Huang, Buzhen and Zhang, Junyu and Wang, Haofan and Tang, Xu and Wang, Yangang},
  booktitle={2024 International Conference on 3D Vision (3DV)},
  pages={1498--1507},
  year={2024},
  organization={IEEE}
}
```

Please also consider citing the following papers that inspired TokenHSI.

```bibtex
@article{tessler2024maskedmimic,
  title={Maskedmimic: Unified physics-based character control through masked motion inpainting},
  author={Tessler, Chen and Guo, Yunrong and Nabati, Ofir and Chechik, Gal and Peng, Xue Bin},
  journal={ACM Transactions on Graphics (TOG)},
  volume={43},
  number={6},
  pages={1--21},
  year={2024},
  publisher={ACM New York, NY, USA}
}

@article{he2024hover,
  title={Hover: Versatile neural whole-body controller for humanoid robots},
  author={He, Tairan and Xiao, Wenli and Lin, Toru and Luo, Zhengyi and Xu, Zhenjia and Jiang, Zhenyu and Kautz, Jan and Liu, Changliu and Shi, Guanya and Wang, Xiaolong and others},
  journal={arXiv preprint arXiv:2410.21229},
  year={2024}
}
```

## 👏 Acknowledgements and 📚 License

This repository builds upon the following awesome open-source projects:

- [ASE](https://github.com/nv-tlabs/ASE): Contributes to the physics-based character control codebase  
- [Pacer](https://github.com/nv-tlabs/pacer): Contributes to the procedural terrain generation and trajectory following task
- [rl_games](https://github.com/Denys88/rl_games): Contributes to the reinforcement learning code  
- [OMOMO](https://github.com/lijiaman/omomo_release)/[SAMP](https://samp.is.tue.mpg.de/InterDiff)/[AMASS](https://amass.is.tue.mpg.de/)/[3D-Front](https://arxiv.org/abs/2011.09127): Used for the reference dataset construction
- [InterMimic](https://github.com/Sirui-Xu/InterMimic): Used for the github repo readme design 

This codebase is released under the [MIT License](LICENSE).  
Please note that it also relies on external libraries and datasets, each of which may be subject to their own licenses and terms of use.

## 🌟 Star History

<p align="center">
    <a href="https://www.star-history.com/#liangpan99/TokenHSI&Date" target="_blank">
        <img width="500" src="https://api.star-history.com/svg?repos=liangpan99/TokenHSI&type=Date" alt="Star History Chart">
    </a>
<p>


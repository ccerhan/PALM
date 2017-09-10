## Patterns of Approximated Localised Moments
PALM is a histogram based whole image descriptor that allows reliable and real-time loop closure detection or visual place recognition in environments with visually similar places such as forests, parks, office corridors. This method relies on computing approximated local Zernike moments across the image. When computed locally, Zernike moments provide a high discrimination ability, which enables the distinguishing of similar-looking places. The method is also fast to compute (~1.4 ms for 320x320 sized image) and gives promising results in terms of global image matching.

**Dependencies:** OpenCV (core and imgproc). Tested with 3.2, should work with 3.2+.

Please contact **ccerhan[at]gmail.com** for any questions and comments, bug reports etc.

**If you intend to use this code for research purposes, please cite:**

[Patterns of approximated localised moments for visual loop closure detection](http://ieeexplore.ieee.org/document/7885256/)

*Can Erhan, Evangelos Sariyanidi, Onur Sencan, Hakan Temeltas*

```
@article{erhan2017iet,
    author = {Erhan, Can and Sariyanidi, Evangelos and Sencan, Onur and Temeltas, Hakan},
    title = {Patterns of approximated localised moments for visual loop closure detection},
    journal = {IET Computer Vision}, 
    year = {2017}, 
    volume = {11}, 
    number = {3}, 
    pages = {237-245},
    doi = {10.1049/iet-cvi.2016.0237}, 
    ISSN = {1751-9632}
}
```

[Efficient visual loop closure detection in different times of day](http://www.ingentaconnect.com/contentone/ist/ei/2017/00002017/00000009/art00002)

*Can Erhan, Evangelos Sariyanidi, Onur Sencan, Hakan Temeltas*

```
@article{erhan2017ie,
    author = {Erhan, Can and Sariyanidi, Evangelos and Sencan, Onur and Temeltas, Hakan},
    title = {Efficient visual loop closure detection in different times of day},
    journal = {Electronic Imaging}, 
    year = {2017}, 
    volume = {2017}, 
    number = {9}, 
    pages = {5-9},
    doi = {10.2352/issn.2470-1173.2017.9.iriacv-258}
}
```

## How to Use

1. Clone the repository
2. Copy **PALM** folder to your project
3. Example usage is located in **main.cpp** file

*More detailed documentation will be added soon...*

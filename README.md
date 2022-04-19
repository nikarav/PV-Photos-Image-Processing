# PV-Photos-Image-Processing

The purpose of this project is to read solar panels scan photos and process them for evaluation. T

## Team 

|  Name | Github  | 
|---|---|
|  Morten Bjerre | [MortenBjerre](https://github.com/MortenBjerre)  |
|  Rasim Deniz |  [Rasim-Deniz](https://github.com/Rasim-Deniz) |
|  Karavasilis Nikolaos | [nikarav](https://github.com/nikarav)  |

## Data 
The photos were taken with an infrared camera. Due to **NDA** aggreement the photos are not included.

## Methodology

The first step is to detect centralized cells inside each panel frame. We filter away any quadrilaterals that are not almost perfectly square. 

Afterwards, we create a mask based on the first quadrilateral, which works as a trust region between the current and next frame. Then, we find matching interest points. The homography can be applied directly to the old quadrilateral, and now we have a matching quadrilaretal in the new frame. We convert the quadrilaterals to rectangular images/matrices and stuff them in to a tensor.


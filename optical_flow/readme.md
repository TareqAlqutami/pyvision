# Optical Flow

## Background
Optical flow is the pattern of apparent motion of image objects between two consecutive frames caused by the movement of object or camera. It is 2D vector field where each vector is a displacement vector showing the movement of points from first frame to second. 
Optical flow generally works on several assumptions such as:
- The pixel intensities of an object do not change between consecutive frames.
- Neighbouring pixels have similar motion.

There are generally two types of optical flow:
### Sparse optical flow
These approaches calculate motion vector for specific set of objects/features such as tracking corner features over the video frames. Famous algorithms are Lucas-Kanade algorithm.

#### Lucas-Kanade method
It assumes that all the neighbouring pixels will have similar motion. It takes a window of pixels around a point then solves the over-determined optical flow equations using least square fitting. The points to track are typically corner features using Shi-Tomasi corner detector.

In order to deal with a large motion, pyramids are used. When we go up in the pyramid, small motions are removed and large motions become small motions. So by applying Lucas-Kanade there, we get optical flow along with the scale.

### Dense Optical flow: 
These approaches calculate motion vector for each pixel in the image. Popular algorihtms are Farneback and RLOF.


## Install dependencies
```
pip install -r ..\requirements.txt
```

## Running the code

### Sparse optical flow
Simple sparse optical flow using  **Lucas-Kanade** algorithm is in  `sparse.py`. It can be run using:
```bash
python sparse.py --video_src <video source>
```
- `video_src` (optional): points to the video source, defaults to camera 0 if not specified. It accepts any source supported by opencv VideoCapture.

### Dense optical flow
all the code related to sparse optical flow is in `dense.py`. It can be run using:
```
python dense.py --algorithm <algorithm> --video_src <video source>
```
- `video_src` (optional): points to the video source, defaults to camera 0 if not specified. It accepts any source supported by opencv VideoCapture.
- `algorithm`(optional): dense optical flow algorithm. supported algorithms are
  -  `farneback` (default)
  -  `lucas-kanade`
  -  `rlof`

### Sparse optical flow tracking
A more practical implementation of Lucas-Kanade sparse optical flow with tracking.
It uses goodFeaturesToTrack for track initialization and back-tracking for match verification
between frames. Run using:
```
python sparse_track.py --video_src <video source>
```
- `video_src` (optional): points to the video source, defaults to camera 0 if not specified. It accepts any source supported by opencv VideoCapture.


# Screenshots
Running using the video demo demo.mp4
```
python dense.py --algorithm lucas-kanade  --video_src ..\demo.mp4
```
- To on/off HSV visualization press 1
- To exit press esc

**Dense Lucas-kanade** 
![lk_dense](screenshots\lk_dense.png)


**Sparse Lucas-Kanade with Tracking**
![lk_track](screenshots\lk_track.png)

## References
- https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
# Player Speed Project
## *Goal is to calculate player speed from NBA gameplay footage*

### Loose Plan:
- Train ConvNet to identify at least 4 keypoints in each NBA frame
- Use player detection and tracking to draw bounding box around each player in a given frame

- Use ConvNet to identify keypoints in each framework
- Use homography to map keypoints in frame to keypoints on a fixed, eagle-eye perspective of NBA court and develop coordinate plane
- Use mapping to map bottom of player bounding box to 2D coordinates on NBA court

# Instantaneous-Frames-Network
This project is for video-based gaze estimation of instantaneous frames network.
## Requirements
**The code is built with following libraries:**

- PyTorch 1.0 or higher
- TensorboardX
- tqdm
- scikit-learn
- pandas
- Python 3.x
## Training
To train the model(s) in the paper, run this command:
```javascript
python main.py <train_video_path> \
     --arch <resnet-backbone> \
     --lr 0.002 --lr_steps 40 70 --epochs 100 \
     --batch-size 128 --dropout 0.3 --consensus_type=avg --npb
```
The results of the training are saved as a .pth file format.

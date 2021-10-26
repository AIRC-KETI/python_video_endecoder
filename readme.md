## 1. Reinstall encoder
```shell
cd encoder
python3 setup.py install
```

## 2. Reinstall decoder
```shell
cd decoder
python3 setup.py install
```

## Install dep

### On Ubuntu (Debian)

```bash
sudo apt-get install -y pkg-config libavutil-dev libavcodec-dev libavformat-dev libswscale-dev
```

### On Max OS X
```bash
brew install ffmpeg pkg-config
```

### from source

```bash
# for python 3
bash install_dep.sh
```

## Sample code

```python
import cv2

import videoencoder
import videodecoder

width, height = 640, 480
# video
cap = cv2.VideoCapture(0)

bitrate = 1000000 # 1Mbps
iframeinterval = 30 # I frame interval
framerate = 30 # fps

# create codec for encoder
enc_codec = videoencoder.create_codec("h264", "bgr24", width, height, bitrate, iframeinterval, framerate)
# create codec for decoder
dec_codec = videodecoder.create_codec("h264", "bgr24")


try:
    dec_out = None
    while True:
        # grab frame
        ret, frame = cap.read()
        
        # push frame to the enc_codec
        enc_out = videoencoder.push_frame_data(enc_codec, frame.tostring())

        if enc_out is not None:
            # push encoded frame to the dec_codec
            dec_out = videodecoder.push_frame_data(dec_codec, enc_out)
        
        cv2.imshow('origin',frame)
        if dec_out is not None:
            cv2.imshow('decoded',dec_out)
        key = cv2.waitKey(int(1000/framerate))
except KeyboardInterrupt:
    exit()
```



import pickle
import cv2
import numpy as np
from PIL import Image
import io

# 1) Load the .bin file
with open('/home/arjun/Downloads/AdaDLProject/AdaDistill/eval/forehead_verification.bin', 'rb') as f:
    bins, issame_list = pickle.load(f)

# 2) Check how many pairs
print(f"Total pairs: {len(issame_list)}")

# 3) Decode the first image of the first pair
img_bytes = bins[0]                     # first image bytes
arr = np.frombuffer(img_bytes, np.uint8) 
img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
cv2.imshow('First image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#—or with PIL in a notebook—
pil_img = Image.open(io.BytesIO(img_bytes))
display(pil_img)

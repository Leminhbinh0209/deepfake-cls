### **Deepfake type classification**
1. **Step 1:** Face extractor by `dlid`
```
python face_extractor.py --video /path/to/to/video --output-folder /path/to/save/faces
```
*For example:*
```
python face_extractor.py --video ./data/YouTube/Trump.mp4 --output-folder ./data/faces/Trump
```
2. **Step 2:** Classify type of deepfake.

Download pre-trained weight [here](https://o365skku-my.sharepoint.com/:u:/g/personal/bmle_o365_skku_edu/EQ1ZknfHPlRElJGQ2BERdIwB_jcOYT3WNO3q4EXe5YZjuw?e=rMNV0f).

Currently, upon 5 types of Deepfake from FaceForensics++: NeuralTexture, Deepfake, Face2Face, FaceSwap, FaceShifter.
```
python test.py --model-name modelname
				--model-weight /path/to/pretrained/model
				--target /path/to/image/or/folder/

```
*For example:*
```
python test.py --model-name resnet50
				--model-weight weights/train_img224_shuflfe_patient18_raw_c23_batch192_cutmix_best.pth
				--target ./data/faces/Trump/

```
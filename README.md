# Introduction 
This repo contains the demo code for OCR on Signboard images from the ArT dataset using PaddleOCR' EAST + CRNN, MMOCR's ASTER and CRAFT.
Google Colab is used for this demonstration.

# Code Structure/Explanation
1.	[Setup on Google Colab](/Signboard_Setup.ipynb) or ➡️[Click here](https://colab.research.google.com/github/chunchet-ng/signboard_ocr/blob/main/Signboard_Setup.ipynb) to kickstart this demo directly at Google Colab
2.	[HMean Calculation for Text Detection & Spotting Tasks](/Detection_Evaluation/HMean.ipynb)
3.	[Accuracy & Normalized Edit Distance Calculation for Text Recognition Task](/Recognition_Evaluation/Norm_Edit_Distance.ipynb)
4.	[Data Exploration for the ArT dataset](/Signboard_OCR/ArT.ipynb)
5.	[Text Detection, Recognition & Spotting on the ArT dataset](/Signboard_OCR/Signboard_OCR.ipynb)

>Do note that you need to run the ArT notebook before going into Signboard_OCR for the first time.

# Takeaways
1.	You will learn how the HMean is calculated for the text detection and text spotting tasks with detailed examples.
2.	You will learn how the accuracy, number of correctly recognized words, and normalized edit distance are calculated for the text recognition task.
3.	You will learn how to make use of the ArT, a commonly used scene text dataset. We will compare the results of using pre-trained regular and irregular scene text methods using the evaluation methods mentioned above. Then, we can analyze the performance gap between the regular and irregular scene text methods on signboard images.

# Credits
1.	[ArT dataset](https://rrc.cvc.uab.es/?ch=14)
2.	[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
3.  [MMOCR](https://github.com/open-mmlab/mmocr)
4.  [CRAFT](https://github.com/clovaai/CRAFT-pytorch)
4.  Huge thanks to [@nwjun](https://github.com/nwjun) and [@alex](https://github.com/AlexofNTU) for making the notebooks better and provide valuable feedbacks to this demo! 💪😇👍

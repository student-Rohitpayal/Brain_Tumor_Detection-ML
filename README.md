# Brain Tumor Detection

A brain tumor is an abnormal growth of tissue in the brain or central spine that can disrupt proper brain function. It is the abnormal growth of tissues in brain. If the tumor originates in the brain, it is called a primary brain tumor. Primary brain tumors can be benign or malignant. Benign brain tumors are not cancerous.


## Requirements

Step 1: Make sure your virtual environment is activated

            -> source venv/bin/activate  # On Linux/Mac
            
            -> cd C:\path\to\Brain-Tumor-Detection-master # On Windows
            -> venv\Scripts\Activate.ps1

Step 2: Then install
            -> pip install -r requirements.txt

## Tumor Detection
The GUI can be used to detect and view the tumor region.

The tensorflow model can be used to detect if the MRI image contains tumor or not.

![alt text](tumordetection.jpg)

The tumor region can be viewed using Image processing methods applied through opencv. Image segmentation using marker-based watershed segmentation algorithm is used to view the tumor region. A watershed is a transformation defined on a grayscale image. We label the region which we are sure of being the foreground or object with one color(or intensity), and label the region which we are sure of being background or non-object with another color and finally the region which we are not sure of anything, we label it with 0. That is our marker. Then apply watershed algorithm. Then our marker will be updated with the labels we gave, and the boundaries of objects will have a value of -1.


✔️ Abstract

A Brain tumor is considered as one of the aggressive diseases, among children and adults. Brain tumors account for 85 to 90 percent of all primary Central Nervous System(CNS) tumors. Every year, around 11,700 people are diagnosed with a brain tumor. The 5-year survival rate for people with a cancerous brain or CNS tumor is approximately 34 percent for men and36 percent for women. Brain Tumors are classified as: Benign Tumor, Malignant Tumor, Pituitary Tumor, etc. Proper treatment, planning, and accurate diagnostics should be implemented to improve the life expectancy of the patients. The best technique to detect brain tumors is Magnetic Resonance Imaging (MRI). A huge amount of image data is generated through the scans. These images are examined by the radiologist. A manual examination can be error-prone due to the level of complexities involved in brain tumors and their properties.
Application of automated classification techniques using Machine Learning(ML) and Artificial Intelligence(AI)has consistently shown higher accuracy than manual classification. Hence, proposing a system performing detection and classification by using Deep Learning Algorithms using Convolution-Neural Network (CNN), Artificial Neural Network (ANN), and Transfer-Learning (TL) would be helpful to doctors all around the world.

✔️ Context

Brain Tumors are complex. There are a lot of abnormalities in the sizes and location of the brain tumor(s). This makes it really difficult for complete understanding of the nature of the tumor. Also, a professional Neurosurgeon is required for MRI analysis. Often times in developing countries the lack of skillful doctors and lack of knowledge about tumors makes it really challenging and time-consuming to generate reports from MRI’. So an automated system on Cloud can solve this problem.

✔️ Definition

To Detect and Classify Brain Tumor using, CNN and TL; as an asset of Deep Learning and to examine the tumor position(segmentation).

✔️ About the data:

The dataset contains 3 folders: yes, no and pred which contains 3060 Brain MRI Images.
Folder 	Description
Yes 	The folder yes contains 1500 Brain MRI Images that are tumorous
No 	The folder no contains 1500 Brain MRI Images that are non-tumorous

Figshare (Academic Dataset)

    Dataset: Brain Tumor Dataset

    Images: 3,064 MRI images

    Types: Glioma, Meningioma, Pituitary, No tumor

    Link: Figshare Brain Tumor Dataset
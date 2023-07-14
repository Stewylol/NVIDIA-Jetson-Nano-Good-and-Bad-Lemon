# Good and Bad Lemon Detector

<div style="display: flex; justify-content: space-between; margin-bottom: 75px">
  <div>
    <img src="https://github.com/Stewylol/NVIDIA-Jetson-Nano-Good-and-Bad-Lemon/assets/139058370/94434e48-8eca-4983-8e93-6ac7befcc7e2" width="300" height="100%">
	<p>Model that detects a bad lemon:</p>
  </div>)



  <div>
    <img src="https://github.com/Stewylol/NVIDIA-Jetson-Nano-Good-and-Bad-Lemon/assets/139058370/4484dbe2-0aea-4afc-8ec1-14c4cdab1808" width="350" height="100%">
	<p>Model that detects a good lemon:</p>
  </div>
</div>

My project is centered around developing a sophisticated software that is capable of distinguishing good and bad lemons, drawing from a robust dataset of 3,000 images. These images, depicting lemons, represent a spectrum of conditions, with varying levels of quality. Each image is categorized as *good* or *bad*, providing the machine learning model with a foundation for understanding the characteristics that determine a lemon's quality. For a more comprehensive understanding of the dataset, you can follow this [link](https://www.kaggle.com/datasets/yusufemir/lemon-quality-dataset).

The core goal of this endeavor is to create an accurate and efficient model that can discern the quality of a lemon with precision. The integration of such a system in the agricultural sector holds potential for revolutionizing quality control processes, paving the way for enhanced accuracy and speed in lemon inspection.

The development and training of this model were accomplished using a Jetson Nano, a compact yet powerful machine learning device designed for edge deployment. It's crucial to note that workers here signifies the number of CPU threads utilized to load the data. The higher the number of workers, the more data batches can be loaded simultaneously, leading to quicker computation and a more streamlined training process. The number of epochs is not a fixed parameter; you can increase it if you aim to further refine the model's accuracy.

Moreover, the model's flexibility allows for the addition of more lemon types. If you wish to broaden its applicability, you can include more lemon images in the dataset and adjust the labels accordingly, thus enhancing the model's versatility and usefulness.

In practical terms, the project's applications are extensive. From farms to supermarkets, the model can be deployed to swiftly and accurately identify lemon quality. This not only streamlines quality control procedures but also provides crucial assistance for individuals who may find it challenging to distinguish between good and bad lemons—like those with color vision deficiency or certain visual impairments. This initiative is also a cost-effective solution as it can be integrated with existing infrastructures and inspection systems, without the need for specialized equipment or major overhauls.

## The Algorithm
The foundation of my project lies in the utilization of a powerful neural network model known as ResNet-18. Developed by researchers at Microsoft, ResNet-18, or Residual Network with 18 layers, is a deep learning model that excels in classifying images across a multitude of categories. The model earned its name from the *residual* or *shortcut* connections that skip one or more layers in the network. These connections help to combat the problem of vanishing gradients, a common issue in deep neural networks that can hinder their ability to learn and therefore their performance.

While the ResNet-18 model has originally been trained on the ImageNet dataset, comprising 1000 classes, it is not limited to this dataset. In fact, one of the key advantages of using ResNet-18 is its adaptability, which allows it to be efficiently retrained using different datasets.

For this project, I employed a rich dataset of 3,000 images, with these images capturing various types of lemons, including apples, bananas, guavas, limes, oranges, and pomegranates. Each image is meticulously labeled as *good* or *bad*, providing clear indications of lemon quality. Using this bespoke dataset, the ResNet-18 model was retrained, and the final layer of the network was adapted to classify these lemon images accurately, thus exemplifying a technique known as transfer learning. For a more comprehensive understanding of the dataset, you can follow this [link](https://www.kaggle.com/datasets/yusufemir/lemon-quality-dataset).

Transfer learning, as employed in this project, involves adjusting the weights of a pre-trained model (ResNet-18 in this case) to classify a customized dataset. Despite the change in the dataset, the network maintains its ability to recognize common features. Consequently, even though it is trained with a new dataset of 3,000 lemon images, the model benefits from its previous training, reusing the knowledge it gained to identify patterns in the new images.

ResNet-18, like other convolutional neural networks (CNNs), is composed of multiple convolutional layers. These layers, made up of various filters, work sequentially, scanning the image pixel by pixel. This process enables the model to break down the image into distinct segments, each containing unique patterns. When stacked, these layers provide the network with the ability to detect complicated and specialized objects in the images, such as the various lemons in our dataset. This is crucial to the successful classification of lemon quality in the project. The more layers present, the more nuanced the model's detection capabilities become, allowing it to localize and identify specific objects with impressive precision.

## Running the Project and Training a Network
First, you will obtain the data needed to train a network, then you will run the training script to train the network. If this is your first time training a dataset on the Jetson Nano, I would first advise using a 220 MB Lemon Quality dataset here: [Lemon Quality Dataset](https://www.kaggle.com/datasets/yusufemir/lemon-quality-dataset). I would advise to start downloading one of the ZIP files now because it will take a while to download.

## Instructions for VS Code Setup and Neural Network Training

A step-by-step guide on how to set up VS Code, organize your dataset, train your neural network, and test it on images.

### Setting up VS Code

1. Launch VS Code.
2. Click on the small green icon at the bottom left of your screen to access the SSH menu.
3. Select **+ Add New SSH Host** to add a new host.
4. Enter **ssh nvidia@x.x.x.x**, replacing **x.x.x.x** with the IP address you usually use in Putty or terminal to connect to the Nano.
5. Pick the first configuration file.
6. Click **Connect** in the prompted window.
7. Choose **Linux** as the operating system when asked.
8. If you're asked to continue, click **Continue**.
9. You'll be asked for a password after connecting to the Nano. Input your Nano password and hit **Enter**.
10. Select **Open Folder** and navigate to **jetson-inference**. Input your password again if required.
11. Click **Yes, I trust the authors** to access and start working on your projects in this directory.

### Preparing the Dataset

12. Navigate to **jetson-inference/python/training/classification/data**.
13. Extract the dataset ZIP file.
14. Inside **jetson-inference/python/training/classification/data**, create a new folder called **lemons**. Inside **lemons**, add three folders: **test**, **train**, **val** and a file named **labels.txt**.
15. In the **train** directory inside **lemons**, create 12 folders named **Apple_bad**, **Apple_Good**, **Banana_Bad**, **Banana_Good**, and similar ones for Guava, Lime, Orange, and Pomegranate.
16. Copy these folders to the **val** and **test** directories.
17. Distribute the images from your ZIP file among these folders, with 80% in the **train** folder, 10% in the **val** folder, and 10% in the **test** folder for each lemon type. Unfortunately, this will be a manual task and may take some time.

## Running the Docker Container

18. Go to the **jetson-inference** folder and run `./docker/run.sh`.
19. Once inside the Docker container, navigate to **jetson-inference/python/training/classification**.

## Training the Neural Network

20. Run the training script with the following command: `python3 train.py --model-dir=models/ANY_NAME_YOU_WANT --batch-size=4 --workers=4 --epoch=1 data/lemons` Replace `ANY_NAME_YOU_WANT` with your desired output file name. This process may take quite some time.
21. You can stop the process at any time using **Ctl+C** and resume it later using the `--resume` and `--epoch-start` flags.

## Exporting the Trained Network

To test your re-trained ResNet-18 model, it needs to be converted into the ONNX format. Follow these steps:

1. Navigate to **jetson-inference/python/training/classification** while still in the Docker container.
2. Run the ONNX export script: `python3 onnx_export.py --model-dir=models/ANY_NAME_YOU_WANT`
3. Go to **jetson-inference/python/training/classification/models/ANY_NAME_YOU_WANT** and look for a file named **resnet18.onnx**. This is your re-trained model!

### Testing the Trained Network on Images
To test your network, you can run images through it:
1. Exit the Docker container by pressing **Ctl + D** in the terminal.
2. On your Nano, navigate to **jetson-inference/python/training/classification**.
3. Check if the model exists on the Nano by executing **ls models/ANY_NAME_YOU_WANT/**. You should see a file named **resnet18.onnx**.
4. Set the **NET** and **DATASET** variables: **`NET=models/ANY_NAME_YOU_WANT DATASET=data/lemons`**
5. Run this command to see how the model works on an image from the test folder: `imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/Apple_Good/PICK_AN_IMAGE.jpg PICK_A_NAME_FOR_THE_IMAGE.jpg`. Keep  in mind that you are able to change **Apple_Good** to any lemon and quality you want, you are able to pick any test image by changing **PICK_AN_IMAGE.jpg**, and are able to change the name of the output image name by changing **PICK_A_NAME_FOR_THE_IMAGE.jpg**.
6 Launch VS Code to view the image output (located in the classification folder). Remember to replace **ANY_NAME_YOU_WANT** with the name you gave your model while training.

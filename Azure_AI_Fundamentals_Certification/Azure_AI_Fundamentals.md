# Azure AI Fundamentals Notes

## Get started with artificial intelligence on Azure

Azure Machine Learning service is a cloud-based platform for creating, managing, and publishing machine learning models:

- Automated machine learning: quickly create an effective machine learning model from data
- Azure Machine Learning designer: A graphical interface enabling no-code development
- Data and compute management: Cloud-based data storage and compute resources
- Pipelines: define pipelines to orchestrate model training, deployment, and management tasks

### Anomaly detection

In Microsoft Azure, the Anomaly Detector service provides an application programming interface (API) that developers can use to create anomaly detection solutions.

### Computer Vision models and capabilities

- Image classification involves training a machine learning model to classify images based on their contents
- Object detection machine learning models are trained to classify individual objects within an image, and identify their location with a bounding box
- Semantic segmentation is an advanced machine learning technique in which individual pixels in the image are classified according to the object to which they belong
- Image analysis extract information from images, including "tags" that could help catalog the image or even descriptive captions that summarize the scene shown in the image
- Face detection is a specialized form of object detection that locates human faces in an image
- Optical character recognition is a technique used to detect and read text in images

### Understand natural language processing

- Analyze and interpret text in documents, email messages, and other sources
- Interpret spoken language, and synthesize speech responses
- Automatically translate spoken or written phrases between languages
- Interpret commands and determine appropriate actions

### Understand conversational AI

Bots can be the basis of AI solutions for:

- Customer support for products or services
- Reservation systems for restaurants, airlines, cinemas, and other appointment based businesses
- Health care consultations and self-diagnosis
- Home automation and personal digital assistants

At Microsoft, AI software development is guided by a set of six principles, designed to ensure that AI applications provide amazing solutions to difficult problems without any unintended negative consequences:

- Fairness
- Reliability and safety
- Privacy and security
- Inclusiveness
- Transparency
- Accountability

---

## Use visual tools to create machine learning models with Azure Machine Learning

### Azure Machine Learning

Azure Machine Learning is a cloud service that you can use to train and manage machine learning models. It includes a wide range of features and capabilities that help data scientists prepare data, train models, publish predictive services, and monitor their usage. Most importantly, it helps data scientists increase their efficiency by automating many of the time-consuming tasks associated with training models; and it enables them to use cloud-based compute resources that scale effectively to handle large volumes of data while incurring costs only when actually used.

> Create a workspace:
>
> - Sign into the Azure portal
> - Create a resource, search for Machine Learning
> - Subscription: Your Azure subscription
> - Resource group: Create or select a resource group
> - Workspace name: Enter a unique name for your workspace
> - Region: Select the geographical region closest to you

There are four kinds of compute resource you can create (In Azure Machine Learning studio, view the Compute page -under Manage-):

- **Compute Instances**: Development workstations that data scientists can use to work with data and models
- **Compute Clusters**: Scalable clusters of virtual machines for on-demand processing of experiment code
- **Inference Clusters**: Deployment targets for predictive services that use your trained models
- **Attached Compute**: Links to existing Azure compute resources, such as Virtual Machines or Azure Databricks clusters.

> Create compute targets:
> On the Compute Instances tab, add a new compute instance
>
> Create compute clusters:
> Compute Clusters tab, add a new compute cluster

### Create a dataset

In Azure Machine Learning, data for model training and other operations is usually encapsulated in an object called a dataset.

> Datasets:
> Create a new dataset from local files

### Run an automated machine learning experiment

In Azure Machine Learning, operations that you run are called experiments. Follow the steps below to run an experiment that uses automated machine learning to train a regression model.

> Automated ML page:
> Create a new Automated ML run

### Deploy a model as a service

After you've used automated machine learning to train some models, you can deploy the best performing model as a service for client applications to use. you can deploy a service as an Azure Container Instances (ACI) or to an Azure Kubernetes Service (AKS) cluster. For production scenarios, an AKS deployment is recommended, for which you must create an inference cluster compute target.

### Clean-up

The web service you created is hosted in an Azure Container Instance. If you don't intend to experiment with it further, you should delete the endpoint to avoid accruing unnecessary Azure usage. You should also stop the training cluster and compute instance resources until you need them again.

- In Azure Machine Learning studio, on the Endpoints tab, then select Delete (🗑) and confirm that you want to delete the endpoint
- On the Compute page, on the Compute Instances tab, select your compute instance and then select Stop

### Create a Regression Model with Azure Machine Learning designer

You can use Microsoft Azure Machine Learning designer to create regression models by using a drag and drop visual interface, without needing to write any code. To use the Azure Machine Learning designer, you create a pipeline that you will use to train a machine learning model. This pipeline starts with the dataset from which you want to train the model.

> Data Transformation section contains a wide range of modules you can use to transform data before model training
>
> - Select Submit, and run the pipeline
> - The dataset is now prepared for model training
> - Select the completed Normalize Data module, and in its Settings pane on the right, on the Outputs + logs tab
> - Select the Visualize icon for the Transformed dataset

### Create and run a training pipeline

> - In the Data Transformations section, drag a Split Data module onto the canvas under the Normalize Data module
> - In the Model Training section, drag a Train Model module to the canvas
> - In the Machine Learning Algorithms section, and under Regression, drag a Linear Regression module to the canvas
> - In  the Model Scoring & Evaluation section and drag a Score Model module to the canvas
> - Select Submit, and run the pipeline

![Fig. 1](Images/Capture.PNG)
*Credits to Microsoft: <https://docs.microsoft.com/en-us/learn/modules/create-regression-model-azure-machine-learning-designer/evaluate-model>*

### Add an Evaluate Model module

In the Model Scoring & Evaluation section, drag an Evaluate Model module to the canvas, under the Score Model module, and connect the output of the Score Model module to the Scored dataset. These include the following regression performance metrics:

- Mean Absolute Error (MAE): The average difference between predicted values and true values
- Root Mean Squared Error (RMSE): The square root of the mean squared difference between predicted and true values
- Relative Squared Error (RSE): A relative metric between 0 and 1 based on the square of the differences between predicted and true values
- Relative Absolute Error (RAE): A relative metric between 0 and 1 based on the absolute differences between predicted and true values
- Coefficient of Determination (R2): This metric is more commonly referred to as R-Squared, and summarizes how much of the variance between predicted and true values is explained by the model

### Create and run an inference pipeline

> - In Azure Machine Learning Studio, click the Designer page to view all of the pipelines you have created
> - In the Create inference pipeline drop-down list, click Real-time inference pipeline

### Evaluate a classification model

The confusion matrix shows cases where both the predicted and actual values were 1 (known as true positives) at the top left, and cases where both the predicted and the actual values were 0 (true negatives) at the bottom right. The other cells show cases where the predicted and actual values differ (false positives and false negatives).

- Accuracy: The ratio of correct predictions (true positives + true negatives) to the total number of predictions
- Precision: The fraction of positive cases correctly identified (the number of true positives divided by the number of true positives plus false positives)
- Recall: The fraction of the cases classified as positive that are actually positive (the number of true positives divided by the number of true positives plus false negatives)
- F1 Score: An overall metric that essentially combines precision and recall.
- ROC curve: (ROC stands for received operator characteristic, but most data scientists just call it a ROC curve). Another term for recall is True positive rate, and it has a corresponding metric named False positive rate, which measures the number of negative cases incorrectly identified as positive compared the number of actual negative cases. Plotting these metrics against each other for every possible threshold value between 0 and 1 results in a curve. In an ideal model, the curve would go all the way up the left side and across the top, so that it covers the full area of the chart. The larger the area under the curve (which can be any value from 0 to 1), the better the model is performing - this is the AUC metric listed with the other metrics below

### Evaluate a clustering model

- Average Distance to Other Center: This indicates how close, on average, each point in the cluster is to the centroids of all other clusters
- Average Distance to Cluster Center: This indicates how close, on average, each point in the cluster is to the centroid of the cluster
- Number of Points: The number of points assigned to the cluster
- Maximal Distance to Cluster Center: The maximum of the distances between each point and the centroid of that point’s cluster. If this number is high, the cluster may be widely dispersed. This statistic in combination with the Average Distance to Cluster Center helps you determine the cluster’s spread

---

## Explore computer vision in Microsoft Azure

In Microsoft Azure, the Computer Vision cognitive service uses pre-trained models to analyze images, enabling software developers to easily build applications that can:

- Interpret an image and suggest an appropriate caption
- Suggest relevant tags that could be used to index an image
- Categorize an image
- Identify objects in an image
- Detect faces and people in an image
- Recognize celebrities and landmarks in an image
- Read text in an image

Creating an object detection solution with Custom Vision consists of three main tasks. First you must use upload and tag images, then you can train the model, and finally you must publish the model so that client applications can use it to generate predictions.

For each of these tasks, you need a resource in your Azure subscription. You can use the following types of resource:

Custom Vision: A dedicated resource for the custom vision service, which can be either a training, a prediction or a both resource.
Cognitive Services: A general cognitive services resource that includes Custom Vision along with many other cognitive services. You can use this type of resource for training, prediction, or both.

---

## Explore natural language processing

### Analyze text with the Text Analytics service

Analyzing text is a process where you evaluate different aspects of a document or phrase, in order to gain insights into the content of that text. There are some commonly used techniques that can be used to build software to analyze text, including:

- Statistical analysis of terms used in the text
- Extending frequency analysis to multi-term phrases, commonly known as N-grams
- Applying stemming or lemmatization algorithms to normalize words before counting them
- Applying linguistic structure rules to analyze sentences
- Encoding words or terms as numeric features that can be used to train a machine learning model
- Creating vectorized models that capture semantic relationships between words by assigning them to locations in n-dimensional space

You can choose to provision either of the following types of resource:

- A Text Analytics resource
- A Cognitive Services resource

## Speech recognition

Speech recognition is concerned with taking the spoken word and converting it into data that can be processed - often by transcribing it into a text representation. The spoken words can be in the form of a recorded voice in an audio file, or live audio from a microphone. Speech patterns are analyzed in the audio to determine recognizable patterns that are mapped to words. Microsoft Azure offers both speech recognition and speech synthesis capabilities through the Speech cognitive service, which includes the following application programming interfaces (APIs):

- The Speech-to-Text API
- The Text-to-Speech API

You can use the speech-to-text API to perform real-time or batch transcription of audio into a text format. The audio source for transcription can be a real-time audio stream from a microphone or an audio file. Real-time speech-to-text allows you to transcribe text in audio streams. You can use real-time transcription for presentations, demos, or any other scenario where a person is speaking.

Microsoft Azure provides cognitive services that support translation. Specifically, you can use the following services:

- The Translator Text service, which supports text-to-text translation
- The Speech service, which enables speech-to-text and speech-to-speech translation

---

## Get started with QnA Maker and Azure Bot Service

You can easily create a user support bot solution on Microsoft Azure using a combination of two core technologies:

- QnA Maker. This cognitive service enables you to create and publish a knowledge base with built-in natural language processing capabilities
- Azure Bot Service. This service provides a framework for developing, publishing, and managing bots on Azure

Conversations typically take the form of messages exchanged in turns. This pattern forms the basis for many user support bots, and can often be based on existing FAQ documentation. To implement this kind of solution, you need:

- A knowledge base of question and answer pairs
- A bot service that provides an interface to the knowledge base through one or more channels

### Creating a QnA Maker knowledge base

The service provides a dedicated QnA Maker portal web-based interface that you can use to create, train, publish, and manage knowledge bases (you must first provision a QnA Maker resource in your Azure subscription). You can use the QnA Maker portal to create a knowledge base that consists of question-and-answer pairs. These questions and answers can be:

- Generated from an existing FAQ document or web page
- Imported from a pre-defined chit-chat data source
- Entered and edited manually

After creating a set of question-and-answer pairs, you must train your knowledge base. This process analyzes your literal questions and answers and applies a built-in natural language processing model to match appropriate answers to questions, even when they are not phrased exactly as specified in your question definitions. When you're satisfied with your trained knowledge base, you can publish it so that client applications can use it over its REST interface. To access the knowledge base, client applications require:

- The knowledge base ID
- The knowledge base endpoint
- The knowledge base authorization key

### Build a bot with the Azure Bot Service

You can create a custom bot by using the Microsoft Bot Framework SDK to write code that controls conversation flow and integrates with your QnA Maker knowledge base. However, an easier approach is to use the automatic bot creation functionality of QnA Maker, which enables you create a bot for your published knowledge base and publish it as an Azure Bot Service application with just a few clicks. After creating your bot, you can manage it in the Azure portal, where you can:

- Extend the bot's functionality by adding custom code.
- Test the bot in an interactive test interface.
- Configure logging, analytics, and integration with other services.

When your bot is ready to be delivered to users, you can connect it to multiple channels; making it possible for users to interact with it through web chat, email, Microsoft Teams, and other common communication media.

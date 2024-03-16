# Drowsiness-Detection-using-DNN

# Demo Link
 `[drive link]` (https://drive.google.com/file/d/1Q5Dzh_d-F-bIyGpWBaDVjPmO1SncpUC2/view?usp=sharing)
# DESCRIPTION
- Drowsiness detection using Deep Neural Networks (DNN) is a technology that aims to identify signs of driver drowsiness or fatigue by analyzing various facial and behavioral cues. 
This technology has significant applications in improving road safety, particularly in the context of preventing accidents caused by drivers who are tired or drowsy.
Drowsiness-related accidents are a major concern worldwide, and technologies like DNN-based drowsiness detection systems can play a crucial role in mitigating these risks. 



# TEAM DETAILS
`TEAM NUMBER` : VH192


| Name	                |   Email                |
| ---------            |  --------------        |
|Pendyala. Sai Kumar	  |  9921004953@klu.ac.in  |
|V.Krishna sai Mahesh	 |  9921004763@klu.ac.in  |
|P. Nikhilesh Varma	   |  9921004947@klu.ac.in  |


# Problem statement

- The problem at hand is to develop an effective and reliable DNN-based drowsiness detection system for real-time monitoring of driver alertness. This system should be capable of accurately identifying signs of drowsiness and fatigue by analyzing facial expressions and behavioral patterns. By addressing this problem, we aim to contribute to the improvement of road safety and reduce the occurrence of accidents caused by drowsy driving.



# About the Project
- The project focuses on creating a prototype system using Deep Neural Networks (DNNs) for detecting drowsiness in individuals, particularly in scenarios like transportation where safety is paramount. The objective is to develop a robust system that can accurately identify signs of drowsiness using indicators such as facial expressions, eye movements, and physiological signals. The methodology involves data collection, preprocessing, feature extraction, model development, evaluation, and prototype integration. The prototype includes modules for data acquisition, preprocessing, DNN model implementation, alerting mechanism by using camera application. The expected outcomes include a reliable prototype for real-time drowsiness detection, improved safety measures, and potential for further research in human behavior monitoring and safety systems. Overall, the project aims to contribute to enhanced safety and well-being by effectively addressing the issue of drowsiness detection.
  

# Technical Implementation
- Drowsiness detection system is a car safety technology which helps prevent accidents caused by the driver getting drowsy. Makes the alert by finding the driver is tired or drowsiness.
- The proposed system is a driver drowsiness prediction system that will identify various scenarios. It will capture closed eyes, open mouth, hands-on eyes or mouth while nodding or yawning.
- An image of the driver captured through the camera serves as the system’s input.
- DNN accuracy for Training 98.3%  and  Testing accuracy 97.31%.

* The parameters that we are interested in discovering from the driver’s image are the following
- Number of blinks .
- Average blinking duration .
- Number of microsleeps, i.e., blinks with a duration of over 500ms .
- Number of yawns .
- Time spent yawning .

![pic1](https://github.com/satya24689/Drowsiness-Detection-using-DNN/assets/141759679/acb597df-2a39-4458-859f-d0f1d91f9d50)


# Techstacks Used
- `cv2`(computer vision)
- `dlib`(Library For machine Learning)
- `scipy.spatial` from `distance`
- `pygame`

![flow chart'](https://github.com/satya24689/Drowsiness-Detection-using-DNN/assets/141759679/ca3aa024-b334-4cd9-a2fd-41565af58f1b)


# How to run Locally
- step 1: install packages
  ```
  !pip install cv2
  !pip install dlib
  !pip install scipy.spatial
  !pip install pygame
  ```
- step 2 : install dat file from online ,which suitable for code
- step 3 : download buzzer or alarm audio file for alerting drowsy people
- step 4 : It will display on screen by questioning sentence ' Are You sleepy' by giving access to camera.
- step 5 : code is executed on jupyter by `run` option
- step 6 : code execution is terminate by clicking `kernel` to click clear all outputs.

- ![face result](https://github.com/satya24689/Drowsiness-Detection-using-DNN/assets/141759679/56cce5d8-8306-4767-ab3e-61c2bd9de82c)
![image](https://github.com/satya24689/Drowsiness-Detection-using-DNN/assets/141759679/98308672-8a6e-4d5f-a7be-4595a640cfe0)


 # what's next ?
 - The future plan for our project 
 - Implement the drowsiness detection in all  required areasareas
 - To reduce the daily life consequences faced by people due to drowsiness.
 - It will be implemented on institutions to decrase the drowsiness on children.
 - In  case of vehicles by detecting drosiness it will be a message to stop the vehicle.

 # Declaration
 - We confirm that the project showcased here was either developed entirely during the hackathon or underwent significant updates within the hackathon timeframe. We understand that if any plagiarism from online sources is detected, our project will be disqualified, and our participation in the hackathon will be revoked.
  

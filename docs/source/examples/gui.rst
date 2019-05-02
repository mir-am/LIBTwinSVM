An example of classification using the GUI
==========================================
In this section we have provided an easy step-by-step *Usage Example* which it shows how the GUI of the LIBTwinSVM works.
For more information on the application and its features, go to this link. 

--------------------
 Step 1: Data Import
--------------------
To use the application and training a specific data, the first step is to import the data. In the following section, we showed how to import data and how to use some of the application features.
 1. By default, the application starts on the Data tab. But just in case, if it was on other tabs by any chance, make sure to select the **Data** tab first.
  .. image:: images/1.jpg
 2. By clicking on the **Open** button in the **Import** box, the **File Explorer** will be open and you can then choose the file you want to train. 
  .. image:: images/2.jpg
 3. You can choose the file content separator if it is other than **comma**. 
 4. In **Read** box in **Data** tab,  you can choose how your data be processed by the application. You can choose to **Normalize** or to **Shuffle** the data.
 5. Then click on **Load** Button, and your data will be imported and also displayed in the feature box. 
  .. image:: images/3.jpg
  
--------------------
 Step 2: Classifying
--------------------
For this section, the data must be already imported and you are going to see how **Classify** tab works.
 1. Switch the tabs to **Classify** tab. 
  .. image:: images/4.jpg
 2. Now you have a bunch of options to choose such as **Classifiers**, **Kernel** type, **Evaluation** Method and **Hyper Parameters**.
 you can choose one of them, or leave them with their default values and just go to the next step. 
 3. There is a save result box on that page. Before starting the training, you have to select a location so the application saves the final results for us. You can also check the **Log FIle** if you need the application logs.
  .. image:: images/5.jpg
 4. Now It is time to *Classify*. By Clicking on the **Run!** button, you will see a **Confirmation** message, pops up on the screen just to check if everything is exactly the way you had set before training.
  .. image:: images/6.jpg
 5. Click OK if everything is the way you want and it takes a few seconds to several minutes (depends on the data size) to be done!
  .. image:: images/7.jpg

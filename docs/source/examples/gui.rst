An example of classification using the GUI
==========================================
In this section we have provided an easy step-by-step *Usage Example* which it shows how the GUI of the LIBTwinSVM works.
For more information on the application and its features, go to this link. 

--------------------
 Step 1: Data Import
--------------------
To use the application, the first step is to import the data. In the following section, we have shown how to import data and how to use some of the application features.
 1. By default, the application starts on the Data tab.
  .. image:: images/1.jpg
 2. By clicking on the **Open** button in the **Import** box, the **File Explorer** will be open and we can then choose the file we want to train. 
  .. image:: images/2.jpg
 3. We can choose the file content separator if it is other than **comma**. 
 4. In **Read** box in **Data** tab,  we can choose how we want the data to be processed by the application. We may want to **Normalize** or  **Shuffle** the data.
 5. Then we must click on **Load** Button, and the data will be imported and also displayed in the feature box. 
  .. image:: images/3.jpg
  
--------------------
 Step 2: Classifying
--------------------
For this section, the data must be already imported and we are going to see how **Classify** tab works.
 1. Switch the tabs to **Classify** tab. 
  .. image:: images/4.jpg
 2. Now we have a bunch of options to choose such as **Classifiers**, **Kernel** type, **Evaluation** Method and **Hyper Parameters**. We can choose one of them, or leave them with their default values and just go to the next step. 
 3. There is a save result box on that page. Before starting the training, we must select a path so the application saves the final results for us. We can also check the **Log FIle** if we need the application logs.
  .. image:: images/5.jpg
 4. Now It is time to *Classify*. By Clicking on the **Run!** button, we can see a **Confirmation** message, pops up on the screen just to check if everything is exactly the way we had set before training.
  .. image:: images/6.jpg
 5. After checking we click OK if everything is the way we want and it takes a few seconds to several minutes (depends on the data size) to be done!
  .. image:: images/7.jpg

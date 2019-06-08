An example of using a pre-trained Model for classification
===========================================================
In this section, we have illustrated how to use a pretrain model, saved by the LIBTwinSVM. Note that, this step requires a pre-trained model file saved in classifying step as a **.joblib** file. If you don't have the previously said file, please refer to [classication usage example](https://libtwinsvm.readthedocs.io/en/latest/examples/GUI/classify.html#).

--------------------
 Step 1: Data Import
--------------------
Please note that, to use a model on test samples, the test data must have the same features as the training data. Below is a step-by-step procudure on how to load your data for reusing a pre-trained model. 

 1. By default, the application starts on the Data tab.
  .. image:: images/1.jpg
 2. By clicking on the **Open** button in the **Import** box, the file dialog will be opened and you can then choose the dataset you want to test with a pre-trained model. 
  .. image:: images/mod1.jpg
 3. You may need to change the column separator if it is other than **comma**. 
 4. In **Read** box in **Data** tab, you may want to **Normalize** and/or  **Shuffle** the data.
 5. Then you must click on **Load** Button, and the data will be loaded and also displayed in the feature box. 
  .. image:: images/mod2.jpg
  
----------------------------------
 Step 2: Using a Pre-trained Model
----------------------------------
Up to now, the dataset should be loaded. Follow the below instructions for evaluating a pre-trained model on test samples.

 1. First, switch the tabs to **Model** tab. 
  .. image:: images/mod3.jpg
 2. Click on the *Select* button in the **Import** box to select your pre-trained model file. You can see a pre-trained model file by LIBTwinSVM in the figure below. 
  .. image:: images/mod4.jpg
 3. After setting the pre-trained model's path, click on the **Load** button to load the model and display the model's characteristics such as its classification type and the used kernel. 
  .. image:: images/mod5.jpg
 4. In the final step, you can evaluate your model by clicking on the **Evaluate** model. Moreover, if you want to save the predicted label of each sample, you can check **Save Predictions** button.
  .. image:: images/mod7.jpg


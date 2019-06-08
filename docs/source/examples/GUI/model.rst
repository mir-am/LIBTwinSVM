An example of using a pre-trained Model for classification
===========================================================
In this section, we have illustrated how to use a pretrain model, saved by the LIBTwinSVM. Note that, this step requires a pre-trained model file saved in classifying step as a **.joblib** file. If you don't have the previously said file, please refer to [classication usage example](https://libtwinsvm.readthedocs.io/en/latest/examples/GUI/classify.html#).

--------------------
 Step 1: Data Import
--------------------
Please note that, to use a model on test samples, the test data must have the same features as the training data. Below is a step-by-step procudure on how to load your data for reusing a pre-trained model. 

 1. By default, the application starts on the Data tab.
  .. image:: images/1.jpg
 2. By clicking on the **Open** button in the **Import** box, the **File Explorer** will be open and you can then choose the file you want to test with a pre-trained model. 
  .. image:: images/mod1.jpg
 3. You may need to change the file content separator if it is other than **comma**. 
 4. In **Read** box in **Data** tab,  you can choose how you want the data to be processed by the application. You may want to **Normalize** or  **Shuffle** the data.
 5. Then you must click on **Load** Button, and the data will be imported and also displayed in the feature box. 
  .. image:: images/mod2.jpg
  
----------------------------------
 Step 2: Using a Pre-trained Model
----------------------------------
So far, the data is imported and You are going to see how **Model** tab works.

 1. First, switch the tabs to **Model** tab. 
  .. image:: images/mod3.jpg
 2. Click on the *Select* button in the **Import** box to set the pre-trained model path. You can see a sample of saved model file by LIBTwinSVM in the figure below. 
  .. image:: images/mod4.jpg
 3. After setting the the pre-trained model's path, click on the **Load** button to make LIBTwinSVM load the model and display its relative information such as its classification type and the used kernel. 
  .. image:: images/mod5.jpg
 4. In the final step, you can evaluate your model by clicking on the **Evaluate** model. You may also want to save the predicted results, you can check **Save Predictions** button and LIBTwinSVM will create the result file in the same directory as the model.
  .. image:: images/mod7.jpg


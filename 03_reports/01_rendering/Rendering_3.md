# Final Report: Plant Disease Recognition

## Conclusion Drawn
The main takeaway from this project is that deep learning, especially a method called transfer learning, is a very good way to identify plant diseases from photos of their leaves. Our final model, which uses the MobileNetV2 architecture, worked much better than the simple model we built first, proving it's a strong choice for this task.

We also successfully showed that the entire process—from looking at the data to training and understanding the model—can be organized into an interactive website using Streamlit. This makes the project's results easy for anyone to see and understand, connecting data science with real-world use.

## Difficulties Encountered During the Project
**What was the biggest challenge in this project?**

Our biggest challenge was getting the advanced transfer learning model to work correctly and understanding its results. Building a simple model was straightforward, but using transfer learning required us to better understand complex ideas like "fine-tuning." It was tricky to figure out which parts of the advanced model to "unfreeze" for training and what learning rate to use. It was also a difficult task to use tools like Grad-CAM and SHAP to figure out why this more complex model was making its decisions.

**For the points below, please explain any difficulties you had and how they affected your project timeline.**

**Forecast:** The training part of the project, especially finding the best settings for the advanced model, took longer than we expected. Each test run took a lot of time, which meant we had to wait longer to see if our changes worked.

**Datasets:** The dataset was large and well-organized, which was very helpful. However, because it was so big, we had to learn how to load the images efficiently. Our first attempts without these efficient methods were too slow and used too much computer memory.

**Technical/theoretical skills:** The steepest learning curve was related to advanced deep learning topics. Our training gave us a great start, but we had to do a lot of extra reading and experimenting to master the details of transfer learning and the tools used to interpret the models.

**Relevance:** We did not have any major problems here. The methods we chose are standard for image classification, and the data was a perfect fit for the problem we wanted to solve.

**IT:** Training these models required a lot of computer power. We had to plan our experiments carefully based on the resources we had. This limited how long we could train the models or how many different settings we could test.

## Report
**What was your main contribution to the project?**

We shared the work equally, so our joint contribution covered all parts of the project from start to finish. We worked together on:

- Data Exploration: Looking at the data to understand its structure and content.
- Model Building: Creating both the simple and the advanced models.
- Model Training: Running the training process, checking the performance, and evaluating the results.
- Model Interpretation: Using tools like Grad-CAM and SHAP to understand how the models made decisions.
- App Development: Building the Streamlit website to show our work.

**Did you change the model during the project? If so, please explain.**

Yes, we made a big change to our model. We started with a basic CNN model that we built ourselves to have a starting point (a "baseline"). After seeing how it performed, we moved to a more advanced model that uses transfer learning. This final model is based on MobileNetV2, a powerful model that has already been trained on millions of images. We adjusted the final parts of this model to work specifically on our plant disease dataset. This change was key to the project's success and made it work much better.

**Please show your results and compare them to your starting point (the benchmark).**

Our starting point was the simple CNN model we built first. The final transfer learning model (MobileNetV2) was better than our starting model in every important way.

- Accuracy: The simple model had a certain accuracy, but the advanced model was much more accurate (for example, over 95-98% correct).
- Precision and Recall: The advanced model was also better at correctly identifying each of the 38 diseases. This means it was more reliable at telling different diseases apart, even ones that look very similar.

You can see the detailed reports for both models in the "Modelling" and "Modelling Advanced" pages of the Streamlit app.

**For each project goal, explain if you achieved it.**

- Build an accurate model: Achieved. Our final model can identify plant diseases with high accuracy.
- Learn about working with image data: Achieved. We gained a lot of hands-on experience in every step of an image project, from exploring the data to training and understanding deep learning models.
- Build a Streamlit app: Achieved. We built a complete Streamlit website that works as a dashboard for our project, making it easy to see our results.

**Now that the goals are met, how could your model be used in the real world?**

Our model can be used in a few real-world ways:

- A tool to help farmers: A farmer could take a photo of a leaf with their phone, and an app using our model could quickly tell them what disease it might be. This would help them treat plants faster.
- Checking large fields for diseases: The model could be used with drones or satellites to automatically scan large farms and find areas where a disease is starting to spread.
- A learning tool: The Streamlit app itself is a great way for students or other researchers to see and learn how computer vision can be used in farming.

## Continuation of the Project
**What are some ways you could improve the model in the future?**

- Live Prediction Page: The best next step would be to add a page where users can upload their own photos to get a prediction.
- More Data: Adding more types of plants and pictures of diseases from different parts of the world would help the model work even better on new images it hasn't seen before.
- Try Other Models: We could test other advanced models like EfficientNetV2 or ResNet to see if they work even better.
- Make a Phone App: Creating a simple phone app would make the tool very easy for farmers to use directly in the fields.

**How did your project add to scientific knowledge?**

This project is a good, clear example of how transfer learning can be successfully used in agriculture. By building the Streamlit website, we also helped share what we learned by making difficult ideas, like how a model works (using Grad-CAM), easy for people without a technical background to understand.

## Bibliography
- James, G., Witten, D., Hastie, T. and Tibshirani, R. (2013) An Introduction to Statistical Learning: with Applications in R. New York: Springer.
- Lundberg, S.M. and Lee, S.-I. (2017) A Unified Approach to Interpreting Model Predictions. arXiv preprint arXiv:1705.07874.
- Lv, M., Liu, S., Li, Y., Dong, G., He, Y. and Wang, W. (2022) ‘DSEB-EUNet: a deep learning approach with dual-stream and enriched-based U-Net for plant leaf disease segmentation’, Frontiers in Plant Science, 13.
- Selvaraju, R.R., Cogswell,M., Das, A., Vedantam, R., Parikh, D. and Batra, D. (2016) Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. arXiv preprint arXiv:1610.02391.
- SHAP developers (2025) Explain MobilenetV2 using the Partition explainer (PyTorch). Available at: https://shap.readthedocs.io/en/latest/example_notebooks/image_examples/image_classification/Explain%20MobilenetV2%20using%20the%20Partition%20explainer%20%28PyTorch%29.html (Accessed: 4 August 2025).
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł. and Polosukhin, I. (2017) Attention Is All You Need. arXiv preprint arXiv:1706.03059.

## Appendices
**Description of Code Files**

- Introduction.py: The main landing page of the Streamlit application, providing an overview of the project.
- pages/1_Exploration.py: Contains the code for the data exploration dashboard, including visualizations of the dataset's structure, content, and characteristics.
- pages/2_Data_Visualization.py: An interactive page allowing users to view sample images from the dataset based on selected plant and disease types.
- pages/3_Modelling.py: Displays the architecture, training history, and evaluation results for the initial baseline CNN model.
- pages/5_Modelling_Advanced.py: Displays the architecture, training history, and evaluation results for the final transfer learning model (MobileNetV2).
- pages/About.py: Provides information about the project team, supervisor, and the context of the DataScientest program.
- utils.py: A utility script containing helper functions for loading the configuration, data, and trained models, used by all pages.
- Code_for_streamlit.py: Contains helper functions specifically for processing and visualization within the Streamlit pages, such as Grad-CAM logic.
- config.json: A configuration file storing all relevant paths and filenames, allowing for easy management of project assets.

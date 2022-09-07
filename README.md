# Agrammatism-Diagnosis-using-LSTM
A Bi-directional LSTM in Python to detect grammatical anomalies in dataset with 90.96% accuracy

## **Abstract**

Agrammatism in non-fluent Aphasia Cases can be defined as a language disorder wherein a patient can only use content words ( nouns, verbs and adjectives ) for communication with their speech devoid of functional word types like conjunctions and articles, generating communication with extremely rudimentary grammar .

  

This characteristic is also seen in Bilingual Speech (sentences with components of two different languages), where grammatical omissions are a key identifier in detecting whether said speech is agrammatic in nature or not. Even though the rate of these omissions decrease after the combination of two different languages in one text, such characteristics are quite prevalent.

  

We describe this approach as a novel method to detect Agrammatism in the Bilingual user input ( speech or text ) using a Bi-Directional Recurrent Neural Network, where a classification based model is employed to detect whether the sentence input is agrammatic in nature or preserves a form of rudimentary grammar.

  

The applications of this approach can be used to timely analyse whether a patient who exhibits forms of Agrammatic speech, particularly those cases where the milders forms of bilingual agrammatism do not exhibit clear signs of non-fluent aphasia and hopes to increase access to diagnosis where diagnostic tests such as the BAT (Bilingual Aphasia Test ) might not be present. Early detection can allow patients to obtain conversation therapy at the optimum time for smooth treatment of the disorder. This framework also proves to be particularly useful in geographies where patients communicate in 2 or more languages and their speech is a hybrid of words from two different languages such as India.

## 1. Introduction

  

Aphasia is a neuropsychological or communication disorder which hampers the ability of a patient to understand or formulate language. There are different types of Aphasia which stem from different parts of the brain

  

Agrammatism usually occurs due to Broca’s Aphasia. For the purposes of this project, we will target and assist the patients of non-fluent Aphasia, wherein sentence structure is devoid of functional words thereby causing lack of comprehension and low self esteem for the patients.

  

## 2. Objectives

  

The research aims to create a coherent framework for analysing whether given text is agrammatic in nature or not, which can lead to a potential diagnosis of non-fluent Aphasia.

  

The other objective is to reduce the cost of this diagnosis, whilst increasing access to automated forms of assessment.

  

## 3. Limitations

  

Since no textual data examples of agrammatic text and their correct grammar forms were available, we had to formulate our own dataset on the basis of speech characteristics in agrammatism due to non-fluent aphasia in “Hinglish”.  
  
The assumptions of this approach is that the content words received in the inputs of both text models are contextually correct. This is a potential limit to cases of severe Agrammatism where such order might not be inherently correct.

  

## 4. Scope

  

The classification method used below can be applied to a dataset, once availability of more detailed levels of agrammatism data that contain the level of agrammatism with respect to gender, tense, grammatical structure in the sentence formation are present.

  

The model, which is tuned to diagnose “Hinglish” Speech, can also be used to identify any other forms of bilingual sentences without extensive tuning of the weights and biases in the neural network.

  

  

## 5. Methodology

  

The framework shown in Figure 1 is broken down into three modules, each module serving a part in the overall detection of agrammatism in the patient’s input. The first module is creating the input text layer for the dataset, since no readily available dataset for agrammatism is available, we created a proxy dataset for detection. The methods used in this module are further explained in 4.1

The second module is the Bi-Directional RNN model used for speech classification ( further expanded in section below ) and the classification output y is the end result of the approach discussed in the results.

  
  
![](https://lh4.googleusercontent.com/EwW8FRr4oSjeOuonnODad65_aPh8wTTvXh5Tf-Upf6UX8U6zteK207w94UuIYb6pkzJWfnl2so-PGj62wpu8iBeYaikyWuYRoxFm8V19yrRkmQ4fZym-0vFZwVNugx6gc_A9m-3Cf2Znpl_chxozTQ)  

## 6. Dataset

  
  

The model uses more than 20,000 examples with 2 inputs associated in each example in the form of [“Speech Type”, “Sentence”] to be fed into the Bidirectional LSTM model.  
  
The dateste for agrammatic and non-agrammatic sentences were created using the speech characteristics of individuals affected by agrammatism. This speech is generally described as one devoid of functional words to provide grammatical structure to the sentence often resulting in telegraphic forms of speech in severe cases.

  

**6.1 Dataset Expression**

  

For a time step t and Mini-Batch input 𝐗t∈ℝ𝑛×𝑑, where the number of examples in the data is n and the inputs in itinerant of the example is d. Thus we can write the mini-batch input as

  

𝐗t∈ ℝ≈40,000

  

Where d ≈ 20,000 and n = 2 (input + output label)

  

These sentences were then stripped of “stopwords” after tokenization using an NLTK function and appended to a dataset with sentences devoid of stopwords ( a proxy for agrammatic text ) and ones containing stopwords.

![](https://lh4.googleusercontent.com/x_Wk8_w27dOhPBfd06lshKXCVRtRDIXLUSraEzBPANZtIeIxKPaHJtcXpXjwVUVpNJVraoAi5UeGTKfyAg4HbIf9hAjahGYlARsaPeKK0C9EHM2mZ0NsdmhHrRxrZ6wykw5ziwC5uvpY64Eh8m8hVg)

Then a glove vector converts each word into a numeric representation of about 50 dimensions, so that the model can process the text. Thus, the figure below illustrates the cleaning and processing data pipeline for the neural network.

  
  
  
  
  
  
  
  
  
  
  
  
  
  
![](https://lh5.googleusercontent.com/AlHXn2mjCg1zX4d4nunV_ZwHSu65Ejzm4FaFc_AH6T23Bgu9Ih1qy4dE86vEz_DDX7iSWyrSew50s8IwPjBQ_-e-sWRawE59IFk4Sh6AZzleYJtrXNc1ZkeufFPC5p23UYxO4ye0akzGQdo030jF-w)  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

## 7. Model

  

A Bi-Directional Recurrent Neural Network is considered an improved version of conventional recurrent neural networks. The conventional neural network employed for text-classification with Ht denoting the hidden state, and xt is the label and text input at each time-step t, with y being the final output label that is obtained after training and testing.

![](https://lh6.googleusercontent.com/UaA36snUjm6kVp4pSNgXKBkWqhZedYYdxwj47fXBuAi4zIYWPPaTOR730Bnnb9iUWJMfAsbfjuPf1Gd6WX46Py-LSVJtijpsMriqwBjKUUkWPcF45Pe_7pYaEVvJra2IaM3r6IrjbwARL6Elye2rsw)

It strengthens the RNN layer in the network structure into a bidirectional RNN layer. In this model, both directions “bi”, both forward encoding information but also reverse encoding information is considered.

  

The network for this classification model can be seen below :

  

![](https://lh6.googleusercontent.com/sSUCc1fgLK02iA2JlzZBfHPpWiMzchj9Hx645wOUVvyKAC9ja5DYLvXNakMA1CUX8G2a2IB4m1LFAV9sVG1Te2gGve7zqkQepIUht8-IDQsqLDHsbH_8bkUBEy2r2F5VBQjQuuveQ3iMhsOVq4yYjQ)

  
  

The code for the same was written in python using the Keras library on Google Colab for fast prototyping. The screenshot for the same can be seen below :  
![](https://lh5.googleusercontent.com/kC1vZsYMnpp-wFeTcksJH4oM8ooyIDlO-XUIe48p3HY7TbPkh9E7jMa51Vfkani5nOjz2m9lAr8obdVEyrh60OZ7Cd3tl6GtgpiMc3q_ZAJZ5EQ1UB7io9w6B-MH06Hkakt461eo6iK6QRZPWdENig)  
  

  
  

## 8. Results

  

The network was trained on a dataset containing more than 20,000 sentences with the Adam Optimizer that has a learning rate of 0.01 for 32 epochs and a batch size of 128. The training performance of the model is visualised below :

  

![](https://lh3.googleusercontent.com/xTqgEn4lGStbYYnzoKJ3sB7tYfHhZqilPnGd5Cr266SM_0kh87Sg2yGRVTF0YEtE1_zToaJjtpeovVGWjEfNcKmN-59zGQ-m4Byvj2UODACzbJjg3qOc4cLnNGLgiqqOak2c6LHwhd6x3n-lzi_BJw)  
  

As you can see from the loss-accuracy graph above, the initial loss and accuracy started at ≈ 0.48 and ≈ 0.79 respectively. As the number of epochs increased, we see a sharp exponential increase in the accuracy and its counter in the loss values of the graph, showcasing significant results with a final accuracy of 0.9096. This was done after the dataset was split into train and test sections to avoid overfitting.

  

## 9. Future Work

  
The framework used for text for detection of agrammatism can be furthered by adding an utterance detection layer which is common in higher levels of agrammatism for detection of potentially telegraphic speech.

  

I'm also researching using bound grammatical morphemes as a potential indicator of Agrammatism detection, especially the Hindi component of the sentences.

  

## 10. Acknowledgment

  

This paper would not have been completed without the constant support and guidance of Mr. Mayank Bhasin of IIT Kharagpur, whose expertise in deep learning and its applications were invaluable during the development of this research. I would also like to thank Professor Rajiv Bhatnagar of the Aphasia and Stroke Institution of India for his constant collaboration with regards to all aphasia and agrammatism related queries. Finally, I would like to extend my thanks to Mr. Pratik Joshi ( ex Google Research India and Microsoft Research ) who aided in the initial stages of this project.

  
  
  
  
  

## 11. Bibliography

  

[1] Arslan, S., Gür, E., & Felser, C. (2017). Predicting the sources of impairedwh-question comprehension in non-fluent aphasia: A cross-linguistic machine learning study on Turkish and German. Cognitive Neuropsychology, 34(5), 312–331. https://doi.org/10.1080/02643294.2017.1394284

[2] Caramazza, A., & Zurif, E. B. (1976). Dissociation of algorithmic and heuristic processes in language comprehension: Evidence from aphasia. Brain and Language, 3(4), 572–582. https://doi.org/10.1016/0093-934x(76)90048-1

[3] Fyndanis, V., & Themistocleous, C. (2018). Morphosyntactic production in agrammatic aphasia: A cross-linguistic machine learning approach. Frontiers in Human Neuroscience, 12. https://doi.org/10.3389/conf.fnhum.2018.228.00075

[4] Zurif, E., Caramazza, A., & Myerson, R. (1972a). Grammatical judgments of agrammatic aphasics. Neuropsychologia, 10(4), 405–417. https://doi.org/10.1016/0028-3932(72)90003-6

[5] Zurif, E., Caramazza, A., & Myerson, R. (1972b). Grammatical judgments of agrammatic aphasics. Neuropsychologia, 10(4), 405–417. https://doi.org/10.1016/0028-3932(72)90003-6

[6] Pengfei Liu, Xipeng Qiu, and Xuanjing Huang. 2016. Recurrent neural network for text classification with multi-task learning. In Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence (IJCAI'16). AAAI Press, 2873–2879.

[7] Honglun Zhang, Liqiang Xiao, Yongkun Wang, and Yaohui Jin. 2017. A generalized recurrent neural architecture for text classification with multi-task learning. In Proceedings of the 26th International Joint Conference on Artificial Intelligence (IJCAI'17). AAAI Press, 3385–3391.

[8] Yuchun Fang, Zhengyan Ma, Zhaoxiang Zhang, Xu-Yao Zhang, and Xiang Bai. 2017. Dynamic multi-task learning with convolutional neural network. In Proceedings of the 26th International Joint Conference on Artificial Intelligence (IJCAI'17). AAAI Press, 1668–1674.

[9] Bhatnagar, S., & Whitaker, H. A. (1984). Agrammatism on inflectional bound morphemes: a case study of a Hindi-speaking aphasic patient. Cortex; a journal devoted to the study of the nervous system and behavior, 20(2), 295–301. [https://doi.org/10.1016/s0010-9452(84)80049-0](https://doi.org/10.1016/s0010-9452(84)80049-0)

[10] Li, J., Xu, Y., & Shi, H. (2019). Bidirectional LSTM with Hierarchical Attention for Text Classification. 2019 IEEE 4th Advanced Information Technology, Electronic and Automation Control Conference (IAEAC). Published. https://doi.org/10.1109/iaeac47372.2019.8997969

[11] Schuster, M., & Paliwal, K. (1997). Bidirectional recurrent neural networks. IEEE Transactions on Signal Processing, 45(11), 2673–2681. https://doi.org/10.1109/78.650093

[12] Zhou, P. (2016, November 21). Text Classification Improved by Integrating Bidirectional LSTM. . . ArXiv.Org. https://arxiv.org/abs/1611.06639

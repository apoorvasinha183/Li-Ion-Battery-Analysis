# CS7643-DEEP LEARNING PROJECT 
# Implementing a Battery Management System using Deep Learning
# Group Members : Antonio Varagnolo, Jorge Fernandez and Apoorva Sinha
# Original Code based in Tensorflow at :https://github.com/nasa/Li-ion-Battery-Prognosis-Based-on-Hybrid-Bayesian-PINN
We implement the paper :
Nascimento, Viana, Corbetta, Kulkarni.
[A Framework for Li-ion Battery Prognosis Based on Hybrid Bayesian Physics-Informed Neural Networks](https://www.nature.com/articles/s41598-023-33018-0). Nature Scientific Reports, 2023.

## Summary
   Recent advancements in electrical transportation have heightened the need for reliable real-time Battery Management Systems (BMS), particularly in public transportation and industrial sectors. Existing BMS solutions face challenges in accurately and timely reporting battery metrics such as SOC and SOH due to excessive computational cost, as in the case of purely scientific models or generalization problems, as it is often the case for purely parametric strategies. To address these challenges, our study revisits and proposes a redesign of Nascimento et al. approach \cite{Fricke_Nascimento_Corbetta_Kulkarni_Viana_2023}, integrating physics-informed machine learning strategies into BMS development, aiming to enhance accuracy, reliability, and real-time performance. We achieved a DC tracking worst case error of 1.1 $\%$ , a discharge time error of $17.6\%$ while our worst case error on dynamic data was $0.7\%$ with worst case drift of $7.8\%$ after 20 hours.
## Details
Most of the stuff in main is a leftover from the old Tensorflow code in the repository linked above . If you want to set this up on your system , you need to first run data_retrieval.py . That will organize the data over which we will train.Then you would want to switch over to the torch folder. There are a lot of pre-trained models there. If you want to train your own network the instructions can be found in the folder's README.











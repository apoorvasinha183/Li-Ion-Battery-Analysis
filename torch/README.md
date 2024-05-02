## If you have not set up your data flow from the parent folder,nothing will work here.
## To organise the data please run battery_data and battery_random_walk_data.py
 Now you want to know which model you'll be dealing with ,there are 3 here out of which two are based in ML and the third one is a pure physics based model . Ignore BatteryRNNCell_old.py it is a tensorflow model . If you want to see the original tensorflow model we suggest you to visit the link in the base README.BatteryRNNCell_mlp is the ML model which has a single MLP which needs to be trained . BatteryRNNCell_PINN is the pure Physics Informed Model . It is still a work in progress.
 For now if you wanna train from scratch you should run :
 1. weight_initialization.py  - This teaches a key non-linearity to your network. These weights will help your model converge to a good value. This is called a warm start.
 2. train_mlp.py to train your model   <-- The warm start happens behind the hood so don't worry.
 3. train_initial_conditions.py <-- Now you will have 12 battery models which you can evaluate.
 (The shell script please_work.sh also works.)
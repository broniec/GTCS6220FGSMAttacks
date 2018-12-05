# Georgia Tech CS 6220 FGSM Attacks
Project for CS 6220 at Georgia Tech

This project is built off of the code found here: https://github.com/utkuozbulak/pytorch-cnn-adversarial-attackshttps://github.com/utkuozbulak/pytorch-cnn-adversarial-attacks

FGSM attacks are manipulations (through noise functions) of the input to a neural network to alter the results. The input may be undifferentiable by humans or it may be noticed by humans but still classified correctly. FGSM are white box attacks which take advantage of the gradient and can either be untargeted (attempt to classify the input incorrectly) or targeted (attempt to classify the input as a specific other class).

Our project consisted of attacking numerous models across 2 datasets and trying numerous approaches to defend against these attacks. We found most success in applying additional top layer models to the neural network to search for patterns across the output weights. Of these, k Means and rbf-kernel svm were the most successful at correctly classifying these images despite the introduced noise.

This area needs much further exploration as neural networks are integrated into more systems we rely on. Though these attacks can be very laborious to conduct (~24 hours with Resnet, multiple days with Densenet), they are one of many approaches hackers may use to disrupt services.

This project was completed in Python 3 with PyTorch & Sklearn used for all learning models. While not necessary, CUDA is strongly recommended. The semeion results have been included but the ImageNet ones were simply too large to upload to GitHub. They can be found here: https://drive.google.com/open?id=1EDFV4moacQLd29L9uIDeYvv0l2KpiVmt. The best place to get started with this project would be `src\semeion\experiment.py` which will conduct the attack with or without the various defense approaches on the Semeion dataset. This also includes the code we used for originally building each of the various models.

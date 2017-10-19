# Noorani-Qaeda-by-Speech-Recognition
This project basically is an attempt to learn Noorani Qaeda using Deep learning. Currently we are using this CNN for audio classification.
https://github.com/drscotthawley/audio-classifier-keras-cnn

# Dataset
We have a dataset of 3080 audios belonging to 28 classes. Classes included are given below.

'qaaf', 'sa', 'seen', 'zal', 'jeem', 'sheen', 'ba', 'zuaa', 'meem', 'suad', 'noon', 'tuaa', 'ta', 'yaa'
'raa', 'zaa', 'kaaf', 'kha', 'ayn', 'ha', 'laam', 'waaw', 'haaa', 'zuad', 'ghayn', 'dal', 'alif', 'faa'

Out of these 3080 audios, 2743 are artificially generated by increasing and decreasing the sample rate of audios.

# Preprocessing
We did 2 types of prerocessing:
1) Silence Removal
2) Noise Removal

We achieved this task by using SOX library.
https://github.com/rabitt/pysox

Some results are :

![Before Removing Noise](https://github.com/asad1996172/Noorani-Qaeda-by-Speech-Recognition/edit/master/plots/before_removing_noise.png)
![After Removing Noise](https://github.com/asad1996172/Noorani-Qaeda-by-Speech-Recognition/edit/master/plots/after_removing_noise.png)
![Before Removing Silence](https://github.com/asad1996172/Noorani-Qaeda-by-Speech-Recognition/edit/master/plots/before_removing_silence.png)
![After Removing Silence](https://github.com/asad1996172/Noorani-Qaeda-by-Speech-Recognition/edit/master/plots/after_removing_silence.png)

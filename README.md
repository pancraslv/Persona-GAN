# Persona-GAN

## Environments
```
python==3.6
torch==1.2.0
```

## Dataset
We use ''train_self_original_no_cands.txt'' and ''valid_self_original_no_cands.txt'' selected from ''Persona-Chat'' for train/validate and test. You can download from [here](https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/convai2).

For data preparation detail, please refer example files in ''./data/demo.train''.

## How to run
### train
```
python network.py
```
You can specify other hyper-parameters, please refer to ''./network.py''.

### test
```
python network.py --test --ckpt models/yuor.model --use_posterior False
```
note that ''yuor.model'' is the model name you want to test, you can download our model checkpoint file [here](https://1drv.ms/u/s!AmgtOa4G3b9msI1JN8HAN7ECEySYMA?e=a6p6vQ).

### example
Here are some examples generated by our model.
```
your persona: eating is something i do when i'm bored.
your persona: i like doing the treadmill and rowing machine.
your persona: i have short hair.
your persona: two dogs live with me.
your persona: i go to the gym regularly.

hello , how are you doing tonight ?

persona-gan: i am doing well , just got back from a run . how are you ?

ground truth:  great just got back from the gym i go twice a day every day .
```
```
your persona: in my spare time i do volunteer work.
your persona: i enjoy being around people.
your persona: i am a professional wrestler.
your persona: i volunteer in a homeless shelter.

hello ! you like tupac or new kids on the block ?
i liked new kids on the block when i was a kid in the 90s !
i was born in 1981 . i take dance lessons too .
i cannot dance at all . but i can wrestle !
i can not wrestle . i'm afraid i'd break something .
i , being a professional wrestler , have broken a few bones . . .
i bet . is it really dangerous , like portrayed on tv ?
no , it is all fake . you only get hurt if you aren't paying attention .
i would get hurt a lot , with my adhd
i see , do you take any medication for that ?
no , it is entirely diet controlled .
wow , so you like avoid certain foods or something ?
i do . i had to go to a nutritionist . i also take fish oil .
do you think it is working ?
i do . what else do you like to do ?

persona-gan: i like to work at the animal shelter .

ground truth: i volunteer at a shelter . i was homeless once , i want to give back
```

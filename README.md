# FCN
Pytorch Implementation of FCN

## Train
```
cd FCN
python train_fcn32s.py -g 0
```
I trained it on a single GTX 1080 Ti, and it took about 4 hours.

## Test
```
cd FCN
python evaluate.py modelpath -g 0
```
|            | Acc   | Acc class | Mean IU | FWAV ACC |
|  ----      | ----  | ----      | ----    | ----     |
| FCN-32s Origin | 90.5  | 76.5      | 63.6    | 83.5    |
| FCN-32s VGG16  | 89.08 | 71.31     | 58.30   | 81.40   |
| FCN-16s Origin | 91.0  | 78.1      | 65.0    | 84.3    |
| FCN-16s        | 89.63 | 73.74     | 59.95   | 82.21   |

The results are worse than the origin models probably because I just fine-tuned each model for only 100000 iterations and used the different pretained backbone(vgg16).

## Todo
Add FCN-8s

## Acknowledgement
This repo is a fork of https://github.com/wkentaro/pytorch-fcn.
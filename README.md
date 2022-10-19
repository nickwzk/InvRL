# Invariant Representation Learning for Multimedia Recommendation

This our official implementation via Pytorch for the paper:

>Xiaoyu Du, Zike Wu, Fuli Feng, Xiangnan He and Jinhui Tang. Invariant Representation Learning for Multimedia Recommendation. In ACM MM`22, October 10–14, 2022, Lisboa, Portugal.

# Requirements

* Python==3.8.10
* Pytorch==1.11.0+cu113
* numpy, scipy, argparse, logging, sklearn, tqdm

# Example to Run the Codes
First, we run UltraGCN for pretraining：
```bash
python main.py --model UltraGCN
```
After that, we run the InvRL model:
```bash
python main.py --model InvRL
```

#1.定义评估函数evaluate()
#2.解析输入的参数（目前打算固定参数）
#3.打印当前使用的参数
#4.解析评估数据集的路径和class_names（这个也可以固定？）
#5.创建model
#6.加载模型权重
#7.调用evaluate()得到评估结果
#8.打印每一种class的评估结果ap
#9.打印所有class的平均评估结果mAP

from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datatime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

#1.定义评估函数evaluate()
def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
  #输入模型model，拟评估数据集地址valid_path，iou_thres阈值，conf_thres阈值，nms_thres阈值，img_size，batch_size
  model.eval() #设置为验证模式

  #加载数据
  dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
  dataloader = torch.utils.data.DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=1,
      collate_fn=dataset.collate_fn
  )

  Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

  labels = []
  sample_metrics = [] #list of tuples(TP, confs, pred)

  #评估第batch_i批
  for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
    
    labels += targets[:, 1].tolist() #提取标签
    #targets：(batch_size, 6)，其中6分别为num, cls, center_x, center_y, widht, height
    #其中num指第num个图片

    #Rescale target
    targets[:, 2:] = xywh2xyxy(targets[:, 2:]) #转换为左上右下形式
    targets[:, 2:] *= img_size #调整为原图大小

    imgs = Variable(imgs.type(Tensor), requires_grad=False) #输入图片组成tensor

    with torch.no_grad():
      outputs = model(imgs) #输入图片喂入model，得到outputs
      outputs = non_max_suppression(
          outputs,
          conf_thres=conf_thres,
          nms_thres=nms_thres,
      ) #outputs进行NMS得到最终结果
    
    sample_metrics += get_batch_statistics(
      outputs,
      targets,
      iou_threshold=iou_thres
    ) #评估一个batch样本的性能

  #Concatenate sample statistics
  true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
  precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

  return precision, recall, AP, f1, ap_class #返回一个batch_size的评估指标


if __name__ == "__main__":
  #2.解析输入的参数（目前打算固定参数）
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
  parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="模型配置文件路径")
  parser.add_argument("--data_config", type=str, default="config/coco.data", help="待检测图像路径")
  parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="模型权重路径")
  parser.add_argument("--class_path", type=str, default="data/coco.names", help="类别标签路径")
  parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
  parser.add_argument("--conf_thres", type=float, default=0.001, help="目标检测结果置信度阈值")
  parser.add_argument("--nms_thres", type=float, default=0.5, help="NMS非极大值抑制阈值")
  parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
  parser.add_argument("--img_size", type=int, default=416, help="image size")
  opt = parser.parse_args()

  #3.打印当前使用的参数
  print(opt)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  #4.解析评估数据集的路径和class_names（这个也可以固定？）
  data_config = parse_data_config(opt.data_config)
  valid_path = data_config["valid"]
  class_names = load_classes(data_config["names"])

  #5.创建model
  model = Darknet(opt.model_def).to(device)

  #6.加载模型的权重
  if opt_weights_path.endswith(".weights"):
    #加载darknet的权重
    model.load_darknet_weights(opt.weights_path)
  else:
    #加载checkpoint的权重
    model.load_state_dict(torch.load(opt.weights_path))
  
  print("计算mAP……")

  #7.调用evaluate()得到评估结果
  precision, recall, AP, f1, ap_class = evaluate(
      model,
      path=valid_path,
      iou_thres=opt.iou_thres,
      conf_thres=opt.conf_thres,
      nms_thres=opt.nms_thres,
      img_size=opt.img_size,
      batch_size=8
  )

  #8.打印每一种class的评估结果ap
  print("Average Precisions:")
  for i, c in enumerate(ap_class):
    print(f" + Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

  #9.打印所有class的平均评估结果mAP
  print(f"mAP: {AP.mean()}")

#1.解析命令行输入的各种参数，如果没有就用默认值
#2.打印出当前的各种参数
#3.创建model
#4.加载model的权重
#5.加载测试图像
#6.加载data/coco.names中的类别名称
#7.计算出batch中所有图片的地址img_paths和对应的检测结果detections
#8.为detections中每个类别的物体选择一种颜色，把检测到的bboxes画到图上

from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


if __name__ == "__main__":
  #1.解析命令行输入的各种参数，如果没有就用默认值
  parser = argparse.ArgumentParser()
  parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
  parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
  parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="模型配置文件路径")
  parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="模型权重路径")
  parser.add_argument("--class_path", type=str, default="data/coco.names", help="类别标签路径")
  parser.add_argument("--conf_thres", type=float, default=0.001, help="目标检测结果置信度阈值")
  parser.add_argument("--nms_thres", type=float, default=0.5, help="NMS非极大值抑制阈值")
  parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
  parser.add_argument("--img_size", type=int, default=416, help="image size")
  parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
  opt = parser.parse_args()

  #2.打印当前使用的参数
  print(opt)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #有则用gpu

  os.makedirs("output", exist_ok=True) #创建多级目录

  #3.创建model
  model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
  #调用darknet模型，parse_model_config，解析模型参数，生成模型参数列表，
  #调用create_modules，根据模型参数列表会生成相应的
  #convolutional, maxpool, upsample, route, shortcut, yolo层

  #4.加载模型的权重
  if opt.weights_path.endswith(".weights"):
    #加载darknet的权重
    model.load_darknet_weights(opt.weights_path)
  else:
    #加载checkpoint的权重
    model.load_state_dict(torch.load(opt.weights_path))

  model.eval() #设置为评估模式

  #5.加载测试图像
  dataloader = DataLoader(
      ImageFolder(opt.image_folder, img_size=opt.img_size),
      batch_size=opt.batch_size,
      shuffle=False,
      num_workers=opt.n_cpu,
  )

  #6.加载data/coco.names中的类别名称
  classes = load_classes(opt.class_path)
  
  Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

  imgs = [] #储存图片
  img_detections = [] #储存每个图像索引的检测结果

  print("\nPerforming object detection:")
  prev_time = time.time()

  #7.算出batch中所有图片的地址img_paths和检测结果detections
  for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    #Configure input
    input_imgs = Variable(input_imgs.type(Tensor))

    #Get detection
    with torch.no_grad(): #torch.no_grad()中的数据不需要计算梯度，也不会进行反向传播
      detections = model(input_imgs) #通过Darknet的forward()函数得到检测结果yolo_outputs
      #非极大值抑制
      detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

    #Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

    #保存图片及检测结果
    imgs.extend(img_paths) #imgs用来储存image paths
    img_detections.extend(detections) #img_detections用来储存detections for each image index
  #得到检测结果，检测部分结束

  #下面是打开对应图片，把检测到的bboxes画到图上
  #Bounding-box colors选一种bbox颜色
  cmap = plt.get_cmap("tab20b")
  colors = [cmap(i) for i in np.linspace(0, 1, 20)]

  print("\nSaving images:")

  #8.为每个类别的物体选择一种颜色，把检测到的bboxes画到图上
  for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
    print("(%d) Image: '%s'" % (img_i, path))

    #Create plot
    img = np.array(Image.open(path))
    plt.figure()
    fig, ax = plt.subplots(1) #由ax获取当前坐标轴
    ax.imshow(img)

    #Draw bounding boxes and labels of detections
    if detections is not None: #有检测结果时才需要画出来
      #Rescale boxes to original image
      
      #detections结果扩展到原图大小，检测时输入图像大小为416*416
      detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
      
      #返回参数数组中所有不同的值，并按照从小到大排序可选参数
      unique_labels = detections[:, -1].cpu().unique()
      n_cls_preds = len(unique_labels) #标签的个数

      #在colors中，随机挑选出标签数n_cls_preds种，为每一类物体分配一种颜色
      bbox_colors = random.sample(colors, n_cls_preds)

      #detections中用的是左上右下点
      for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
        print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
        bow_w = x2 - x1
        bow_h = y2 - y1
        
        #依据预测的类，查找到该用哪种颜色
        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]

        #创建一个线条宽度为2，对应颜色的的长方形的框
        bbox = patches.Rectangle((x1, y1), bow_w, bow_h, linewidth=2, edgecolor=color, facecolor="none")

        ax.add_patch(bbox) #Add the bbox to the plot

        #Add label
        plt.text(
            x1, y1, s=classes[int(cls_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": color, "pad": 0},
        )


    #Save generated image with detections
    plt.axis("off") #关闭坐标轴
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    filename = path.split("/")[-1].split(".")[0]
    plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0) #保存文件
    plt.close()

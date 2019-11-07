##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import sys
import time
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import utils
from net import Net, Vgg16,LapNet

from option import Options



DATASET="F:\COCO"
VGG_MODEL_DIR="F:\\vgg_model"
STYLE_FOLDER="F:\\style"
MODEL_SAVE_DIR="F:\\model_save"

LR=1e-3
CONTENT_WEIGHT=1.0
STYLE_WEIGHT=100.0
LAP_WEIGHT=10.0

# def main():
	# figure out the experiments type
	# args = Options().parse()
	# if args.subcommand is None:
	# 	raise ValueError("ERROR: specify the experiment type")
	# if args.cuda and not torch.cuda.is_available():
	# 	raise ValueError("ERROR: cuda is not available, try running on CPU")
	#
	# if args.subcommand == "train":
	# 	# Training the model
	# 	train(args)

	# elif args.subcommand == 'eval':
	# 	# Test the pre-trained model
	# 	evaluate(args)
	#
	# elif args.subcommand == 'optim':
	# 	# Gatys et al. using optimization-based approach
	# 	optimize(args)

	# else:
	# 	raise ValueError('Unknow experiment type')




def train():

	# 设定随机数种子
	np.random.seed(42)
	torch.manual_seed(42)

	torch.cuda.manual_seed(42)
	kwargs = {'num_workers': 0, 'pin_memory': False}

	#载入数据集
	transform = transforms.Compose([transforms.Scale(256),
																	transforms.CenterCrop(256),
																	transforms.ToTensor(),
																	transforms.Lambda(lambda x: x.mul(255))])
	train_dataset = datasets.ImageFolder(DATASET, transform)
	train_loader = DataLoader(train_dataset, 4, **kwargs)

	# 创建前半部分模型：multi- style generative 多风格生成网络
	style_model = Net(128)
	# 载入模型数据
	print(style_model)
	# 设定优化器
	optimizer = Adam(style_model.parameters(), LR)
	mse_loss = torch.nn.MSELoss()

	# 创建后半部分模型:用于提取特征的VGG16
	vgg = Vgg16()
	lap=LapNet()
	utils.init_vgg16(VGG_MODEL_DIR)
	vgg.load_state_dict(torch.load(os.path.join(VGG_MODEL_DIR, "vgg16.weight")))

	style_model.cuda()
	vgg.cuda()

	# 载入风格图片
	style_loader = utils.StyleLoader(STYLE_FOLDER,512)

	tbar = trange(2)

	# 训练
	for e in tbar:
		style_model.train()

		agg_lap_loss=0.
		agg_content_loss = 0.
		agg_style_loss = 0.
		count = 0

		for batch_id, (x, _) in enumerate(train_loader):
			n_batch = len(x)
			count += n_batch
			optimizer.zero_grad()
			x = Variable(utils.preprocess_batch(x))


			x = x.cuda()

			# 从这批图里取第i张作为这次模型的目标
			style_v = style_loader.get(batch_id)
			style_model.setTarget(style_v)



			# 提取这张图片的Gram（风格信息）
			style_v = utils.subtract_imagenet_mean_batch(style_v)
			features_style = vgg(style_v)
			gram_style = [utils.gram_matrix(y) for y in features_style]

			y = style_model(x)
			xc = Variable(x.data.clone())

			# 归一化操作
			y = utils.subtract_imagenet_mean_batch(y)
			xc = utils.subtract_imagenet_mean_batch(xc)

			# 提取特征(应该是在这个VGG里面添加lap_layer )
			features_y = vgg(y)
			features_xc = vgg(xc)

			f_xc_c = Variable(features_xc[1].data, requires_grad=False)

			content_loss = CONTENT_WEIGHT * mse_loss(features_y[1], f_xc_c)

			content_lap=lap(xc)
			y_lap=lap(y)



			lap_loss=LAP_WEIGHT*(2*mse_loss(content_lap,y_lap)/torch.numel(content_lap))

			style_loss = 0.
			for m in range(len(features_y)):

				gram_y = utils.gram_matrix(features_y[m])
				gram_s = Variable(gram_style[m].data, requires_grad=False).repeat(4, 1, 1, 1)

				style_loss += STYLE_WEIGHT * mse_loss(gram_y, gram_s[:n_batch, :, :])

			total_loss = content_loss + style_loss+lap_loss
			total_loss.backward()
			optimizer.step()

			agg_lap_loss+=lap_loss.data[0]
			agg_content_loss += content_loss.data[0]
			agg_style_loss += style_loss.data[0]

			if (batch_id + 1) % 500 == 0:
				mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\tlap: {:.6f}\ttotal: {:.6f}".format(
					time.ctime(),
					e + 1, count,
					len(train_dataset),
					agg_content_loss / (batch_id + 1),
					agg_style_loss / (batch_id + 1),
					agg_lap_loss /(batch_id+1),
					(agg_content_loss + agg_style_loss) / (batch_id + 1)
				)
				tbar.set_description(mesg)

			if (batch_id + 1) % (4 * 500) == 0:
				# save model
				style_model.eval()
				style_model.cpu()
				save_model_filename = "Epoch_" + str(e) + "iters_" + str(count) + "_" + \
															str(time.ctime()).replace(' ', '_') + "_" + str(
					CONTENT_WEIGHT) + "_" + str(STYLE_WEIGHT)+"_" + str(LAP_WEIGHT) + ".model"
				save_model_path = os.path.join(MODEL_SAVE_DIR, save_model_filename)
				torch.save(style_model.state_dict(), save_model_path)
				style_model.train()
				style_model.cuda()
				tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

	# save model
	style_model.eval()
	style_model.cpu()
	save_model_filename = "Final_epoch_" + str(2) + "_" + \
												str(time.ctime()).replace(' ', '_') + "_" + str(
		CONTENT_WEIGHT) + "_" + str(STYLE_WEIGHT) + ".model"
	save_model_path = os.path.join(MODEL_SAVE_DIR, save_model_filename)
	torch.save(style_model.state_dict(), save_model_path)

	print("\nDone, trained model saved at", save_model_path)








if __name__ == "__main__":
	train()
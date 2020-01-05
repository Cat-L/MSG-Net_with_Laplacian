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
from tqdm import  trange

import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision import transforms

import utils
from net import Net,LapNet,VGG16_from_pth




DATASET="L:\COCO"
VGG_MODEL_DIR="L:/vgg_model"
STYLE_FOLDER="L:/style"
MODEL_SAVE_DIR="L:/model_save"

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


	try:
		if not os.path.exists(VGG_MODEL_DIR):
			os.makedirs(VGG_MODEL_DIR)
		if not os.path.exists(MODEL_SAVE_DIR):
			os.makedirs(MODEL_SAVE_DIR)
	except OSError as e:
		print(e)
		sys.exit(1)

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
	style_model = Net(ngf=64)
	# 载入模型数据
	print(style_model)
	# 设定优化器
	optimizer = Adam(style_model.parameters(), LR)
	mse_loss = torch.nn.MSELoss()

	lap = LapNet()
	# 创建后半部分模型:用于提取特征的VGG16
	# vgg = Vgg16()
	# utils.init_vgg16(VGG_MODEL_DIR)
	# vgg.load_state_dict(torch.load(os.path.join(VGG_MODEL_DIR, "vgg16.weight")))

	vgg = torchvision.models.vgg16(pretrained=False)
	utils.init_vgg16_from_pth(VGG_MODEL_DIR)
	vgg.load_state_dict(torch.load(os.path.join(VGG_MODEL_DIR, 'vgg16-00b39a1b.pth')))


	style_model.cuda()
	vgg.cuda()
	lap.cuda()
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

			style_v=style_v.cuda()
			style_model.setTarget(style_v)



			# 提取这张图片的Gram（风格信息）
			style_v = utils.subtract_imagenet_mean_batch(style_v)
			features_style = VGG16_from_pth(vgg,style_v)
			gram_style = [utils.gram_matrix(y) for y in features_style]

			y = style_model(x)
			xc = Variable(x.data.clone())

			# 归一化操作
			y = utils.subtract_imagenet_mean_batch(y)
			xc = utils.subtract_imagenet_mean_batch(xc)

			# 提取特征(应该是在这个VGG里面添加lap_layer )
			features_y = VGG16_from_pth(vgg,y)
			features_xc = VGG16_from_pth(vgg,xc)

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

			agg_lap_loss+=lap_loss.item()
			agg_content_loss += content_loss.item()
			agg_style_loss += style_loss.item()

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
				save_model_filename = "Epoch_" + str(e) + "iters_" + str(count) + "_" +str(time.ctime()).replace(' ', '_') + "_" + str(CONTENT_WEIGHT) + "_" + str(STYLE_WEIGHT)+"_"+str(LAP_WEIGHT) + ".model"
				save_model_path = os.path.join(MODEL_SAVE_DIR, save_model_filename)
				torch.save(style_model.state_dict(), save_model_path)
				style_model.train()
				style_model.cuda()
				tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

	# save model
	style_model.eval()
	style_model.cpu()
	save_model_filename = "Final_epoch_" + str(2) + "_" +str(time.ctime()).replace(' ', '_') + "_" + str(CONTENT_WEIGHT) + "_" + str(STYLE_WEIGHT) + ".model"
	save_model_path = os.path.join(MODEL_SAVE_DIR, save_model_filename)
	torch.save(style_model.state_dict(), save_model_path)

	print("\nDone, trained model saved at", save_model_path)


def evaluate():
	print("now into evaluate")
	# for content_image_ in eachFile(CONTENT_FOLDER):
	#
	# 	content_image = utils.tensor_load_rgbimage(content_image_, size=512, keep_asp=True)
	# 	content_image = content_image.unsqueeze(0)
	#
	# 	for style_image_ in eachFile(STYLE_FOLDER):
	# 		style = utils.tensor_load_rgbimage(style_image_, size=512)
	# 		style = style.unsqueeze(0)
	# 		style = utils.preprocess_batch(style)
	#
	# 		style_model = Net(ngf=64)
	#
	# 		model = get_finalmodel(MODEL_SAVE_DIR)[-1]
	# 		model_dict = torch.load(model)
	# 		model_dict_clone = model_dict.copy()
	# 		for key, value in model_dict_clone.items():
	# 			if key.endswith(('running_mean', 'running_var')):
	# 				del model_dict[key]
	# 		style_model.load_state_dict(model_dict, False)
	#
	# 		style_model.cuda()
	# 		content_image = content_image.cuda()
	# 		style = style.cuda()
	#
	# 		style_v = Variable(style)
	#
	# 		content_image = Variable(utils.preprocess_batch(content_image))
	# 		style_model.setTarget(style_v)
	#
	# 		output = style_model(content_image)
	# 		# output = utils.color_match(output, style_v)
	# 		output_image = remix_name(content_name=content_image_, style_name=style_image_, model_name=model)
	# 		utils.tensor_save_bgrimage(output.data[0], output_image, 1)





if __name__ == "__main__":
	# train(
	print("aaa")
	evaluate()



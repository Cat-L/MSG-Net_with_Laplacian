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
from net import Net, Vgg16

from option import Options


def main():
	# figure out the experiments type
	args = Options().parse()
	if args.subcommand is None:
		raise ValueError("ERROR: specify the experiment type")
	if args.cuda and not torch.cuda.is_available():
		raise ValueError("ERROR: cuda is not available, try running on CPU")

	if args.subcommand == "train":
		# Training the model
		train(args)

	# elif args.subcommand == 'eval':
	# 	# Test the pre-trained model
	# 	evaluate(args)
	#
	# elif args.subcommand == 'optim':
	# 	# Gatys et al. using optimization-based approach
	# 	optimize(args)

	else:
		raise ValueError('Unknow experiment type')




def train(args):

	# 设定随机数种子
	check_paths(args)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	if args.cuda:
		torch.cuda.manual_seed(args.seed)
		kwargs = {'num_workers': 0, 'pin_memory': False}
	else:
		kwargs = {}

	#载入数据集
	transform = transforms.Compose([transforms.Scale(args.image_size),
																	transforms.CenterCrop(args.image_size),
																	transforms.ToTensor(),
																	transforms.Lambda(lambda x: x.mul(255))])
	train_dataset = datasets.ImageFolder(args.dataset, transform)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)

	# 创建前半部分模型：multi- style generative 多风格生成网络
	style_model = Net(ngf=args.ngf)
	# 载入模型数据
	if args.resume is not None:
		print('Resuming, initializing using weight from {}.'.format(args.resume))
		style_model.load_state_dict(torch.load(args.resume))
	print(style_model)
	# 设定优化器
	optimizer = Adam(style_model.parameters(), args.lr)
	mse_loss = torch.nn.MSELoss()

	# 创建后半部分模型:用于提取特征的VGG16
	vgg = Vgg16()
	utils.init_vgg16(args.vgg_model_dir)
	vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))

	if args.cuda:
		style_model.cuda()
		vgg.cuda()

	# 载入风格图片
	style_loader = utils.StyleLoader(args.style_folder, args.style_size)

	tbar = trange(args.epochs)

	# 训练
	for e in tbar:
		style_model.train()

		agg_content_loss = 0.
		agg_style_loss = 0.
		count = 0

		for batch_id, (x, _) in enumerate(train_loader):
			n_batch = len(x)
			count += n_batch
			optimizer.zero_grad()
			x = Variable(utils.preprocess_batch(x))

			if args.cuda:
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

			content_loss = args.content_weight * mse_loss(features_y[1], f_xc_c)

			lap_loss=args.lap_weight*mse_loss()

			style_loss = 0.
			for m in range(len(features_y)):

				gram_y = utils.gram_matrix(features_y[m])
				gram_s = Variable(gram_style[m].data, requires_grad=False).repeat(args.batch_size, 1, 1, 1)

				style_loss += args.style_weight * mse_loss(gram_y, gram_s[:n_batch, :, :])

			total_loss = content_loss + style_loss
			total_loss.backward()
			optimizer.step()

			agg_content_loss += content_loss.data[0]
			agg_style_loss += style_loss.data[0]

			if (batch_id + 1) % args.log_interval == 0:
				mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
					time.ctime(), e + 1, count, len(train_dataset),
												agg_content_loss / (batch_id + 1),
												agg_style_loss / (batch_id + 1),
												(agg_content_loss + agg_style_loss) / (batch_id + 1)
				)
				tbar.set_description(mesg)

			if (batch_id + 1) % (4 * args.log_interval) == 0:
				# save model
				style_model.eval()
				style_model.cpu()
				save_model_filename = "Epoch_" + str(e) + "iters_" + str(count) + "_" + \
															str(time.ctime()).replace(' ', '_') + "_" + str(
					args.content_weight) + "_" + str(args.style_weight) + ".model"
				save_model_path = os.path.join(args.save_model_dir, save_model_filename)
				torch.save(style_model.state_dict(), save_model_path)
				style_model.train()
				style_model.cuda()
				tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

	# save model
	style_model.eval()
	style_model.cpu()
	save_model_filename = "Final_epoch_" + str(args.epochs) + "_" + \
												str(time.ctime()).replace(' ', '_') + "_" + str(
		args.content_weight) + "_" + str(args.style_weight) + ".model"
	save_model_path = os.path.join(args.save_model_dir, save_model_filename)
	torch.save(style_model.state_dict(), save_model_path)

	print("\nDone, trained model saved at", save_model_path)


def check_paths(args):
	try:
		if not os.path.exists(args.vgg_model_dir):
			os.makedirs(args.vgg_model_dir)
		if not os.path.exists(args.save_model_dir):
			os.makedirs(args.save_model_dir)
	except OSError as e:
		print(e)
		sys.exit(1)





if __name__ == "__main__":
	main()
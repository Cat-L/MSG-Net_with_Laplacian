import os
import sys
import time
import numpy as np
from tqdm import trange
import logging
from loguru import logger

import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from torchvision import datasets
from torchvision import transforms

import utils
from net import Net, LapNet, VGG16_from_pth

DATASET = "D:\ML\COCO"
VGG_MODEL_DIR = "D:\ML\\vgg_model"
CONTENT_FOLDER = "D:\ML\images\content"
STYLE_FOLDER = "D:\ML\images\9styles"


LR = 1e-3
EPOCHOS = 2





def train(option):
    logging.basicConfig(level=logging.INFO, filename=(str("./{}_train.log".format(time.ctime())).replace(":", "_")))

    CONTENT_WEIGHT = option.content_weight
    STYLE_WEIGHT = option.style_weight
    LAP_WEIGHT =option.lap_weight

    logger.add((os.path.join(option.single_dir, "train_running.log")))
    try:
        if not os.path.exists(VGG_MODEL_DIR):
            os.makedirs(VGG_MODEL_DIR)
        if not os.path.exists(os.path.join(option.single_dir, "model_save_dir")):
            os.makedirs(os.path.join(option.single_dir, "model_save_dir"))
    except OSError as e:
        print(e)
        sys.exit(1)

    # 设定随机数种子
    np.random.seed(42)
    torch.manual_seed(42)

    torch.cuda.manual_seed(42)
    kwargs = {'num_workers': 0, 'pin_memory': True}

    # 载入数据集
    transform = transforms.Compose([transforms.Scale(256),
                                    transforms.CenterCrop(256),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))])

    train_dataset = datasets.ImageFolder(DATASET, transform)
    train_loader = DataLoader(train_dataset, 10, **kwargs)

    # 创建前半部分模型：multi- style generative 多风格生成网络
    style_model = Net(ngf=64)
    # 载入模型数据
    # print(style_model)
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

    vgg.cuda()
    lap.cuda()
    # 载入风格图片
    style_loader = utils.StyleLoader(STYLE_FOLDER, 512)

    # with open(str("./{}.log".format(time.ctime())).replace(":","_"), 'w+') as file:

    # 训练
    for e in range(EPOCHOS):
        style_model.cuda()
        style_model.train()

        agg_lap_loss = 0.
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0

        for batch_id, (x, _) in enumerate(tqdm(train_loader, dynamic_ncols=True)):

            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            x = Variable(utils.preprocess_batch(x))

            x = x.cuda()

            # 从这批图里取第i张作为这次模型的目标
            style_v = style_loader.get(batch_id)

            style_v = style_v.cuda()
            style_model.setTarget(style_v)

            # 提取这张图片的Gram（风格信息）
            style_v = utils.subtract_imagenet_mean_batch(style_v)
            features_style = VGG16_from_pth(vgg, style_v)
            gram_style = [utils.gram_matrix(y) for y in features_style]

            y = style_model(x)
            xc = Variable(x.data.clone())

            # 归一化操作
            y = utils.subtract_imagenet_mean_batch(y)
            xc = utils.subtract_imagenet_mean_batch(xc)

            # 提取特征(应该是在这个VGG里面添加lap_layer )
            features_y = VGG16_from_pth(vgg, y)
            features_xc = VGG16_from_pth(vgg, xc)

            f_xc_c = Variable(features_xc[1].data, requires_grad=False)
            # print(features_y[1].data.size(),f_xc_c.size())
            content_loss = CONTENT_WEIGHT * mse_loss(features_y[1], f_xc_c)

            content_lap = lap(xc)
            y_lap = lap(y)

            lap_loss = LAP_WEIGHT * (2 * mse_loss(content_lap, y_lap) / torch.numel(content_lap))

            style_loss = 0.
            for m in range(len(features_y)):
                gram_y = utils.gram_matrix(features_y[m])
                gram_s = Variable(gram_style[m].data, requires_grad=False).repeat(4, 1, 1, 1)

                style_loss += STYLE_WEIGHT * mse_loss(gram_y, gram_s[:n_batch, :, :])

            total_loss = content_loss + style_loss + lap_loss
            total_loss.backward()
            optimizer.step()

            agg_lap_loss += lap_loss.item()
            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % 500 == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\tlap: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(),
                    e + 1, count,
                    len(train_dataset),
                    agg_content_loss / (batch_id + 1),
                    agg_style_loss / (batch_id + 1),
                    agg_lap_loss / (batch_id + 1),
                    (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                # tqdm.set_description(mesg)
                logging.info(mesg)

            if (batch_id + 1) % (4 * 500) == 0:
                # save model
                style_model.eval()
                style_model.cpu()
                save_model_filename = "Epoch_" + str(e) + "iters_" + str(count) + "_" + str(time.ctime()).replace(
                    ' ',
                    '_').replace(
                    ":", "_") + "_" + str(CONTENT_WEIGHT) + "_" + str(STYLE_WEIGHT) + "_" + str(
                    LAP_WEIGHT) + ".model"
                save_model_path = os.path.join(os.path.join(option.single_dir, "model_save_dir"),
                                               save_model_filename)
                torch.save(style_model.state_dict(), save_model_path)
                style_model.train()
                style_model.cuda()
                # tqdm.set_description("\nCheckpoint, trained model saved at", save_model_path)
                logging.info(str("\nCheckpoint, trained model saved at" + save_model_path))

        # save model
        style_model.eval()
        style_model.cpu()
        save_model_filename = "Final_epoch_" + str(2) + "_" + str(time.ctime()).replace(' ', '_').replace(":",
                                                                                                          "_") + "_" + str(
            CONTENT_WEIGHT) + "_" + str(STYLE_WEIGHT) + "_" + str(LAP_WEIGHT) + ".model"
        save_model_path = os.path.join(os.path.join(option.single_dir, "model_save_dir"),
                                       save_model_filename)
        torch.save(style_model.state_dict(), save_model_path)

        logging.info(str("\nDone, trained model saved at" + save_model_path))

def evaluate(option):

    print("now into evaluate")

    try:

        logging.basicConfig(level=logging.INFO,
                            filename=(os.path.join(option.single_dir, str("./{}_evaluate.log".format(time.ctime())).replace(":", "_"))))
        for content_image_ in utils.eachFile(os.path.join(option.image_folder, "content")):

            print("now using content image: ", content_image_)

            content_image = utils.tensor_load_rgbimage(content_image_, size=512, keep_asp=True)
            content_image = content_image.unsqueeze(0)

            for style_image_ in utils.eachFile(os.path.join(option.image_folder, "9styles")):

                print("now using style image: ", style_image_)

                style = utils.tensor_load_rgbimage(style_image_, size=512)
                style = style.unsqueeze(0)
                style = utils.preprocess_batch(style)

                style_model = Net(ngf=64)

                model = utils.get_finalmodel(os.path.join(option.single_dir, "model_save_dir"))[0]
                model_dict = torch.load(model)
                model_dict_clone = model_dict.copy()
                for key, value in model_dict_clone.items():
                    if key.endswith(('running_mean', 'running_var')):
                        del model_dict[key]
                style_model.load_state_dict(model_dict, False)

                style_model.cuda()
                content_image = content_image.cuda()
                style = style.cuda()

                style_v = Variable(style)

                content_image = Variable(utils.preprocess_batch(content_image))
                style_model.setTarget(style_v)

                output = style_model(content_image)
                # output = utils.color_match(output, style_v)
                output_image = utils.remix_name(option.single_dir, content_name=content_image_,
                                                style_name=style_image_, )
                utils.tensor_save_bgrimage(output.data[0], output_image, 1)
                print("image ", output_image, "stylish done")

        print("done")

    except Exception as e:
        logging.error(e)
        print(e)



if __name__ == '__main__':
    train()

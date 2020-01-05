import torch
from net import Net
from torch.autograd import Variable
import utils
from utils import get_finalmodel, eachFile, remix_name
import os
import logging


def evaluate(option):
    print("now into evaluate")

    for i in range(len(option.single_test_folders)):
        try:
            logging.basicConfig(level=logging.INFO,
                                filename=(os.path.join(option.single_test_folders[i], "evaluate_running.log")))
            for content_image_ in eachFile(os.path.join(option.image_folder, "content")):
                print("now using content image: ", content_image_)
                content_image = utils.tensor_load_rgbimage(content_image_, size=512, keep_asp=True)
                content_image = content_image.unsqueeze(0)

                for style_image_ in eachFile(os.path.join(option.image_folder, "9styles")):
                    print("now using style image: ", style_image_)
                    style = utils.tensor_load_rgbimage(style_image_, size=512)
                    style = style.unsqueeze(0)
                    style = utils.preprocess_batch(style)

                    style_model = Net(ngf=64)

                    model = get_finalmodel(os.path.join(option.single_test_folders[i], "model_save_dir"))[0]
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
                    output_image = remix_name(option.single_test_folders[i], content_name=content_image_,
                                              style_name=style_image_, )
                    utils.tensor_save_bgrimage(output.data[0], output_image, 1)
                    print("image ", output_image, "stylish done")
            print("done")
        except Exception as e:
            logging.error(e)
            print(e)
            continue

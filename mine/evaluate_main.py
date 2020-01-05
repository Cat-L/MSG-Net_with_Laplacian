DATASET = "F:\COCO"
VGG_MODEL_DIR = "F:/vgg_model"
STYLE_FOLDER = "F:\images\9styles"
CONTENT_FOLDER = "F:\images\content"
MODEL_SAVE_DIR = "F:/model_save"

import torch
from net import Net
from torch.autograd import Variable
import utils
from utils import get_finalmodel, eachFile, remix_name


def evaluate():
    print("now into evaluate")
    for content_image_ in eachFile(CONTENT_FOLDER):
        print("now using content image: ", content_image_)
        content_image = utils.tensor_load_rgbimage(content_image_, size=512, keep_asp=True)
        content_image = content_image.unsqueeze(0)

        for style_image_ in eachFile(STYLE_FOLDER):
            print("now using style image: ", style_image_)
            style = utils.tensor_load_rgbimage(style_image_, size=512)
            style = style.unsqueeze(0)
            style = utils.preprocess_batch(style)

            style_model = Net(ngf=64)

            model = get_finalmodel(MODEL_SAVE_DIR)[-1]
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
            output_image = remix_name(image_folder=".\\"+model.split(".")[0]+'_output',content_name=content_image_, style_name=style_image_)
            utils.tensor_save_bgrimage(output.data[0], output_image, 1)
            print("image ", output_image, "stylish done")
    print("done")


if __name__ == "__main__":
    evaluate()

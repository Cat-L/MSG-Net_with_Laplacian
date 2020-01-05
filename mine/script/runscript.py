from script.lab_evaluate import evaluate
from script.lab_train import train
import os
import time


class Option:
    save_path = None

    class single_one:
        def __init__(self,weights,save_path):
            self.content_weight=weights[0]
            self.style_weight=weights[1]
            self.lap_weight=weights[2]

            path_=os.path.join(save_path, str("{}_{}_{}_{}".format(
                self.content_weight,
                self.style_weight,
                self.lap_weight,
                str(time.ctime()).replace(":", "_").replace(" ","_"))))
            os.makedirs(path_)

            self.single_dir=path_

    def __init__(self):
        self.weight_list = [
            [100, 100, 50],
            [100, 100, 100],
            [80,120,10],
                            ]
        self.main_dir=str("D:\ML\\")

        self.data_set = str("D:\ML\COCO")
        self.vgg_model_folder = str("D:\ML\\vgg_model")
        self.image_folder = str("D:\ML\images\\")

        # the order of the list : content list ,style list ,lap list

        self.save_path = str(self.main_dir+"{}".format(str(time.ctime()).replace(":", "_").replace(" ","_")))
        os.mkdir(self.save_path)
        self.single_test=[]
        for single in self.weight_list:
            self.single_test.append(self.single_one(single,self.save_path))


if __name__ == '__main__':
    option = Option()
    for i in option.single_test:
        train(i)
        evaluate(i)

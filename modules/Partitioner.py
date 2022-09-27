import torch
import logging
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# 定义判别器，用于输出需要添加图像的概率
class Estimator(torch.nn.Module):
    '''
    :param bert_model: 将所需要的bert模型传入进来
    :param resnet_model: 将需要的resnet模型传入进来
    '''
    def __init__(self, CLIPModel):
        super(Estimator, self).__init__()
        self.CLIPModel = CLIPModel
        self.classfier = torch.nn.Linear(512, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.CLIPModel.eval()

    def forward(self, input_ids, input_mask, pixel_values):
        with torch.no_grad():
            outputs = self.CLIPModel(input_ids=input_ids, pixel_values=pixel_values, attention_mask=input_mask)
            text_embeds = outputs.text_embeds
            image_embeds = outputs.image_embeds
            final_representation = text_embeds * image_embeds
        final_representation = self.classfier(final_representation)
        self.prob = self.sigmoid(final_representation)

        return self.prob

class Sampler(object):
    def __init__(self, probs):
        self.probs = probs

    def get_index(self, threshold):
        n = self.probs.shape[0]
        select_idx = []
        unselect_idx = []
        action_holder = []
        select_num = int(n * threshold)
        unselect_num = n - select_num
        temp_probs = self.probs.reshape(-1)
        sorted_result, indeices = torch.sort(temp_probs)
        indeices = indeices.tolist()

        unselect_idx.extend(indeices[:unselect_num])
        select_idx.extend(indeices[unselect_num:])

        action_holder = [1] * n

        for i, ind in enumerate(indeices):
            if (i <unselect_num):
                action_holder[ind] = 0
        return select_idx, unselect_idx, action_holder


    def get_result_index(self, threshold):
        n = self.probs.shape[0]
        select_idx = []
        unselect_idx = []
        select_num = int (n * threshold)
        unselect_num = n - select_num
        temp_probs = self.probs.reshape(-1)
        sorted_result, indeices = torch.sort(temp_probs)
        indeices = indeices.tolist()

        unselect_idx.extend(indeices[:unselect_num])
        select_idx.extend(indeices[unselect_num:])

        return select_idx, unselect_idx

    def get_result_thredshold(self, threshold):
        select_idx = []
        unselect_idx = []
        probs = self.probs.tolist()
        for i, p in enumerate(probs):
            if p > threshold:
                select_idx.append(i)
            else:
                unselect_idx.append(i)

        return select_idx, unselect_idx

    # It has better performance, but is more difficult to converge!
    def select_by_pros(self):
        all_score = self.probs
        n = all_score.shape[0]
        select_idx = []
        unselect_idx = []
        action_holder = []
        for i in range(n):
            key = random.random()
            if key < all_score[i].item():
                select_idx.append(i)
                action_holder.append(1)
            else:
                unselect_idx.append(i)
                action_holder.append(0)
        return select_idx, unselect_idx, action_holder

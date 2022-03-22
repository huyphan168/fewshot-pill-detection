import pandas as pd
import os
import json
import PIL
from PIL import Image, ExifTags
from glob import glob
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import shutil
import numpy as np
from scipy.sparse.construct import rand
from tqdm import tqdm
import yaml
import pickle as pkl

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


s_path = "/home/vishc1/projects/huy/fewshot_pill_detection/pills_image_20220106"
datasets_path = "/home/vishc1/datasets/vaipe"

if not os.path.exists(datasets_path):
    os.mkdir(datasets_path)
    for s_ in ["train", "test"]:
            os.mkdir(os.path.join(datasets_path,s_))
            for s in ["images", "labels"]:
                os.mkdir(os.path.join(datasets_path, s_, s))
# if not os.path.exists(model_path):
#     os.mkdir(model_path)
# if not os.path.exists(model_path + "/emed"):
#     os.mkdir(model_path+ "/emed")
#     os.mkdir(model_path + "/emed" + "/images")
#     os.mkdir(model_path + "/emed" + "/labels")
#     for s_ in ["train","val","test"]:
#         for s in ["images", "labels"]:
#             os.mkdir(os.path.join(model_path, "emed", s, s_))

def name2id(label_names):
    return {name: i for i, name in enumerate(label_names)}

def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s

def main():
    Label_names = []
    for path in os.listdir(s_path + "/json"):
        with open(os.path.join(s_path, "json", path), "r") as f:
            annotation = json.load(f)
            img_path = annotation["path"]
            boxes = annotation["boxes"]
            for box in boxes:
                Label_names.append(box["label"])
    Label_names = set(sorted(Label_names))
    # ids = name2id(Label_names)
    dfs = []
    for path in os.listdir(s_path + "/json"):
      if True:
        with open(s_path + "/json/" + path, "r") as f:
            annotation = json.load(f)
            img_path = annotation["path"]
            boxes = annotation["boxes"]
            img = Image.open(s_path + "/pics/" + img_path)
            w,h = exif_size(img)
            x_mid, y_mid, w_box, h_box, label = [], [], [], [], []
            for box in boxes: 
                x_mid.append((box["x"]/w + box["x"]/w + box["w"]/w)/2)
                y_mid.append((box["y"]/h + box["y"]/h + box["h"]/h)/2)
                w_box.append(box["w"]/w)
                h_box.append(box["h"]/h)
                if box["label"][:2] == "a_":
                  label.append(box["label"][2:])
                else:
                  label.append(box["label"])
    
            annotation_dict = {"image_id": path[:-5], "label":label, "x_mid":x_mid, "y_mid":y_mid, "w": w_box, "h": h_box}
            df = pd.DataFrame(annotation_dict)
            dfs.append(df)

    data_df = pd.concat(dfs, axis=0)
    data_df = pd.concat(dfs, axis=0)
    vl_ct = dict(data_df.label.value_counts())
    base_names = [name for name in Label_names if vl_ct[name] >= 100]
    base_names = sorted(base_names)
    #Few shot pills are the ones with less than 100 samples and greater than 30 samples
    few_shot_names = [name for name in Label_names if 30 < vl_ct[name] < 100]
    eliminated_names = [name for name in Label_names if vl_ct[name] <= 30]
    print(eliminated_names)
    eliminated_ids = data_df[data_df["label"].isin(eliminated_names)].image_id.unique()
    data_df = data_df[data_df.image_id.isin([x for x in data_df.image_id.unique() if x not in eliminated_ids])]
    # few_shot_names = sorted([name for name in Label_names if name not in base_names])
    few_shot_names = sorted(few_shot_names)
    print(few_shot_names)
    Label_names_elim = base_names + few_shot_names
    ids = name2id(sorted(Label_names_elim))
    data_df["label_id"] = np.zeros(len(data_df))
    data_df["label_id"] = data_df.label.map(ids)

    with open(datasets_path + "/name2id.pkl", "wb") as f:
        pkl.dump(ids, f)
    with open(datasets_path + "/base_names.pkl", "wb") as f:
        pkl.dump(base_names, f)
    with open(datasets_path + "/few_shot_names.pkl", "wb") as f:
        pkl.dump(few_shot_names, f)
    
    print("We have number of base classes:", len(base_names))
    print("We have number of few shot classes:", len(few_shot_names))
    image_ids = data_df.image_id.unique()
    y = {id: [] for id in image_ids}
    for image_id in tqdm(image_ids):
        df = list(data_df[data_df['image_id'] == image_id]['label_id'])
        y[image_id] = df
    
    # msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    # msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
    # train_index,  test_val_index= [(x, y) for x,y in msss1.split(image_ids, y)][0]
    train_index, test_val_index = multilabel_balance_split(image_ids, y, Label_names_elim, 15)
    print(len([i for i in train_index if i in test_val_index]))
    train_files = image_ids[train_index]

    # test_index, val_index = [(x, y) for x,y in msss2.split(np.array(image_ids)[test_val_index], np.array(y)[test_val_index])][0]
    test_files = image_ids[test_val_index]
    print("Occurences", len([i for i in train_files if i in test_files]))
    for image_id in list(set(train_files)):
        shutil.copy(s_path + '/pics/' + image_id + '.jpg', os.path.join(datasets_path, "train", "images", image_id + ".jpg"))
        shutil.copy(s_path + '/json/' + image_id + '.json', os.path.join(datasets_path, "train", "labels", image_id + ".json"))
    for image_id in list(set(test_files)):
        shutil.copy(s_path + '/pics/' + image_id + '.jpg', os.path.join(datasets_path, "test", "images", image_id + ".jpg"))
        shutil.copy(s_path + '/json/' + image_id + '.json', os.path.join(datasets_path, "test", "labels", image_id + ".json"))


def multilabel_balance_split(img_ids, y, Label_names, shots=13):
    import random
    random.seed(20)
    img_ids = list(img_ids)
    ids_index = {id: i for i, id in enumerate(img_ids)}
    counter = {c: 0 for c in range(len(Label_names))}
    ids_test = []
    img_id_dict = {c: [id for id in img_ids if c in y[id]] for c in range(len(Label_names))}
    for cls in tqdm(range(len(Label_names))):
        if counter[cls] > shots or cls==len(Label_names)-1:
            continue
        while True:
            if len(img_id_dict[cls]) >= shots :
                sampled_ids = random.sample(img_id_dict[cls], shots)
            else:
                sampled_ids= img_id_dict[cls]
            for img in sampled_ids:
                if img in ids_test:
                    continue
                if len([c for c in y[img] if c == cls]) + counter[cls] > shots and any([counter[c] + len([k for k in y[img] if k == c])> shots-5 for c in range(len(Label_names))]):
                    continue
                # print("before:",counter[cls])
                ids_test.append(img)
                for c in y[img]:
                    counter[c] += 1
                if counter[cls] in [shots+i for i in range(-2, 17)]:
                    break
            # print(counter)
            # print(cls)
            # print(counter[cls])
            if counter[cls] in [shots+i for i in range(-2, 17)]:
                    break
    print(counter)
    index_test = [ids_index[id] for id in ids_test]
    print(len(index_test))
    return [idx for idx in ids_index.values() if idx not in index_test],index_test
            



                
if __name__ == '__main__':
    main()
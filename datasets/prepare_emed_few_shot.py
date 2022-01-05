import argparse
import json
import os
import random
import pickle as pkl
dataset_path = "/mnt/disk1/huyvinuni/datasets/emed"
data_path = "/mnt/disk1/huyvinuni/datasets/emed/all_train"

def parse_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args

def generate_seeds(args):  
    anno = {name: [] for name in CLASS2ID.keys()}
    for json_path in os.listdir(os.path.join(data_path, "labels")):
        if json_path.endswith(".json"):
            with open(os.path.join(data_path, "labels", json_path)) as f:
                json_file = json.load(f)
                boxes = json_file['boxes']
                for box in boxes:
                    box["image_id"] = json_file["path"]
                    if box["label"] in CLASS2ID.keys():
                        anno[box["label"]].append(box)
    i = 1
    if True:
        for c in CLASS2ID.keys():
            print(c)
            #building img_ids with the keys are img_ids containing boxs of class c, the values is the the boxes of class c belonging to that image
            img_ids = {}
            for a in anno[c]:
                if a['image_id'] in img_ids:
                    img_ids[a['image_id']].append(a)
                else:
                    img_ids[a['image_id']] = [a]
            sample_shots = []
            sample_imgs = []
            for shots in [5, 10, 13]:
                while True:
                    assert len(img_ids) >= shots, print(shots, c)
                    imgs = random.sample(list(img_ids.keys()), shots)
                    for img in imgs:
                        skip = False
                        for s in sample_shots:
                            if img == s['image_id']:
                                skip = True
                                break
                        if skip:
                            continue
                        if len(img_ids[img]) + len(sample_shots) > shots:
                            continue
                        sample_shots.extend(img_ids[img])
                        sample_imgs.append(img)
                        if shots > 1:
                            if len(sample_shots) in [shots-2, shots-1, shots, shots + 1]:
                                break
                        else:
                            if len(sample_shots) == 1:
                                break
                    if shots > 1:
                        if len(sample_shots) in [shots-2,shots-1, shots, shots+1]:
                            break
                    else:
                        if len(sample_shots) == 1:
                            break
                new_data = {
                    'images': sample_imgs,
                    'annotations': sample_shots,
                }
                save_path = get_save_path_seeds(CLASS2ID[c], shots, i)
                new_data['categories'] = [c for c in CLASS2ID.keys()]
                with open(save_path, 'w') as f:
                    json.dump(new_data, f)

def get_save_path_seeds(cls, shots, seed):
    prefix = 'full_box_{}shot_{}_train'.format(shots, cls)
    save_dir = os.path.join(dataset_path, 'emedsplit', 'seed' + str(seed))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, prefix + '.json')
    return save_path

if __name__ == '__main__':
    CLASS2ID = pkl.load(open(dataset_path + '/name2id.pkl', 'rb'))
    ID2CLASS = {v: k for k, v in CLASS2ID.items()}
    generate_seeds(parse_args())
    